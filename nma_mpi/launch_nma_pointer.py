from absl import app
from absl import flags

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import time
import pickle

from ml_collections import config_flags

import experiment_utils as eutils
from mpi_utils import *

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

from jax.config import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
#import mpi4jax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from construct_nma_shape import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.utils.movie_utils import create_movie_nma, create_static_image_nma
from varmintv2.solver.optimization_speed import SparseNewtonIncrementalSolver

import optax
import haiku as hk

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/nma_mpi/experiments',
            source_root='n/fs/mm-iga/Varmint/nma_mpi')

config_flags.DEFINE_config_file('config', 'config/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
    _density = 1.25


def main(argv):
    comm = MPI.COMM_WORLD
    rprint(f'Initializing MPI with JAX. Visible JAX devices: {jax.devices()}', comm=comm)
    rprint(f'There are {len(jax.devices())} devices available.', comm=comm)
    local_rank = find_local_rank(comm)
    dev_id = local_rank % len(jax.devices())
    print(f'Using GPU {dev_id} on node {MPI.Get_processor_name()}.')

    args = FLAGS
    eutils.prepare_experiment_directories(args, comm)
    # args.seed and args.exp_dir should be set.

    config = args.config
    eutils.save_args(args, comm)
    npr.seed(config.seed)

    mat = NeoHookean2D(TPUMat)

    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=config.grid_str, patch_ncp=config.ncp,
                         quad_degree=config.quad_deg, spline_degree=config.spline_deg,
                         material=mat)

    init_radii = np.concatenate(
        (
            generate_rectangular_radii((n_cells,), config.ncp),
        )
    )

    potential_energy_fn = cell.get_potential_energy_fn()
    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(), device=jax.devices()[dev_id])

    potential_energy_fn = jax.jit(potential_energy_fn, device=jax.devices()[dev_id])
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn, device=jax.devices()[dev_id])
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn, device=jax.devices()[dev_id])

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(np.array(init_radii))
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    mat_params = (
        TPUMat.shear * np.ones(ref_ctrl.shape[0]),
        TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
    )

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, max_iter=1000,
                                              step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, dev_id=dev_id)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn()

    def _radii_to_ref_and_init_x(radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)
        return ref_ctrl, init_x
    
    radii_to_ref_and_init_x = jax.jit(_radii_to_ref_and_init_x, device=jax.devices()[dev_id])
    fixed_locs_from_dict = jax.jit(cell.fixed_locs_from_dict, device=jax.devices()[dev_id])

    def simulate(disps, radii):
        ref_ctrl, current_x = radii_to_ref_and_init_x(radii)

        increment_dict = {
            '99': np.array([0.0, 0.0]),
            '98': np.array([0.0, 0.0]),
            '97': np.array([0.0, 0.0]),
            '96': np.array([0.0, 0.0]),
            '2': np.array([-disps[0], 0.0]),
            #'3': np.array([0.0, 0.0]),
            #'3': np.array([0.0, -disps[1]]),
            '4': np.array([-disps[2], 0.0]),
            #'5': np.array([0.0, 0.0]),
            #'5': np.array([0.0, -disps[3]]),
        }

        current_x, all_xs, all_fixed_locs = optimize(current_x, increment_dict, tractions, ref_ctrl, mat_params)

        #return current_x, (None, None, None)
        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    p1 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([12.5, 12.5])), axis=-1) < 1e-14

    def tanh_clip(x):
        return np.tanh(x) * 4.0
    def nn_fn(input):
        mlp = hk.Sequential([
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(10), jax.nn.softplus,
            hk.Linear(4),  tanh_clip,
        ])

        return mlp(input)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.array([0.0, 0.0])
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    def loss_fn(all_params, displacements):
        delta = displacements - np.array([12.5, 12.5])
        nn_params, radii = all_params
        mat_inputs = nn_fn_t.apply(nn_params, delta)
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        # We want our identified point (p1) at a specified location (displacements).
        return np.sum(np.abs(final_x_local[p1] - displacements)) / ref_ctrl[p1].shape[0]

    print(f'Starting NMA optimization on device {dev_id}')

    curr_all_params = (init_nn_params, init_radii)
    if args.reload:
        print('Loading parameters.')
        with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{args.load_iter}.pkl'), 'rb') as f:
            curr_all_params = pickle.load(f)
        print('\tDone.')
        iter_num = args.load_iter
        processed_iter_num = args.load_iter
    else:
        iter_num = 0
        processed_iter_num = 0

    mpi_size = comm.Get_size()
    lr = 0.0001 * mpi_size

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)

    def pytree_reduce(comm, pytree, scale=1.0, token=None):
        raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
        #reduce_sum, token = mpi4jax.allreduce(raveled, op=MPI.SUM, comm=comm, token=token)
        reduce_sum = comm.allreduce(raveled.block_until_ready(), op=MPI.SUM)
        token = None

        return unravel(reduce_sum * scale), token

    def test_pytrees_equal(comm, pytree, token=None):
        if comm.rank == 0:
            print('Testing if parameters have deviated.')
            vtime = time.time()
        raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
        #all_params, token = mpi4jax.gather(raveled, root=0, comm=comm, token=token)
        all_params = comm.gather(raveled.block_until_ready(), root=0)
        token = None
        if comm.rank == 0:
            for i in range(mpi_size-1):
                assert np.allclose(all_params[i], all_params[i+1])
            print(f'\tVerified in {time.time() - vtime} s.')

        return token

    ewa_loss = None
    ewa_weight = 0.95

    def simulate_element(params, disp):
        loss, grad_loss = loss_val_and_grad(params, disp)
        return loss, grad_loss

    print(f'Arrived at barrier {comm.rank}', flush=True)
    #token = mpi4jax.barrier(comm=comm)
    comm.barrier()
    token = None
    for i in range(args.load_iter + 1, 10000):
        iter_time = time.time()
        target_disps = onp.random.uniform(11.0, 14.0, size=(mpi_size, 2))
        loss, grad_loss = simulate_element(curr_all_params, target_disps[comm.rank])
        avg_loss, token= pytree_reduce(comm, loss, scale=1./mpi_size, token=token)
        avg_grad_loss, token = pytree_reduce(comm, grad_loss, scale=1./mpi_size, token=token)

        if ewa_loss == None:
            ewa_loss = loss
        else:
            ewa_loss = ewa_loss * ewa_weight + avg_loss * (1 - ewa_weight)
        if comm.rank == 0:
            rprint(f'Iteration {i} Loss: {avg_loss} EWA Loss: {ewa_loss} Time: {time.time() - iter_time}', comm=comm)

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)

        if i % 10 == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            token = test_pytrees_equal(comm, curr_all_params, token=token)

            # Generate video
            if comm.rank == 0:
                test_disps = np.array([11.0, 14.0])
                test_pts = np.array([
                    [11.0, 14.0],
                ])

                rprint(f'Generating image and video with optimization so far.', comm=comm)

                curr_nn_params, curr_radii = curr_all_params
                delta = test_disps - np.array([12.5, 12.5])
                mat_inputs = nn_fn_t.apply(curr_nn_params, delta)

                optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(mat_inputs, curr_radii)

                all_velocities = np.zeros_like(all_displacements)
                all_fixed_vels = np.zeros_like(all_fixed_locs)

                image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
                vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
                create_static_image_nma(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path, test_pts)
                ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
                create_movie_nma(cell.element, ctrl_seq, vid_path, test_pts, comet_exp=None, p1=p1)

                # Pickle parameters
                rprint('Saving parameters.', comm=comm)
                with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump(curr_all_params, f)
                rprint('\tDone.', comm=comm)
        #token = mpi4jax.barrier(comm=comm, token=token)
        comm.barrier()

if __name__ == '__main__':
    app.run(main)
