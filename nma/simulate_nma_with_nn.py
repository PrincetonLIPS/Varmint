import pickle
from comet_ml import Experiment
import time
import os
import argparse

import sys
import os
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from construct_nma_shape import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie_nma, create_static_image_nma

import jax.experimental.host_callback as hcb

from varmintv2.utils import analysis_utils as autils
from varmintv2.utils import experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolverHCB, SparseNewtonSolverHCBRestart, SparseNewtonSolverHCBRestartPrecondition

import scipy.optimize

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

import optax
import haiku as hk

import matplotlib.pyplot as plt

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(
    parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=5)
parser.add_argument('-s', '--splinedeg', type=int, default=2)

parser.add_argument('--simtime', type=float, default=50.0)
parser.add_argument('--dt', type=float, default=0.5)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--comet', dest='comet', action='store_true')

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--strategy', choices=['ilu_preconditioning', 'superlu', 'lu'],
                    default='ilu_preconditioning')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
    _density = 1.25

class TPOMat(Material):
    _E = 0.02
    _nu = 0.47
    _density = 1.14

class SteelMat(Material):
    _E = 200.0
    _nu = 0.3
    _density = 8.0


if __name__ == '__main__':
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args)
    # args.seed and args.exp_dir should be set.

    eutils.save_args(args)
    npr.seed(args.seed)

    if args.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="gTBUDHLLNaIqxMWyjHKQgtlkW",
            project_name="general",
            workspace="denizokt",
        )
    else:
        experiment = None

    mat = NeoHookean2D(TPUMat)

    grid_str = "C1000 C0200 C0000 C0300 C0000 C0400 C0000 C0100\n"\
               "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C1001 C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"

    #grid_str = "S1000 S0200 S0000 S0300 S0000 S0400 S0000 S0100\n"\
    #           "S1000 S0000 S0000 S0000 S0000 S0000 S0000 S0000\n"\
    #           "S1000 S0000 S0000 S0000 S0000 S0000 S0000 S0000\n"\
    #           "S1000 S0000 S0000 S0000 S0000 S0000 S0000 S0000\n"\
    #           "S1000 S0000 S0000 S0000 S0000 S0000 S0000 S0000\n"\
    #           "S1000 S0000 S0000 S0000 S0000 S0000 S0000 S0000\n"\
    #           "C1001 S0001 S0001 S0001 S0001 S0001 S0001 S0001\n"

    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=grid_str, patch_ncp=args.ncp,
                         quad_degree=args.quaddeg, spline_degree=args.splinedeg,
                         material=mat)

    init_radii = np.concatenate(
        (
            #generate_bertoldi_radii((n_cells,), args.ncp, 0.12, -0.06),
            #generate_circular_radii((1,), args.ncp),
            generate_rectangular_radii((n_cells,), args.ncp),
        )
    )
    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    potential_energy_fn = jax.jit(potential_energy_fn)
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(np.array(init_radii))
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    optimizer = SparseNewtonSolverHCBRestartPrecondition(cell, potential_energy_fn, max_iter=1000,
                                                         step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn(x0, (fixed_locs, tractions, ref_ctrl))

    n_increments = 50

    @jax.jit
    def simulate(disps, radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)

        increments = disps / n_increments
        increments = increments[..., np.newaxis] * np.arange(n_increments + 1)
        increments = increments.T  # increments is (n_increments, n_boundaries)

        def sim_increment(x_prev, increment):
            fixed_displacements = {
                '1': np.array([0.0, 0.0]),
                '2': np.array([0.0, -increment[0]]),
                '3': np.array([0.0, -increment[1]]),
                '4': np.array([0.0, -increment[2]]),
            }
            fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
            new_x = optimize(x_prev, (fixed_locs, tractions, ref_ctrl))
            strain_energy = strain_energy_fn(new_x, fixed_locs, tractions, ref_ctrl)

            return new_x, (new_x, fixed_locs, strain_energy)
        
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = jax.lax.scan(sim_increment, init_x, increments)
        return final_x, (all_xs, all_fixed_locs, all_strain_energies)

    p1 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([40.0, 25.0])), axis=-1) < 1e-14
    p2 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([40.0, 15.0])), axis=-1) < 1e-14
    p3 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([40.0,  5.0])), axis=-1) < 1e-14

    test_pts = np.array([
        [42.0, 25.0],
        [39.0, 15.0],
        [40.0,  5.0],
    ])

    test_disps = np.array([2.0, -1.0, 0.0])

    def clip_fn(x):
        return np.clip(x, -2.0, 2.0)
    def nn_fn(input):
        mlp = hk.Sequential([
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(3),   clip_fn,
        ])

        return mlp(input)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.array([0.0, 0.0, 0.0])
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    def loss_fn(all_params, displacements):
        nn_params, radii = all_params
        mat_inputs = nn_fn_t.apply(nn_params, displacements)
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        return np.sum(np.abs(final_x_local[p1][..., 0] - ref_ctrl[p1][..., 0] - displacements[0])) / ref_ctrl[p1].shape[0] + \
               np.sum(np.abs(final_x_local[p2][..., 0] - ref_ctrl[p2][..., 0] - displacements[1])) / ref_ctrl[p2].shape[0] + \
               np.sum(np.abs(final_x_local[p3][..., 0] - ref_ctrl[p2][..., 0] - displacements[2])) / ref_ctrl[p3].shape[0]

    print('Starting NMA optimization')
    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    curr_all_params = (init_nn_params, init_radii)
    lr = 0.1

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(curr_all_params)
    for i in range(1, 10000001):
        target_disps = onp.random.uniform(-1.0, 1.0, 3)

        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_all_params, target_disps)
        print(f'Iteration {i} Loss: {loss} Radii Grad Norm: {np.linalg.norm(grad_loss[1])} Time: {time.time() - iter_time}')
        gc.collect()

        updates, opt_state = optimizer.update(grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)
        curr_nn_params, curr_radii = curr_all_params
        curr_radii = np.clip(curr_radii, 0.2, 0.8)
        curr_all_params = (curr_nn_params, curr_radii)

        if i % 10 == 0:
            print(f'Generating image and video with optimization so far.')
            curr_nn_params, curr_radii = curr_all_params
            mat_inputs = nn_fn_t.apply(curr_nn_params, test_disps)

            optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(mat_inputs, curr_radii)

            all_velocities = np.zeros_like(all_displacements)
            all_fixed_vels = np.zeros_like(all_fixed_locs)

            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
            create_static_image_nma(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path, test_pts)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
            create_movie_nma(cell.element, ctrl_seq, vid_path, test_pts, comet_exp=None)

            # Pickle parameters
            print('Saving parameters.')
            with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                pickle.dump(curr_all_params, f)
            print('\tDone.')
    
    final_loss = loss_fn(curr_all_params, test_disps)
    print(f'Final loss: {final_loss}')
    print(f'Finished simulation {args.exp_name}')

"""
    print('Simulating initial radii')
    sim_time = time.time()
    curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = simulate(init_disps, init_radii)
    print(f'Finished sim in {time.time() - sim_time} seconds.')
    #print(f'Loss is: {loss_fn(init_radii)}')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_static_image_nma(cell.element, g2l(curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(init_radii)), image_path, target_pts)

    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy.dat'), "w")
    onp.savetxt(scriptFile_se, all_strain_energies,"%f")
    #scriptFile_in = open(os.path.join(args.exp_dir, f'increments.dat'), "w")
    #onp.savetxt(scriptFile_in, increments,"%f")
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(init_radii))
    create_movie_nma(cell.element, ctrl_seq, vid_path, target_pts, comet_exp=None)

    #plt.plot(increments, all_strain_energies)
    #plt.savefig(os.path.join(args.exp_dir, f'strain_energy_graph-{args.exp_name}.png'))
"""