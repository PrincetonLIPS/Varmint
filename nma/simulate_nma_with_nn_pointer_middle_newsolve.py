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

from varmintv2.solver.optimization_speed import SparseNewtonSolverHCBRestartPrecondition

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

    grid_str = "C0000 C0300 C0000\n"\
               "C2000 S0000 C0040\n"\
               "C0000 C0005 C0000\n"

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
                                                         step_size=1.0, tol=1e-5, ls_backtrack=0.95, update_every=10)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn()
    #optimize = jax.jit(optimizer.get_optimize_fn(x0, (fixed_locs, tractions, ref_ctrl)))

    n_increments = 50

    @jax.jit
    def radii_to_ref_and_init_x(radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)
        return ref_ctrl, init_x
    fixed_locs_from_dict = jax.jit(cell.fixed_locs_from_dict)

    def simulate(disps, radii):
        ref_ctrl, current_x = radii_to_ref_and_init_x(radii)

        increments = disps / n_increments
        increments = increments[..., np.newaxis] * np.arange(n_increments + 1)
        increments = increments.T  # increments is (n_increments, n_boundaries)

        all_xs = []
        all_fixed_locs = []
        for increment in increments:
            fixed_displacements = {
                '99': np.array([0.0, 0.0]),
                '98': np.array([0.0, 0.0]),
                '97': np.array([0.0, 0.0]),
                '96': np.array([0.0, 0.0]),
                '2': np.array([-increment[0], 0.0]),
                '3': np.array([0.0, -increment[1]]),
                '4': np.array([-increment[2], 0.0]),
                '5': np.array([0.0, -increment[3]]),
            }
            fixed_locs = fixed_locs_from_dict(ref_ctrl, fixed_displacements)
            current_x = optimize(current_x, (fixed_locs, tractions, ref_ctrl))
            all_xs.append(current_x)
            all_fixed_locs.append(fixed_locs)

        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    p1 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([7.5, 7.5])), axis=-1) < 1e-14

    test_pts = np.array([
        [6.0, 9.0],
    ])

    test_disps = np.array([6.0, 9.0])

    def clip_fn(x):
        return np.clip(x, -2.5, 2.5)

    def tanh_clip(x):
        return np.tanh(x) * 2.5
    def nn_fn(input):
        mlp = hk.Sequential([
            hk.Linear(300), jax.nn.softplus,
            hk.Linear(300), jax.nn.softplus,
            hk.Linear(100), jax.nn.softplus,
            hk.Linear(4),   tanh_clip,
        ])

        return mlp(input)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.array([0.0, 0.0])
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    def loss_fn(all_params, displacements):
        nn_params, radii = all_params
        mat_inputs = nn_fn_t.apply(nn_params, displacements)
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        # We want our identified point (p1) at a specified location (displacements).
        return np.sum(np.abs(final_x_local[p1] - displacements)) / ref_ctrl[p1].shape[0]

    print('Starting NMA optimization')
    curr_all_params = (init_nn_params, init_radii)
    lr = 0.00005 * jax.device_count()

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(curr_all_params)
    loss_val_and_grad = jax.value_and_grad(loss_fn)

    ewa_loss = None
    ewa_weight = 0.95
    for i in range(1, 10000001):
        target_disps = onp.random.uniform(6.0, 9.0, size=(2,))
        #target_disps = test_disps

        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_all_params, target_disps)

        if ewa_loss == None:
            ewa_loss = loss
        else:
            ewa_loss = ewa_loss * ewa_weight + loss * (1 - ewa_weight)
        print(f'Iteration {i} Loss: {loss} EWA Loss: {ewa_loss} Radii Grad Norm: {np.linalg.norm(grad_loss[1])} Time: {time.time() - iter_time}')

        updates, opt_state = optimizer.update(grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)
        curr_nn_params, curr_radii = curr_all_params
        curr_radii = np.clip(curr_radii, 0.5, 0.5)
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