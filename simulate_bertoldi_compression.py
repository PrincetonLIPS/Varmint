#from comet_ml import Experiment
import time
import os
import argparse

import sys
import os
import gc

from varmintv2.solver.optimization_speed import SparseNewtonIncrementalSolver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from nma.construct_nma_shape import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie, create_static_image

import jax.experimental.host_callback as hcb

from varmintv2.utils import analysis_utils as autils
from varmintv2.utils import experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolverHCBRestartPrecondition

import scipy.optimize

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

import optax

import matplotlib.pyplot as plt

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(
    parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-c', '--ncp', type=int, default=8)
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

    experiment = None

    mat = NeoHookean2D(TPUMat)

    grid_str = "C0200 C0200 C0200 C0200 C0200 C0200 C0200 C0200\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0001 C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"
    
    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=grid_str, patch_ncp=args.ncp,
                         quad_degree=args.quaddeg, spline_degree=args.splinedeg,
                         material=mat)

    init_radii = np.concatenate(
        (
            generate_bertoldi_radii((n_cells,), args.ncp, 0.12, -0.06),
            #generate_circular_radii((n_cells,), args.ncp),
            #generate_rectangular_radii((n_cells,), args.ncp),
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

    ref_ctrl = radii_to_ctrl_fn(init_radii)
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, max_iter=1000,
                                              step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, save_mats=100)


    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn()

    def _radii_to_ref_and_init_x(radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)
        return ref_ctrl, init_x
    
    radii_to_ref_and_init_x = jax.jit(_radii_to_ref_and_init_x)
    fixed_locs_from_dict = jax.jit(cell.fixed_locs_from_dict)

    def simulate(radii):
        ref_ctrl, current_x = radii_to_ref_and_init_x(radii)

        increment_dict = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([0.0, -5.0]),
        }

        current_x, all_xs, all_fixed_locs = optimize(current_x, increment_dict, tractions, ref_ctrl)

        #return current_x, (None, None, None)
        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    print('Starting adjoint optimization')
    curr_radii = init_radii

    for i in range(1, 3):
        iter_time = time.time()
        optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)
        print(f'Iteration {i} Time: {time.time() - iter_time}')

        if i % 2 == 0:
            print(f'Generating image and video with optimization so far.')
            #optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)

            all_velocities = np.zeros_like(all_displacements)
            all_fixed_vels = np.zeros_like(all_fixed_locs)

            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
            create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
            create_movie(cell.element, ctrl_seq, vid_path)
    
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