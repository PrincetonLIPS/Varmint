import time
import os
import argparse

import sys
import os
import gc

from varmint.solver.optimization_speed import SparseNewtonIncrementalSolver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from bertoldi_compression_geometry import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

from varmint.utils import analysis_utils as autils
from varmint.utils import experiment_utils as eutils

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
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=3)
parser.add_argument('-s', '--splinedeg', type=int, default=2)
parser.add_argument('--size', type=int, default=8)

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
    _nu = 0.3
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

    if args.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
    elif args.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)

    if args.size == 8:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif args.size == 7:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif args.size == 6:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif args.size == 5:
        grid_str = "C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001\n"
    elif args.size == 4:
        grid_str = "C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001\n"

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

    if args.mat_model == 'NeoHookean2D':
        mat_params = (
            TPUMat.shear * np.ones(ref_ctrl.shape[0]),
            TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
        )
    elif args.mat_model == 'LinearElastic2D':
        mat_params = (
            TPUMat.lmbda * np.ones(ref_ctrl.shape[0]),
            TPUMat.mu * np.ones(ref_ctrl.shape[0]),
        )

    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, max_iter=1000,
                                              step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, save_mats=0, print_runtime_stats=True)

    x0 = l2g(ref_ctrl, ref_ctrl)
    print(f'Optimizing over {x0.size} degrees of freedom.')
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
            '2': np.array([0.0, -1.0 * args.size]),
            '96': np.array([0.0, 0.0]),
        }

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(current_x, increment_dict, tractions, ref_ctrl, mat_params)
        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    curr_radii = init_radii

    # Compiling iteration
    iter_time = time.time()
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)
    print(f'Compile + Solve Time: {time.time() - iter_time}')

    # Compiling iteration
    iter_time = time.time()
    #with jax.profiler.trace("/u/doktay/jax-varmint-traces-afteroptimization/", create_perfetto_trace=True):
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)
    print(f'Pure Solve Time: {time.time() - iter_time}')

    print(f'Generating image and video with optimization so far.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path)
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
    create_movie(cell.element, ctrl_seq, vid_path)

    print(f'Finished simulation {args.exp_name}')
