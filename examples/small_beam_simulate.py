#from comet_ml import Experiment
import time
import os
import argparse

import sys
import os
import gc

from varmint.solver.optimization_speed import SparseNewtonIncrementalSolver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.solver.discretize import HamiltonianStepper
from varmint.utils.movie_utils import create_movie, create_static_image

import jax.experimental.host_callback as hcb

from small_beam_geometry import construct_beam

from varmint.utils import analysis_utils as autils
from varmint.utils import experiment_utils as eutils

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
parser.add_argument('-c', '--ncp', type=int, default=15)
parser.add_argument('-q', '--quaddeg', type=int, default=5)
parser.add_argument('-s', '--splinedeg', type=int, default=2)

parser.add_argument('--simtime', type=float, default=50.0)
parser.add_argument('--dt', type=float, default=0.5)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--comet', dest='comet', action='store_true')

parser.add_argument('--save', dest='save', action='store_true')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
    _density = 1.25


if __name__ == '__main__':
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args)
    # args.seed and args.exp_dir should be set.

    eutils.save_args(args)
    npr.seed(args.seed)

    # Nonlinear material. For linear elasticity, do LinearElastic2D(TPUMat).
    mat = NeoHookean2D(TPUMat)

    # Construct geometry (simple beam).
    beam, ref_ctrl = construct_beam(
            len_x=10, len_y=3, patch_ncp=10, spline_degree=3, material=mat)

    potential_energy_fn = beam.get_potential_energy_fn()
    potential_energy_fn = jax.jit(potential_energy_fn)
    strain_energy_fn = jax.jit(beam.get_strain_energy_fn())

    # Defines the material parameters.
    # Can ignore this. Only useful when doing optimization wrt material params.
    mat_params = (
        TPUMat.shear * np.ones(ref_ctrl.shape[0]),
        TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})

    optimizer = SparseNewtonIncrementalSolver(
            beam, potential_energy_fn, max_iter=1000,
            step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, save_mats=0)

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()
    init_x = l2g(ref_ctrl, ref_ctrl)

    optimize = optimizer.get_optimize_fn()
    fixed_locs_from_dict = jax.jit(beam.fixed_locs_from_dict)
    def simulate():
        current_x = init_x
        increment_dict = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([-1.0]),
        }

        current_x, all_xs, all_fixed_locs = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    print('Starting optimization (may be slow because of compilation).')
    iter_time = time.time()
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate()
    print(f'\tSolve time: {time.time() - iter_time}')

    print(f'Generating image and video with optimization.')
    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
    create_static_image(
            beam.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], ref_ctrl), image_path)
    ctrl_seq, _ = beam.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, ref_ctrl)
    create_movie(beam.element, ctrl_seq, vid_path)

    print(f'Finished simulation {args.exp_name}')

