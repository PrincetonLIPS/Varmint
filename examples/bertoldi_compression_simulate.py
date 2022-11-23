from absl import app
from absl import flags

import time
import os
import argparse

import sys
import os
import gc

from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

from geometry.bertoldi_compression_geometry import construct_cell2D, generate_bertoldi_radii
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

from varmint.utils import analysis_utils as autils
from varmint.utils import experiment_utils as eutils
from varmint.utils.mpi_utils import rprint

import numpy as onp
import jax.numpy as np
import jax

from ml_collections import config_dict
from ml_collections import config_flags

import optax

import matplotlib.pyplot as plt

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)


eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/experiments',
            source_root='n/fs/mm-iga/Varmint/')

config = config_dict.ConfigDict({
    'ncp': 5,
    'quaddeg': 3,
    'splinedeg': 2,
    'size': 8,
    'mat_model': 'NeoHookean2D',
})

config_flags.DEFINE_config_dict('config', config)


class TPUMat(Material):
    _E = 0.07
    _nu = 0.3
    _density = 1.25


def main(argv):
    args, dev_id, local_rank = eutils.initialize_experiment(verbose=True)
    config = args.config

    if config.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)

    if config.size == 8:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif config.size == 7:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif config.size == 6:
        grid_str = "C0200 C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001 C0001\n"
    elif config.size == 5:
        grid_str = "C0200 C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001 C0001\n"
    elif config.size == 4:
        grid_str = "C0200 C0200 C0200 C0200\n"\
                   "C0000 C0000 C0000 C0000\n"\
                   "C0000 C0000 C0000 C0000\n"\
                   "C0001 C0001 C0001 C0001\n"

    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=grid_str, patch_ncp=config.ncp,
                         quad_degree=config.quaddeg, spline_degree=config.splinedeg,
                         material=mat)

    init_radii = np.concatenate(
        (
            generate_bertoldi_radii((n_cells,), config.ncp, 0.12, -0.06),
        )
    )
    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    l2g, g2l = cell.get_global_local_maps()
    ref_ctrl = radii_to_ctrl_fn(init_radii)

    if config.mat_model == 'NeoHookean2D':
        mat_params = (
            TPUMat.shear * np.ones(ref_ctrl.shape[0]),
            TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
        )
    elif config.mat_model == 'LinearElastic2D':
        mat_params = (
            TPUMat.lmbda * np.ones(ref_ctrl.shape[0]),
            TPUMat.mu * np.ones(ref_ctrl.shape[0]),
        )

    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, max_iter=1000,
                                              step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, save_mats=0, print_runtime_stats=True)

    x0 = l2g(ref_ctrl, ref_ctrl)
    rprint(f'Optimizing over {x0.size} degrees of freedom.')
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
            '2': np.array([0.0, -1.0 * config.size]),
            '96': np.array([0.0, 0.0]),
        }

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(current_x, increment_dict, tractions, ref_ctrl, mat_params)
        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    curr_radii = init_radii

    # Compiling iteration
    iter_time = time.time()
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)
    rprint(f'Compile + Solve Time: {time.time() - iter_time}')

    # Compiling iteration
    iter_time = time.time()
    #with jax.profiler.trace("/u/doktay/jax-varmint-traces-afteroptimization/", create_perfetto_trace=True):
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(curr_radii)
    rprint(f'Pure Solve Time: {time.time() - iter_time}')

    rprint(f'Generating image and video with optimization so far.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path)
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
    create_movie(cell.element, ctrl_seq, vid_path)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    app.run(main)
