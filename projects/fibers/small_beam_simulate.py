import varmint

import time
import os
import argparse

import sys
import os
import gc

from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

from geometry.small_beam_geometry import construct_beam

from varmint.utils.mpi_utils import rprint

import jax
import jax.numpy as jnp
import numpy as onp

import optax

import matplotlib.pyplot as plt


varmint.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/fibers/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/fibers/')

config = varmint.config_dict.ConfigDict({
    'ncp': 10,
    'quaddeg': 5,
    'splinedeg': 3,
    'mat_model': 'NeoHookean2D',

    'len_x': 10,
    'len_y': 3,

    'solver_parameters': {
        'tol': 1e-8,
    },
})

varmint.config_flags.DEFINE_config_dict('config', config)

class TPUMat(Material):
    _E = 0.07
    _nu = 0.3
    _density = 1.25


def main(argv):
    args, dev_id, local_rank = varmint.initialize_experiment(verbose=True)
    config = args.config

    if config.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)
    else:
        raise ValueError(f'Unknown material model: {config.mat_model}')

    # Construct geometry (simple beam).
    beam, ref_ctrl = construct_beam(
            len_x=config.len_x, len_y=config.len_y,
            patch_ncp=config.ncp, spline_degree=config.splinedeg,
            quad_degree=config.quaddeg, material=mat)

    # Defines the material parameters.
    # Can ignore this. Only useful when doing optimization wrt material params.
    mat_params = (
        TPUMat.E * jnp.ones(ref_ctrl.shape[0]),
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})

    # We would like to minimize the potential energy.
    potential_energy_fn = beam.get_potential_energy_fn()
    optimizer = SparseNewtonIncrementalSolver(
            beam, potential_energy_fn, **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    def simulate():
        current_x = l2g(ref_ctrl, ref_ctrl)
        increment_dict = {
            '1': jnp.array([0.0, 0.0]),
            '2': jnp.array([-1.0]),  # Only applied to y-coordinate.
        }

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        # Unflatten sequence to local configuration.
        ctrl_seq = cell.unflatten_sequence(
            all_xs, all_fixed_locs, ref_ctrl)
        final_x_local = g2l(current_x, all_fixed_locs[-1], ref_ctrl)

        return final_x_local, [ref_ctrl] + ctrl_seq

    rprint('Starting optimization (may be slow because of compilation).')
    iter_time = time.time()
    final_x_local, ctrl_seq = simulate()
    rprint(f'\tSolve time: {time.time() - iter_time}')

    rprint(f'Generating image and video with optimization.')

    # Reference configuration
    ref_config_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref.png')
    create_static_image(beam.element, ref_ctrl, ref_config_path)

    # Deformed configuration
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    create_static_image(beam.element, final_x_local, image_path)

    # Deformation sequence movie
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_movie(beam.element, ctrl_seq, vid_path)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
