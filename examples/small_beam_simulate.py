from absl import app
from absl import flags

import time
import os
import argparse

import sys
import os
import gc

from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver
from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

from geometry.small_beam_geometry import construct_beam

from varmint.utils import analysis_utils as autils
from varmint.utils import experiment_utils as eutils
from varmint.utils.mpi_utils import rprint

import numpy.random as npr
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

    # Construct geometry (simple beam).
    beam, ref_ctrl = construct_beam(
            len_x=config.len_x, len_y=config.len_y,
            patch_ncp=config.ncp, spline_degree=config.splinedeg,
            quad_degree=config.quaddeg, material=mat)

    potential_energy_fn = beam.get_potential_energy_fn()
    strain_energy_fn = jax.jit(beam.get_strain_energy_fn())

    # Defines the material parameters.
    # Can ignore this. Only useful when doing optimization wrt material params.
    mat_params = (
        TPUMat.E * np.ones(ref_ctrl.shape[0]),
        TPUMat.nu * np.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})

    optimizer = SparseNewtonIncrementalSolver(
            beam, potential_energy_fn, **config.solver_parameters)

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()
    init_x = l2g(ref_ctrl, ref_ctrl)

    optimize = optimizer.get_optimize_fn()
    def simulate():
        current_x = init_x
        increment_dict = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([-1.0]),
        }

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    rprint('Starting optimization (may be slow because of compilation).')
    iter_time = time.time()
    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate()
    rprint(f'\tSolve time: {time.time() - iter_time}')

    rprint(f'Generating image and video with optimization.')
    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    ref_config_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref.png')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_static_image(beam.element, ref_ctrl, ref_config_path)
    create_static_image(
            beam.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], ref_ctrl), image_path)
    ctrl_seq, _ = beam.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, ref_ctrl)
    create_movie(beam.element, ctrl_seq, vid_path)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    app.run(main)
