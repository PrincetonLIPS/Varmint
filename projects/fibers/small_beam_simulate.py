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
    'quaddeg': 5,
    'mat_model': 'NeoHookean2D',

    'fidelity': 50,
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


def get_energy_density_fn(element, material):
    """Return a function that computes the pointwise energy density.

    # TODO(doktay): This function is slightly hacky... maybe Varmint
    # design should change a bit so that this is easier to do.
    """

    def deformation_fn(point, ctrl):
        return element.get_map_fn_fixed_ctrl(ctrl)(point)
    def jacobian_u_fn(point, ctrl):
        return element.get_map_jac_fn(point[onp.newaxis, :])(ctrl)
    def jacobian_ctrl_fn(point, ctrl):
        return element.get_ctrl_jacobian_fn(point[onp.newaxis, :])(ctrl)
    vmap_energy_fn = jax.vmap(material.get_energy_fn(), in_axes=(0,))

    defgrads_fn = jax.vmap(
        lambda A, B: jnp.linalg.solve(B.T, A.T).T,
        in_axes=(0, 0),
    )

    def energy_density(point, def_ctrl, ref_ctrl):
        # Copied from varmint.physics.energy
        def_jacs = jacobian_u_fn(point, def_ctrl)
        ref_jacs = jacobian_u_fn(point, ref_ctrl)

        defgrads = defgrads_fn(def_jacs, ref_jacs)

        return vmap_energy_fn(defgrads) * 1e3

    return energy_density


def get_global_energy_fn(energy_density_fn, find_patch, l2g, ref_ctrl, dirichlet_ctrl):
    def global_energy_density(point, global_coords, ref_ctrl, dirichlet_ctrl):
        local_coords = g2l(global_coords, dirichlet_ctrl, ref_ctrl)

        # Figure out which patch the point belongs to.
        # Transform `point` to [0, 1] x [0, 1]
        patch_index, point = find_patch(point)

        return energy_density_fn(point, local_coords[patch_index], ref_ctrl[patch_index])


def main(argv):
    args, dev_id, local_rank = varmint.initialize_experiment(verbose=True)
    config = args.config

    if config.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)
    else:
        raise ValueError(f'Unknown material model: {config.mat_model}')

    # Define implicit function for geometry.
    def domain(params, x):
        x, y = x[0], x[1]
        x_radius, y_radius = params

        return 1 - jnp.maximum(((x - 5.0) / x_radius)**2, ((y - 1.5) / y_radius)**2)
    geometry_params = (3, 0.5)

    # Construct geometry (simple beam).
    beam, ref_ctrl, find_patch = construct_beam(
            domain_oracle=domain, params=geometry_params,
            len_x=config.len_x, len_y=config.len_y, fidelity=config.fidelity,
            quad_degree=config.quaddeg, material=mat)

    # Boundary conditions
    increment_dict = {
        '1': jnp.array([0.0, 0.0]),
        '2': jnp.array([-1.0]),  # Only applied to y-coordinate.
    }

    # Defines the material parameters.
    # Can ignore this. Only useful when doing optimization wrt material params.
    mat_params = (
        TPUMat.E * jnp.ones(ref_ctrl.shape[0]),
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})
    dirichlet_ctrl = beam.fixed_locs_from_dict(increment_dict)

    # We would like to minimize the potential energy.
    potential_energy_fn = beam.get_potential_energy_fn()
    optimizer = SparseNewtonIncrementalSolver(
            beam, potential_energy_fn, **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    # Now get the functions that we need to do fiber sampling.
    energy_density_fn = get_energy_density_fn(element, mat)
    global_energy_fn = get_global_energy_fn(energy_density_fn, find_patch, g2l, ref_ctrl, dirichlet_ctrl)

    def simulate():
        current_x = l2g(ref_ctrl, ref_ctrl)

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        # Unflatten sequence to local configuration.
        ctrl_seq = beam.unflatten_sequence(
            all_xs, all_fixed_locs, ref_ctrl)
        final_x_local = g2l(current_x, all_fixed_locs[-1], ref_ctrl)

        return final_x_local, [ref_ctrl] + ctrl_seq

    # Reference configuration
    ref_config_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref.png')
    create_static_image(beam.element, ref_ctrl, ref_config_path)

    rprint('Starting optimization (may be slow because of compilation).')
    iter_time = time.time()
    final_x_local, ctrl_seq = simulate()
    rprint(f'\tSolve time: {time.time() - iter_time}')

    rprint(f'Generating image and video with optimization.')

    # Deformed configuration
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    create_static_image(beam.element, final_x_local, image_path)

    # Deformation sequence movie
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_movie(beam.element, ctrl_seq, vid_path)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
