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
from plotting.small_beam_plot import plot_small_beam
import estimators as est

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
    'quaddeg': 10,
    'mat_model': 'NeoHookean2D',
    'ncp': 10,
    'splinedeg': 1,

    'fidelity': 100,
    'len_x': 10,
    'len_y': 4,

    'num_fibers': 1000,
    'fiber_len': 0.5,

    'solver_parameters': {
        'tol': 1e-8,
    },

    'jax_seed': 24,
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
        return element.get_map_jac_fn(point[onp.newaxis, :])(ctrl).squeeze()
    def jacobian_ctrl_fn(point, ctrl):
        return element.get_ctrl_jacobian_fn(point[onp.newaxis, :])(ctrl).squeeze()
    energy_fn = material.get_energy_fn()

    def energy_density(point, def_ctrl, ref_ctrl, mat_params):
        # Copied from varmint.physics.energy
        def_jacs = jacobian_u_fn(point, def_ctrl)
        ref_jacs = jacobian_u_fn(point, ref_ctrl)

        # Deformation gradients
        defgrads = def_jacs @ jnp.linalg.inv(ref_jacs)

        return energy_fn(defgrads, *mat_params) * 1e3 * jnp.abs(jnp.linalg.det(ref_jacs))

    return energy_density


def get_global_energy_fn(energy_density_fn, find_patch, occupied_pixels, g2l, ref_ctrl,
                         dirichlet_ctrl, fidelity):
    def global_energy_density(params, point):
        global_coords, ref_ctrl, dirichlet_ctrl, mat_params = params
        local_coords = g2l(global_coords, dirichlet_ctrl, ref_ctrl)

        # Figure out which patch the point belongs to.
        # Transform `point` to [0, 1] x [0, 1]
        patch_index, point = find_patch(point)

        valid_patch = energy_density_fn(point, jnp.take(local_coords, patch_index, axis=0),
                                        jnp.take(ref_ctrl, patch_index, axis=0),
                                        jax.tree_util.tree_map(
                                            lambda x: jnp.take(x, patch_index, axis=0),
                                                               mat_params))

        # TODO(doktay): Why am I getting NaNs??
        # Is it because quadrature points are getting sampled near the internal boundaries
        # which might contain degenerate elements?
        return jax.lax.cond(jnp.logical_and(patch_index >= 0,
                                            jnp.logical_and(~jnp.isnan(valid_patch),
                                                            occupied_pixels[patch_index])),
                            lambda: valid_patch * (fidelity ** 2),
                            lambda: 0.0)

    return global_energy_density


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

        return 1 - jnp.maximum(((x - 5.0) / x_radius)**2, ((y - 2.0) / y_radius)**2)
    geometry_params = (4.01, 0.5)

    # Construct geometry (simple beam).
    beam, ref_ctrl, occupied_pixels, find_patch = construct_beam(
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
    E_min = 1e-9
    solver_mat_params = (
        TPUMat.E * occupied_pixels + E_min * ~occupied_pixels,
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )

    mat_params = (
        TPUMat.E  * jnp.ones(ref_ctrl.shape[0]),
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})
    dirichlet_ctrl = beam.fixed_locs_from_dict(ref_ctrl, increment_dict)

    # We would like to minimize the potential energy.
    potential_energy_fn = jax.jit(beam.get_potential_energy_fn())
    optimizer = SparseNewtonIncrementalSolver(
            beam, potential_energy_fn, **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    # Now get the functions that we need to do fiber sampling.
    energy_density_fn = get_energy_density_fn(beam.element, mat)
    vmap_energy_density_fn = jax.jit(jax.vmap(energy_density_fn, in_axes=(0, None, None, None)))
    global_energy_fn = get_global_energy_fn(energy_density_fn, find_patch, occupied_pixels, g2l,
                                            ref_ctrl, dirichlet_ctrl, config.fidelity)

    def simulate():
        current_x = l2g(ref_ctrl, ref_ctrl)

        current_x, all_xs, all_fixed_locs, solved_increment = optimize(
                current_x, increment_dict, tractions, ref_ctrl, solver_mat_params)

        # Unflatten sequence to local configuration.
        ctrl_seq = beam.unflatten_sequence(
            all_xs, all_fixed_locs, ref_ctrl)
        final_x_local = g2l(current_x, all_fixed_locs[-1], ref_ctrl)

        return final_x_local, [ref_ctrl] + ctrl_seq

    # Reference configuration
    ref_config_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref.png')
    plot_small_beam(config, beam.element, ref_ctrl[occupied_pixels], None, ref_config_path)

    ref_ctrl = jnp.array(ref_ctrl)
    rprint('Starting optimization (may be slow because of compilation).')
    iter_time = time.time()
    final_x_local, ctrl_seq = simulate()
    rprint(f'\tSolve time: {time.time() - iter_time}')

    # Define the functions to compute the two terms in the energy total derivative.
    # Use this adjoint for a vjp with the energy density function
    @jax.jit
    def energy_vjp(params, point):
        adjoint, final_x_global, ref_ctrl, dirichlet_ctrl, mat_params = params

        def global_energy_density_vjp_form(final_x_global):
            integrand_params = (final_x_global, ref_ctrl, dirichlet_ctrl, mat_params)
            return global_energy_fn(integrand_params, point)

        primal, tangents = \
                jax.jvp(global_energy_density_vjp_form, (final_x_global,), (adjoint,))
        return tangents

    global_energy_fn = jax.jit(global_energy_fn)

    # Now it's time for fiber sampling. Use the alternatively defined energy functions
    # together with fiber sampling to estimate the gradient.
    key = config.jax_rng

    # Compute gradient with respect to geometry through fiber sampling here.
    # First solve adjoint problem.
    solution_point_args = (dirichlet_ctrl, tractions, ref_ctrl, solver_mat_params)
    final_x_global = l2g(final_x_local, ref_ctrl)
    grad_final_x = optimizer.grad_fun(final_x_global, *solution_point_args)

    adjoint = optimizer.linear_solve(final_x_global, solution_point_args, -grad_final_x,
                                     solve_tol=config.solver_parameters.tol)

    # Sample fibers for FMC and points for standard MC.
    bounds = jnp.array([0.0, 0.0, config.len_x, config.len_y])
    fibers, key = est.sample_fibers(key, bounds, config.num_fibers, config.fiber_len)
    points, key = est.sample_points(key, bounds, config.num_fibers)

    # Actually perform MC.
    def field_value(fibers, geometry_params, integrand_params):
        return est.estimate_field_value(domain, global_energy_fn, fibers,
                                        (geometry_params, integrand_params))
    field_value_grad = jax.grad(field_value, argnums=1)

    def adjoint_field_value(fibers, geometry_params, adjoint_integrand_params):
        return est.estimate_field_value(domain, energy_vjp, fibers,
                                        (geometry_params, adjoint_integrand_params))
    adjoint_field_value_grad = jax.grad(adjoint_field_value, argnums=1)

    # Parameters for the energy function.
    integrand_params = (final_x_global, ref_ctrl, dirichlet_ctrl, solver_mat_params)
    estimated_area = field_value(fibers, geometry_params, integrand_params)
    estimated_area_mc = est.estimate_field_value_mc(domain, global_energy_fn, points,
                                                    (geometry_params, integrand_params))


    pEptheta = field_value_grad(fibers, geometry_params, integrand_params)

    # Parameters for adjoint
    adjoint_integrand_params = (adjoint, final_x_global, ref_ctrl, dirichlet_ctrl, solver_mat_params)
    pdEptheta = adjoint_field_value_grad(fibers, geometry_params, adjoint_integrand_params)

    total_energy_derivative = jax.tree_util.tree_map(lambda x, y: x + y, pEptheta, pdEptheta)


    print(f'Estimated integrand with fiber sampling: {estimated_area}.')
    print(f'Estimated integrand with Monte Carlo: {estimated_area_mc}.')
    print(f'Estimated integrand with quadrature: '
          f'{potential_energy_fn(final_x_global, *solution_point_args)}')
    print(f'Estimated total derivative: {total_energy_derivative}.')
    print(f'\tFirst term: {pEptheta} \n\tSecond term: {pdEptheta}')

    rprint(f'Generating image and video with optimization.')

    # Deformed configuration
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.png')
    plot_small_beam(config, beam.element, ref_ctrl[occupied_pixels], final_x_local[occupied_pixels], image_path)

    # Deformation sequence movie
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized.mp4')
    create_movie(beam.element, ctrl_seq, vid_path)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
