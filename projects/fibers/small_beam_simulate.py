import varmint

import time
import os
import pickle

from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver
from varmint.solver.cholesky_solver import SparseCholeskyLinearSolver

from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

import varmint.geometry.bsplines as bsplines

from geometry.small_beam_geometry import construct_beam
from plotting.small_beam_plot import plot_small_beam, visualize_domain, visualize_pixel_domain
from implicit_differentiation import bisect
import estimators as est
import geometry.mesher as mshr

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
    'quaddeg': 3,
    'mat_model': 'LinearElastic2D',

    'fidelity': 200,
    'len_x': 10,
    'len_y': 4,

    'domain_ncp': 20,
    'domain_degree': 1,

    'num_fibers': 10000,
    'fiber_len': 0.05,

    'solver_parameters': {},

    'E_min_schedule': False,
    'schedule_update_interval': 100,
    'schedule_decay_rate': 0.5,
    'inverse_fibers': False,

    'shape_derivative': False,
    'shape_derivative_optimization_iters': 500,
    'shape_derivative_optimization_lr': 0.01,

    'jax_seed': 24,

    'area_penalty': 10,

    'num_iters': 10000,
    'vis_every': 10,
    'save_every': 100,
    'plot_deformed': False,

    'lr': 0.1,
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


def get_global_energy_fn(energy_density_fn, find_patch, g2l, ref_ctrl, dirichlet_ctrl, fidelity):
    def global_energy_density(params, point):
        global_coords, occupied_pixels, mat_params = params
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

    @jax.vmap
    def checkerboard4(point):
        centers = jnp.array([[0.25, 0.25],
                             [0.75, 0.75],
                             [0.25, 0.75],
                             [0.75, 0.25]])
        radius = 0.2

        return -jnp.min(jnp.linalg.norm(point - centers, axis=-1) - radius, axis=0)

    @jax.vmap
    def checkerboard16(point):
        centers = jnp.array([[0.15, 0.15], [0.15, 0.4], [0.15, 0.6], [0.15, 0.85],
                             [0.4, 0.15], [0.4, 0.4], [0.4, 0.6], [0.4, 0.85],
                             [0.6, 0.15], [0.6, 0.4], [0.6, 0.6], [0.6, 0.85],
                             [0.85, 0.15], [0.85, 0.4], [0.85, 0.6], [0.85, 0.85]])
        radius = 0.05

        return -jnp.min(jnp.linalg.norm(point - centers, axis=-1) - radius, axis=0)

    xx = jnp.linspace(0, 1, config.domain_ncp)
    yy = jnp.linspace(0, 1, config.domain_ncp)
    chkr_points = jnp.stack(jnp.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
    chkr = checkerboard4(chkr_points).reshape(config.domain_ncp, config.domain_ncp)

    # Define implicit function for geometry using Bsplines.
    def domain(params, point):
        offset, bparams = params
        point = point.reshape(1, -1)
        point = point / jnp.array([config.len_x, config.len_y])

        return bsplines.bspline2d(point, bparams, knots, knots, config.domain_degree).squeeze() + offset

    # Initial geometry parameters are a checkerboard pattern
    init_controls = -1 * onp.ones((config.domain_ncp, config.domain_ncp, 1)) + 0.8
    #init_controls[1:-1:2, 1:-1:2] += 2
    init_controls = chkr

    knots = bsplines.default_knots(config.domain_degree, config.domain_ncp)
    bspline_params = jnp.array(init_controls)
    geometry_params = (0.0, bspline_params)

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
    if config.E_min_schedule:
        E_min = TPUMat.E * 1e-1
    else:
        E_min = 1e-9

    mat_params = (
        TPUMat.E  * jnp.ones(ref_ctrl.shape[0]),
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict({})
    dirichlet_ctrl = beam.fixed_locs_from_dict(ref_ctrl, increment_dict)

    # We would like to minimize the potential energy.
    potential_energy_fn = jax.jit(beam.get_potential_energy_fn())
    optimizer = SparseCholeskyLinearSolver(beam, potential_energy_fn,
                                           **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    bounds = jnp.array([0.0, 0.0, config.len_x, config.len_y])

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    # Now get the functions that we need to do fiber sampling.
    energy_density_fn = get_energy_density_fn(beam.element, mat)
    vmap_energy_density_fn = jax.jit(jax.vmap(energy_density_fn, in_axes=(0, None, None, None)))
    global_energy_fn = get_global_energy_fn(energy_density_fn, find_patch, g2l,
                                            ref_ctrl, dirichlet_ctrl, config.fidelity)

    # Define the functions to compute the two terms in the energy total derivative.
    # Use this adjoint for a vjp with the energy density function
    @jax.jit
    def energy_vjp(params, point):
        adjoint, final_x_global, occupied_pixels, mat_params = params

        def global_energy_density_vjp_form(final_x_global):
            integrand_params = (final_x_global, occupied_pixels, mat_params)
            return global_energy_fn(integrand_params, point)

        primal, tangents = \
                jax.jvp(global_energy_density_vjp_form, (final_x_global,), (adjoint,))
        return tangents

    global_energy_fn = jax.jit(global_energy_fn)

    # Functions to perform MC using fiber sampling.
    @jax.jit
    def field_value(fibers, geometry_params, integrand_params):
        return est.estimate_field_value(domain, global_energy_fn, fibers,
                                        (geometry_params, integrand_params))
    field_value_grad = jax.jit(jax.grad(field_value, argnums=1))

    # Define simulation function
    def simulate(solver_mat_params):
        current_x = l2g(ref_ctrl, ref_ctrl)
        current_x = optimize(
                current_x, increment_dict, {}, ref_ctrl, solver_mat_params)

        fixed_locs = beam.fixed_locs_from_dict(ref_ctrl, increment_dict)
        final_x_local = g2l(current_x, fixed_locs, ref_ctrl)

        return final_x_local

    @jax.jit
    def area_estimate(fibers, geometry_params):
        return est.estimate_field_area(domain, fibers, geometry_params)

    def area_penalty(fibers, geometry_params):
        area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
        return jax.nn.relu(area_estimate - 0.5) ** 2 * config.area_penalty  # lol
    area_penalty_grad = jax.jit(jax.grad(area_penalty, argnums=1))

    def min_area_penalty(fibers, geometry_params):
        area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
        return jax.nn.relu(0.4 - area_estimate) ** 2 * config.area_penalty  # lol
    min_area_penalty_grad = jax.jit(jax.grad(min_area_penalty, argnums=1))

    key = config.jax_rng

    outer_optimizer = optax.adam(config.lr)
    opt_state = outer_optimizer.init(geometry_params[1])

    fiber_energies = []
    quad_energies = []

    @jax.jit
    def shape_derivative_optimization(fibers, geometry_params, integrand_params):
        # First compute shape derivative for energy.
        energy_term, surface_points, is_surface = est.compute_shape_derivative(lambda x: x, domain, global_energy_fn,
                                                                    fibers, (geometry_params, integrand_params))
        def max_area_penalty_fn(area_estimate):
            return jax.nn.relu(area_estimate - 0.5) ** 2 * config.area_penalty
        max_area_penalty_term, _, _ = est.compute_shape_derivative(max_area_penalty_fn, domain, lambda _, x: 1.0,
                                                                fibers, (geometry_params, None))

        def min_area_penalty_fn(area_estimate):
            return jax.nn.relu(0.4 - area_estimate) ** 2 * config.area_penalty
        min_area_penalty_term, _, _ = est.compute_shape_derivative(min_area_penalty_fn, domain, lambda _, x: 1.0,
                                                                fibers, (geometry_params, None))

        domain_perturbations = -(energy_term - max_area_penalty_term - min_area_penalty_term) * config.lr

        # Now optimize geometric parameters to match the perturbations at intersection points.
        vmap_domain = jax.vmap(domain, in_axes=(None, 0))
        def loss_fn(geometric_params):
            return jnp.sum(is_surface * (vmap_domain(geometric_params, surface_points) - domain_perturbations) ** 2)
        grad_fn = jax.grad(loss_fn, argnums=0)
        initial_loss = loss_fn(geometry_params)

        def body_fn(geometry_params):
            return jax.tree_util.tree_map(lambda x, y: x - y * config.shape_derivative_optimization_lr,
                                            geometry_params, grad_fn(geometry_params))

        geometry_params = jax.lax.fori_loop(0, config.shape_derivative_optimization_iters, lambda i, x: body_fn(x), geometry_params)
        final_loss = loss_fn(geometry_params)

        return geometry_params, initial_loss, final_loss

    # Here we start optimization
    rprint('Starting optimization (may be slow because of compilation).')
    for i in range(config.num_iters):

        # Compute an offset that approximately satisfies the area constraint.
        offset, bspline_params = geometry_params
        def real_area(offset):
            params = (offset, bspline_params)
            _, _, occupied_pixels, _ = mshr.find_occupied_pixels(domain, params,
                                                                config.len_x, config.len_y,
                                                                config.fidelity, center=True)
            return jnp.mean(occupied_pixels), occupied_pixels
        current_area, occupied_pixels = real_area(offset)
        previous_over = None
        offset_increment = 0.01
        while current_area < 0.495 or current_area > 0.505:
            if current_area < 0.495:
                if previous_over == True:
                    offset_increment /= 2
                offset -= offset_increment
                previous_over = False
            else:
                if previous_over == False:
                    offset_increment /= 2
                offset += offset_increment
                previous_over = True
            current_area, occupied_pixels = real_area(offset)
        geometry_params = (offset, bspline_params)
        ###### Finished computing offset.

        solver_mat_params = (
            TPUMat.E * occupied_pixels + E_min * ~occupied_pixels,
            TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )
        rounded_mat_params = (
            TPUMat.E  * occupied_pixels,
            TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )

        ref_ctrl = jnp.array(ref_ctrl)
        iter_time = time.time()
        final_x_local = simulate(solver_mat_params)
        rprint(f'Iteration {i} Solve time: {time.time() - iter_time}')

        # Now it's time for fiber sampling. Use the alternatively defined energy functions
        # together with fiber sampling to estimate the gradient.
        fiber_start_time = time.time()

        # Compute gradient with respect to geometry through fiber sampling here.
        # First solve adjoint problem.
        solution_point_args = (dirichlet_ctrl, tractions, ref_ctrl, solver_mat_params)
        final_x_global = l2g(final_x_local, ref_ctrl)

        # Sample fibers for FMC and points for standard MC.
        fibers, key = est.sample_fibers(key, bounds, config.num_fibers, config.fiber_len)

        # Parameters for the energy function.
        integrand_params = (final_x_global, occupied_pixels, solver_mat_params)

        solution_point_args = (dirichlet_ctrl, tractions, ref_ctrl, rounded_mat_params)
        quadrature_energy = potential_energy_fn(final_x_global, *solution_point_args)
        quad_energies.append(quadrature_energy)
        curr_real_area = jnp.mean(occupied_pixels)

        if not config.shape_derivative:
            estimated_energy = field_value(fibers, geometry_params, integrand_params)

            if config.inverse_fibers:
                inverted_integrand_params = (final_x_global, 1 - occupied_pixels, solver_mat_params)
                inverted_estimated_energy = field_value(fibers, -1 * geometry_params, inverted_integrand_params)

                estimated_energy += inverted_estimated_energy

            fiber_energies.append(estimated_energy)

            pEptheta = field_value_grad(fibers, geometry_params, integrand_params)[1]  # Only need the gradient wrt geometry params.
            total_energy_derivative = pEptheta  #jax.tree_util.tree_map(lambda x, y: x + y, pEptheta, pdEptheta)

            # Compute total area and penalize if deviating outside of [0.4, 0.5].
            # area_pen_val = area_penalty_grad(fibers, geometry_params)
            # min_area_pen_val = min_area_penalty_grad(fibers, geometry_params)
            # area_penalty_grad_norm = jnp.linalg.norm(area_pen_val + min_area_pen_val)

            curr_area_estimate = area_estimate(fibers, geometry_params)

            config.summary_writer.scalar('Fiber energies', estimated_energy, step=i)
            config.summary_writer.scalar('Fiber area estimate', curr_area_estimate, step=i)
            #config.summary_writer.scalar('Area penalty grad norm', area_penalty_grad_norm, step=i)
            print(f'\tEstimated area with fiber sampling: {curr_area_estimate}.')
            print(f'\tEstimated integrand with fiber sampling: {estimated_energy}.')
            #print(f'\tArea penalty grad: {area_penalty_grad_norm}')
            print(f'Iteration {i} Fiber sampling time: {time.time() - fiber_start_time}')

        config.summary_writer.scalar('Quad energies', quadrature_energy, step=i)
        config.summary_writer.scalar('Real area', curr_real_area, step=i)
        config.summary_writer.scalar('E_min', E_min / TPUMat.E, step=i)

        print(f'\tCurrent E_min value: {E_min}.')
        print(f'\tReal area after rounding: {curr_real_area}.')
        print(f'\tEstimated integrand with quadrature: {quadrature_energy}')

        if i % config.vis_every == 0 or i == config.num_iters - 1:
            vis_start_time = time.time()

            # Implicit topology
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-implicit-iter{i}.png')
            visualize_domain(config, i, domain, geometry_params, image_path)

            # Pixelixed topology
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-pixelized-iter{i}.png')
            visualize_pixel_domain(config, i, occupied_pixels, image_path)

            # Energy graph
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-energy-plot.png')
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(fiber_energies, label='Fiber estimated energy')
            ax.plot(quad_energies, label='Quadrature energy')
            ax.legend()
            fig.savefig(image_path)
            plt.close(fig)

            config.summary_writer.flush()

            rprint(f'Generated visualizations in {time.time() - vis_start_time} secs.')

            if config.plot_deformed:
                # Deformed configuration
                image_path = \
                        os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-iter{i}.png')
                plot_small_beam(config, beam.element, ref_ctrl[occupied_pixels],
                                final_x_local[occupied_pixels], image_path)

        if i % config.save_every == 0:
            file_path = os.path.join(args.exp_dir,
                                     f'sim-{args.exp_name}-pickles-geoparams-iter{i}.png')
            onp.save(file_path, geometry_params)

            file_path = os.path.join(args.exp_dir,
                                     f'sim-{args.exp_name}-pickles-energies-iter{i}.png')
            with open(file_path, 'wb') as f:
                pickle.dump((fiber_energies, quad_energies), f)


        if config.shape_derivative:
            shape_derivative_start = time.time()
            geometry_params, init_loss, final_loss = shape_derivative_optimization(fibers, geometry_params, integrand_params)
            rprint(f'Shape derivative time: {time.time() - shape_derivative_start}')
            rprint(f'\tInitial loss: {init_loss} Final loss: {final_loss}')
        else:
            updates, opt_state = outer_optimizer.update(total_energy_derivative, opt_state)
            new_bspline_params = jax.tree_util.tree_map(lambda x, y: x - y, geometry_params[1], updates)
            geometry_params = (geometry_params[0], new_bspline_params)

        if config.E_min_schedule:
            if i % config.schedule_update_interval == 0:
                E_min = max(E_min * config.schedule_decay_rate, 1e-9)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)