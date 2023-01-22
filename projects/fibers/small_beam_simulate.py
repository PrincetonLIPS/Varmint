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
    'maximize': True,

    'max_update': 1.0,

    'lr': 0.1,
})

varmint.config_flags.DEFINE_config_dict('config', config)

class SteelMat(Material):
    _E = 200.0
    _nu = 0.30
    _density = 8.0


def generate_point_load_fn(ref_ctrl, g2l, point):
    # point is an array of shape (2,)
    index = jnp.sum((ref_ctrl - point) ** 2, axis=-1) < 1e-8
    num_indices = jnp.sum(index)

    def point_load_fn(current_x, fixed_locs, ref_ctrl, force):
        def_ctrl = g2l(current_x, fixed_locs, ref_ctrl)

        # index can contain multiple entries. We want to divide by number of
        # occurrences to get the force correctly.
        return -jnp.sum((def_ctrl[index] - ref_ctrl[index]) * force) / num_indices

    return point_load_fn


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
        mat = NeoHookean2D(SteelMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(SteelMat)
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
        centers = jnp.array([[0.13, 0.13], [0.13, 0.37], [0.13, 0.63], [0.13, 0.87],
                             [0.37, 0.13], [0.37, 0.37], [0.37, 0.63], [0.37, 0.87],
                             [0.63, 0.13], [0.63, 0.37], [0.63, 0.63], [0.63, 0.87],
                             [0.87, 0.13], [0.87, 0.37], [0.87, 0.63], [0.87, 0.87]])
        radius = 0.1

        return -jnp.min(jnp.linalg.norm(point - centers, axis=-1) - radius, axis=0)

    xx = jnp.linspace(0, 1, config.domain_ncp)
    yy = jnp.linspace(0, 1, config.domain_ncp)
    chkr_points = jnp.stack(jnp.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
    chkr = checkerboard4(chkr_points).reshape(config.domain_ncp, config.domain_ncp)

    # Define implicit function for geometry using Bsplines.
    def domain(params, point):
        point = point.reshape(1, -1)
        point = point / jnp.array([config.len_x, config.len_y])

        return bsplines.bspline2d(point, params, knots, knots, config.domain_degree).squeeze()

    # Initial geometry parameters are a checkerboard pattern
    init_controls = -1 * onp.ones((config.domain_ncp, config.domain_ncp, 1)) + 0.8
    #init_controls[1:-1:2, 1:-1:2] += 2
    init_controls = chkr

    knots = bsplines.default_knots(config.domain_degree, config.domain_ncp)
    geometry_params = jnp.array(init_controls)

    # Construct geometry (simple beam).
    beam, ref_ctrl, occupied_pixels, find_patch = construct_beam(
            domain_oracle=domain, params=geometry_params,
            len_x=config.len_x, len_y=config.len_y, fidelity=config.fidelity,
            quad_degree=config.quaddeg, material=mat)

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    increment_dict = {
        '1': jnp.array([0.0, 0.0]),
        #'2': jnp.array([0.0, -disp_t]),
        '3': jnp.array([0.0, 0.0]),
    }

    tractions_dict = {
        'A': jnp.array([0.0, -0.0]),
    }

    # Defines the material parameters.
    if config.E_min_schedule:
        E_min = SteelMat.E * 1e-1
    else:
        E_min = 1e-9

    mat_params = (
        SteelMat.E  * jnp.ones(ref_ctrl.shape[0]),
        SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    tractions = beam.tractions_from_dict(tractions_dict)
    dirichlet_ctrl = beam.fixed_locs_from_dict(ref_ctrl, increment_dict)

    # We would like to minimize the potential energy.
    potential_energy_fn = jax.jit(beam.get_potential_energy_fn())

    # Need a version of potential energy without point forces
    # to do adjoint optimization.
    nopoint_grad_fn = jax.jit(jax.grad(beam.get_potential_energy_fn()))

    # Add a point load at the top left corner.
    point_load_fn = \
        generate_point_load_fn(ref_ctrl, g2l, jnp.array([0.0, config.len_y]))

    # Magnitude of point load.
    point_force = jnp.array([0.0, -1.0])

    # Use this objective function in the solver instead of the standard potential energy.
    def potential_energy_with_point_load(current_x, fixed_locs, tractions, ref_ctrl, mat_params):
        return potential_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, mat_params) \
                + point_load_fn(current_x, fixed_locs, ref_ctrl, point_force)

    strain_energy_fn = jax.jit(beam.get_strain_energy_fn())

    optimizer = SparseCholeskyLinearSolver(beam, potential_energy_with_point_load,
                                           **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    bounds = jnp.array([0.0, 0.0, config.len_x, config.len_y])

    # Now get the functions that we need to do fiber sampling.
    energy_density_fn = get_energy_density_fn(beam.element, mat)
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

    @jax.jit
    def adjoint_field_value(fibers, geometry_params, adjoint_integrand_params):
        return est.estimate_field_value(domain, energy_vjp, fibers,
                                        (geometry_params, adjoint_integrand_params))
    adjoint_field_value_grad = jax.jit(jax.grad(adjoint_field_value, argnums=1))

    # Define simulation function
    def simulate(solver_mat_params):
        current_x = l2g(ref_ctrl, ref_ctrl)
        current_x = optimize(
                current_x, increment_dict, tractions_dict, ref_ctrl, solver_mat_params)

        fixed_locs = beam.fixed_locs_from_dict(ref_ctrl, increment_dict)
        final_x_local = g2l(current_x, fixed_locs, ref_ctrl)

        return final_x_local

    @jax.jit
    def area_estimate(fibers, geometry_params):
        return est.estimate_field_area(domain, fibers, geometry_params)

    def area_penalty(fibers, geometry_params):
        area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
        return jax.nn.relu(area_estimate - 0.5) * config.area_penalty  # lol
    area_penalty_grad = jax.jit(jax.grad(area_penalty, argnums=1))

    def min_area_penalty(fibers, geometry_params):
        area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
        return jax.nn.relu(0.4 - area_estimate) * config.area_penalty  # lol
    min_area_penalty_grad = jax.jit(jax.grad(min_area_penalty, argnums=1))

    key = config.jax_rng

    outer_optimizer = optax.sgd(config.lr)
    opt_state = outer_optimizer.init(geometry_params)

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
        _, _, occupied_pixels, _ = mshr.find_occupied_pixels(domain, geometry_params,
                                                            config.len_x, config.len_y,
                                                            config.fidelity, center=True)

        solver_mat_params = (
            SteelMat.E * occupied_pixels + E_min * ~occupied_pixels,
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )
        rounded_mat_params = (
            SteelMat.E  * occupied_pixels,
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
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
        solution_point_args = (dirichlet_ctrl,
                               beam.tractions_from_dict({}),
                               ref_ctrl,
                               solver_mat_params)
        final_x_global = l2g(final_x_local, ref_ctrl)

        # Make sure to get grad wrt objective function, which does not
        # include point force energy.
        grad_final_x = nopoint_grad_fn(final_x_global, *solution_point_args)

        # Adjoint solve with existing Cholesky matrix.
        adjoint = optimizer.factor(grad_final_x)

        # Sample fibers for FMC and points for standard MC.
        fibers, key = est.sample_fibers(key, bounds, config.num_fibers, config.fiber_len)

        # Parameters for the energy function.
        integrand_params = (final_x_global, occupied_pixels, solver_mat_params)

        solution_point_args = (dirichlet_ctrl, tractions, ref_ctrl, rounded_mat_params)
        quadrature_energy = strain_energy_fn(final_x_global, *solution_point_args)
        quad_energies.append(quadrature_energy)
        curr_real_area = jnp.mean(occupied_pixels)

        if not config.shape_derivative:
            estimated_energy = field_value(fibers, geometry_params, integrand_params)

            if config.inverse_fibers:
                inverted_integrand_params = (final_x_global, 1 - occupied_pixels, solver_mat_params)
                inverted_estimated_energy = field_value(fibers, -1 * geometry_params, inverted_integrand_params)

                estimated_energy += inverted_estimated_energy

            fiber_energies.append(estimated_energy)

            pEptheta = field_value_grad(fibers, geometry_params, integrand_params)

            # Parameters for adjoint
            adjoint_integrand_params = (adjoint, final_x_global, occupied_pixels, solver_mat_params)
            pdEptheta = adjoint_field_value_grad(fibers, geometry_params, adjoint_integrand_params)

            total_energy_derivative = jax.tree_util.tree_map(lambda x, y: x + y, pEptheta, pdEptheta)

            # Compute total area and penalize if deviating outside of [0.4, 0.5].
            area_pen_val = area_penalty_grad(fibers, geometry_params)
            min_area_pen_val = min_area_penalty_grad(fibers, geometry_params)
            area_penalty_grad_norm = jnp.linalg.norm(area_pen_val + min_area_pen_val)

            curr_area_estimate = area_estimate(fibers, geometry_params)

            config.summary_writer.scalar('Fiber energies', estimated_energy, step=i)
            config.summary_writer.scalar('Fiber area estimate', curr_area_estimate, step=i)
            config.summary_writer.scalar('Area penalty grad norm', area_penalty_grad_norm, step=i)
            print(f'\tEstimated area with fiber sampling: {curr_area_estimate}.')
            print(f'\tEstimated integrand with fiber sampling: {estimated_energy}.')
            print(f'\tArea penalty grad: {area_penalty_grad_norm}')
            print(f'Iteration {i} Fiber sampling time: {time.time() - fiber_start_time}')

        config.summary_writer.scalar('Quad energies', quadrature_energy, step=i)
        config.summary_writer.scalar('Real area', curr_real_area, step=i)
        config.summary_writer.scalar('E_min', E_min / SteelMat.E, step=i)

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
            if config.maximize:
                updates, opt_state = outer_optimizer.update(
                    total_energy_derivative - min_area_pen_val - area_pen_val, opt_state)
            else:
                updates, opt_state = outer_optimizer.update(
                    -total_energy_derivative - min_area_pen_val - area_pen_val, opt_state)

            print('max absolute update: ', jnp.max(jnp.abs(updates)))
            # if max absolute update is greater than max_update, scale all updates by max_update / max absolute update
            if jnp.max(jnp.abs(updates)) > config.max_update:
                updates = jnp.multiply(updates, config.max_update / jnp.max(jnp.abs(updates)))

            geometry_params = jax.tree_util.tree_map(lambda x, y: x - y, geometry_params, updates)

        if config.E_min_schedule:
            if i % config.schedule_update_interval == 0:
                E_min = max(E_min * config.schedule_decay_rate, 1e-9)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
