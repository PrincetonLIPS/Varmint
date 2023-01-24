import varmint

import sys
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

    'fidelity': 100,
    'len_x': 75,
    'len_y': 25,

    'init_pattern': '4',

    'domain_ncp': 20,
    'domain_degree': 1,

    'num_fibers': 10000,
    'fiber_len': 0.05,

    'solver_parameters': {},

    'E_min_schedule': False,
    'schedule_update_interval': 100,
    'schedule_decay_rate': 0.5,
    'inverse_fibers': False,
    'min_area_penalty': False,
    'max_area_penalty': True,
    'constrain_spatial_norm': True,
    'spatial_norm_penalty': 0.1,

    'jax_seed': 24,

    'area_penalty': 10,
    'area_penalty_norm_bound': 10,

    'num_iters': 10000,
    'vis_every': 10,
    'save_every': 100,
    'plot_deformed': False,
    'maximize': True,

    'gradient_check': False,
    'n_grad_check': 10,
    'grad_check_eps': 1e-6,

    'max_update': 0.01,
    'max_value': 1.0,
    'reload': False,

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
    if config.init_pattern == '4':
        chkr = checkerboard4(chkr_points).reshape(config.domain_ncp, config.domain_ncp)
    elif config.init_pattern == '16':
        chkr = checkerboard16(chkr_points).reshape(config.domain_ncp, config.domain_ncp)
    else:
        raise ValueError(f'Unknown init pattern {config.init_pattern}')

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

    # Reload
    if config.reload:
        reload_path = '/n/fs/mm-iga/Varmint/projects/fibers/experiments/bridge_16init_lr0001_75x25/sim-bridge_16init_lr0001_75x25-pickles-geoparams-iter4700.png.npy'
        geometry_params = onp.load(reload_path)
        print('loaded')

    # Construct geometry (simple beam).
    beam, ref_ctrl, occupied_pixels, find_patch, gen_stratified_fibers, coords = construct_beam(
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
        tractions = beam.tractions_from_dict({})
        strain_energy = strain_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, solver_mat_params)
        final_x_local = g2l(current_x, fixed_locs, ref_ctrl)

        return current_x, final_x_local, strain_energy

    #@jax.jit
    #def area_estimate(fibers, geometry_params):
    #    return est.estimate_field_area(domain, fibers, geometry_params)

    #def area_penalty(fibers, geometry_params):
    #    area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
    #    return (jax.nn.relu(area_estimate - 0.524) ** 2) * config.area_penalty  # lol
    #area_penalty_grad = jax.jit(jax.grad(area_penalty, argnums=1))

    #def min_area_penalty(fibers, geometry_params):
    #    area_estimate = est.estimate_field_area(domain, fibers, geometry_params)
    #    return (jax.nn.relu(0.4 - area_estimate) ** 2) * config.area_penalty  # lol
    #min_area_penalty_grad = jax.jit(jax.grad(min_area_penalty, argnums=1))

    key = config.jax_rng

    outer_optimizer = optax.adam(config.lr)
    opt_state = outer_optimizer.init(geometry_params)

    quad_energies = []

    # Baseline model
    #path = '/n/fs/mm-iga/Varmint/projects/symgroups/experiments/topopt_with_save/iter_900_pixels.npy'
    #baseline_pixels = onp.load(path)[::-1, :].T.flatten()
    #baseline_mat_params = (
    #        SteelMat.E * baseline_pixels + E_min * ~baseline_pixels,
    #        SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
    #)
    #_, _, baseline_se_p = simulate(baseline_mat_params)
    #print(f'Baseline SE: {baseline_se_p}')

    ## SIMP Baseline topology
    #image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-pixelized-baseline.png')
    #visualize_pixel_domain(config, 0, baseline_pixels, image_path)

    @jax.jit
    def cell_area_estimate(key, geometry_params):
        _, fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    @jax.jit
    def cell_area_estimate2(key, geometry_params):
        _, fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    #@jax.jit
    def cell_area_estimate3(key, geometry_params):
        _, fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    domain_spatial_grad = jax.jit(jax.vmap(jax.grad(domain, argnums=1), in_axes=(None, 0)))

    def objective(cell_area):
        solver_mat_params = (
            E_min + (SteelMat.E - E_min) * cell_area,
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )
        _, _, se_p = simulate(solver_mat_params)
        return se_p

    def area_penalty(cell_area):
        area_estimate = jnp.mean(cell_area)

        return (jax.nn.relu(area_estimate - 0.524) ** 2) * config.area_penalty  # lol

    def min_area_penalty(cell_area):
        area_estimate = jnp.mean(cell_area)

        return (jax.nn.relu(0.4 - area_estimate) ** 2) * config.area_penalty  # lol

    def max_spatial_norm(geometry_params):
        domain_spatial_grad_norms = jnp.linalg.norm(domain_spatial_grad(geometry_params, coords), axis=-1)
        return jnp.max(domain_spatial_grad_norms)

    # Here we start optimization
    rprint('Starting optimization (may be slow because of compilation).')
    for i in range(config.num_iters):
        _, _, occupied_pixels, _ = mshr.find_occupied_pixels(domain, geometry_params,
                                                            config.len_x, config.len_y,
                                                            config.fidelity, center=True)

        #t_strat_sample = time.time()
        key, subkey = jax.random.split(key)

        # Get the function to compute vjp here. Will be more efficient than
        # evaluating this function three times.
        cell_area, cell_area_vjp = jax.vjp(cell_area_estimate, subkey, geometry_params)
        #print(f'cell area est in time {time.time() - t_strat_sample} cell_area: {cell_area.shape}')

        obj_val, obj_ca_bar = jax.value_and_grad(objective)(cell_area)
        obj_val = obj_val.block_until_ready()
        total_energy_derivative = cell_area_vjp(obj_ca_bar)[1].block_until_ready()

        cell_area, cell_area_vjp = jax.vjp(cell_area_estimate2, subkey, geometry_params)
        area_pen_val, area_pen_ca_bar = jax.value_and_grad(area_penalty)(cell_area)
        area_pen_val = area_pen_val.block_until_ready()
        area_pen_grad = cell_area_vjp(area_pen_ca_bar)[1].block_until_ready()

        #cell_area, cell_area_vjp = jax.vjp(cell_area_estimate3, subkey, geometry_params)
        #min_area_pen_val, min_area_pen_ca_bar = jax.value_and_grad(min_area_penalty)(cell_area)
        #min_area_pen_val = min_area_pen_val.block_until_ready()
        #min_area_pen_grad = cell_area_vjp(min_area_pen_ca_bar)[1].block_until_ready()
        min_area_pen_grad = 0.0

        quadrature_energy = obj_val
        quad_energies.append(quadrature_energy)

        # Compute norms to clip large gradients
        total_energy_derivative_norm = jnp.linalg.norm(total_energy_derivative)
        area_penalty_grad_norm = jnp.linalg.norm(area_pen_grad + min_area_pen_grad)

        if area_penalty_grad_norm > config.area_penalty_norm_bound * total_energy_derivative_norm:
            area_pen_grad = area_pen_grad \
                           * config.area_penalty_norm_bound \
                           * total_energy_derivative_norm / area_penalty_grad_norm
            #min_area_pen_grad = min_area_pen_grad \
            #                   * config.area_penalty_norm_bound \
            #                   * total_energy_derivative_norm / area_penalty_grad_norm

        grad_max_spatial_norm = jax.grad(max_spatial_norm)(geometry_params)

        # Rescale spatial norm to energy derivative norm
        grad_max_spatial_norm = grad_max_spatial_norm * total_energy_derivative_norm / jnp.linalg.norm(grad_max_spatial_norm)

        curr_real_area = jnp.mean(occupied_pixels)
        #curr_area_estimate = area_estimate(fibers, geometry_params)

        # Compute the spatial grad of the domain at each nodal point, to see if we get blowup.
        domain_spatial_grad_norms = jnp.linalg.norm(domain_spatial_grad(geometry_params, coords), axis=-1)

        #config.summary_writer.scalar('Fiber area estimate', curr_area_estimate, step=i)
        config.summary_writer.scalar('Area penalty grad norm', area_penalty_grad_norm, step=i)
        print(f'iter: {i:4} r_area: {curr_real_area:<6.4} '
              f'fs_j: {obj_val:<10.6} r_j: {obj_val:<10.6} '
              f'min_spatial_norm: {jnp.min(domain_spatial_grad_norms):<10.6} max_spatial_norm: {jnp.max(domain_spatial_grad_norms):<10.6} '
              #f'fs_time: {fiber_sampling_time:<5.3} s_time: {solve_time:<5.3} '
              f'area_pen: {area_penalty_grad_norm:<6.3} '
              f'max_sg_norm: {jnp.linalg.norm(grad_max_spatial_norm):<6.4} '
              f'energy_d_norm: {total_energy_derivative_norm:<6.4} '
              #f'direct_grad_n: {direct_grad_norm:<8.6} adjoint_grad_n: {adjoint_grad_norm:<8.6} '
              #f'dot_prod: {dot_product_dist:<4.3} area_obj_dot_dist {area_obj_dot_product_dist:<4.3} '
              )

        config.summary_writer.scalar('Min spatial norm', jnp.min(domain_spatial_grad_norms), step=i)
        config.summary_writer.scalar('Max spatial norm', jnp.max(domain_spatial_grad_norms), step=i)
        config.summary_writer.scalar('Quad energies', quadrature_energy, step=i)
        config.summary_writer.scalar('Real area', curr_real_area, step=i)
        config.summary_writer.scalar('E_min', E_min / SteelMat.E, step=i)

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
            ax.plot(quad_energies, label='Quadrature energy')
            ax.legend()
            fig.savefig(image_path)
            plt.close(fig)

            config.summary_writer.flush()

            #rprint(f'Generated visualizations in {time.time() - vis_start_time} secs.')

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
                pickle.dump((quad_energies), f)

        if config.maximize:
            all_grads = total_energy_derivative
        else:
            all_grads = -total_energy_derivative

        if config.min_area_penalty:
            all_grads -= min_area_pen_grad
        if config.max_area_penalty:
            all_grads -= area_pen_grad

        if config.constrain_spatial_norm:
            all_grads -= grad_max_spatial_norm * config.spatial_norm_penalty

        updates, opt_state = outer_optimizer.update(all_grads, opt_state)

        # if max absolute update is greater than max_update, scale all updates by max_update / max absolute update
        if jnp.max(jnp.abs(updates)) > config.max_update:
            updates = jnp.multiply(updates, config.max_update / jnp.max(jnp.abs(updates)))

        geometry_params = jax.tree_util.tree_map(lambda x, y: x - y, geometry_params, updates)

        # Clip geometry params.
        geometry_params = jnp.clip(geometry_params, -config.max_value, config.max_value)

        if config.E_min_schedule:
            if i % config.schedule_update_interval == 0:
                E_min = max(E_min * config.schedule_decay_rate, 1e-9)

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
