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

import varmint.utils.filtering as filtering
import varmint.geometry.bsplines as bsplines

from geometry.cantilever_geometry import construct_beam
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

    'nx': 80,
    'ny': 50,
    'len_x': 160,
    'len_y': 100,

    'simp_exponent': 1,

    'init_pattern': '16',

    'domain_ncp': 20,
    'domain_degree': 1,

    'num_fibers': 10000,
    'fiber_len': 0.05,

    'vol_constraint': 0.5,

    'solver_parameters': {},

    'schedule_update_interval': 100,
    'schedule_decay_rate': 0.5,
    'inverse_fibers': False,
    'max_area_penalty': True,
    'constrain_spatial_norm': True,
    'spatial_norm_penalty': 0.1,

    'jax_seed': 24,

    'area_penalty': 10,
    'area_penalty_norm_bound': 10,

    'num_iters': 10000,
    'vis_every': 1,
    'save_every': 10,
    'plot_deformed': False,
    'maximize': True,

    'oc_update': True,
    'gradient_check': False,
    'n_grad_check': 10,
    'grad_check_eps': 1e-6,

    'filter': False,

    'eta': 0.1,
    'eta_decay': 0.9,
    'decay_interval': 25,

    'slow_constraint': True,

    'max_value': 0.5,
    'move_limit': 0.01,
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
    elif config.init_pattern == 'dense':
        chkr = -onp.ones((config.domain_ncp, config.domain_ncp))
        chkr[:, config.domain_ncp//2:] = 1.0
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

    if config.oc_update:
        geometry_params = jnp.clip(geometry_params, -config.max_value, config.max_value)

    # Reload
    if config.reload:
        reload_path = '/n/fs/mm-iga/Varmint/projects/fibers/experiments/cantilever_against_simp_simp1_slowerdecay_movelimitmatch/sim-cantilever_against_simp_simp1_slowerdecay_movelimitmatch-pickles-geoparams-iter400.png.npy'
        geometry_params = onp.load(reload_path)
        print('loaded')

    # Construct geometry (simple beam).
    beam, ref_ctrl, occupied_pixels, _, gen_stratified_fibers, coords = construct_beam(
            domain_oracle=domain, params=geometry_params,
            len_x=config.len_x, len_y=config.len_y, nx=config.nx, ny=config.ny,
            quad_degree=config.quaddeg, material=mat)

    # ref_ctrl is in "local" coordinates. The "global" coordinates are reparameterized
    # versions (with Dirichlet conditions factored out) to perform unconstrained
    # optimization on. The l2g and g2l functions translate between the two representations.
    l2g, g2l = beam.get_global_local_maps()

    increment_dict = {
        '1': jnp.array([0.0, 0.0]),
    }

    tractions_dict = {
        #'A': jnp.array([0.0, -0.0]),
    }

    # Defines the material parameters.
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
        generate_point_load_fn(ref_ctrl, g2l, jnp.array([config.len_x, 0.0]))

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

    key = config.jax_rng

    outer_optimizer = optax.adam(config.lr)
    opt_state = outer_optimizer.init(geometry_params)

    quad_energies = []

    # Baseline model
    path = '/n/fs/mm-iga/Varmint/projects/symgroups/experiments/cantilever_simp/eled-400.npy'
    baseline_pixels = onp.load(path)[::-1, :].T
    print(f'Baseline area: {onp.mean(baseline_pixels.flatten() ** 3)}')
    baseline_mat_params = (
            E_min + baseline_pixels.flatten() ** 3 * (SteelMat.E - E_min),
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )
    _, _, baseline_se_p = simulate(baseline_mat_params)
    print(f'Baseline SE: {baseline_se_p}')

    # SIMP Baseline topology
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-pixelized-baseline.png')
    visualize_pixel_domain(config, 0, baseline_pixels, image_path)

    @jax.jit
    def cell_area_estimate(key, geometry_params):
        fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    @jax.jit
    def cell_area_estimate2(key, geometry_params):
        fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    @jax.jit
    def cell_area_estimate3(key, geometry_params):
        fibers_per_cell = gen_stratified_fibers(key)
        cell_area = jax.vmap(est.estimate_field_area, in_axes=(None, 0, None))(domain, fibers_per_cell, geometry_params)
        return cell_area

    domain_spatial_grad = jax.jit(jax.vmap(jax.grad(domain, argnums=1), in_axes=(None, 0)))

    def objective(_cell_area):
        if config.filter:
            filtered_cell_area = filtering.physical_density(_cell_area.reshape(config.nx, config.ny), 2.5, 20.0)
        else:
            filtered_cell_area = _cell_area
        cell_area = filtered_cell_area.flatten()
        cell_area = cell_area ** config.simp_exponent
        solver_mat_params = (
            E_min + (SteelMat.E - E_min) * cell_area,
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )
        _, _, se_p = simulate(solver_mat_params)
        return se_p

    def objective_nofilter(_cell_area):
        filtered_cell_area = _cell_area
        cell_area = filtered_cell_area.flatten()
        cell_area = cell_area ** config.simp_exponent
        solver_mat_params = (
            E_min + (SteelMat.E - E_min) * cell_area,
            SteelMat.nu * jnp.ones(ref_ctrl.shape[0]),
        )
        _, _, se_p = simulate(solver_mat_params)
        return se_p

    def constraint(_cell_area):
        if config.filter:
            filtered_cell_area = filtering.mean_density(_cell_area.reshape(config.nx, config.ny), 2.5, 20.0)
        else:
            filtered_cell_area = _cell_area
        cell_area = filtered_cell_area.flatten()
        cell_area = cell_area ** config.simp_exponent
        return jnp.mean(cell_area) - config.vol_constraint

    def area_penalty(cell_area):
        cell_area = cell_area ** config.simp_exponent
        area_estimate = jnp.mean(cell_area)

        return (jax.nn.relu(area_estimate - config.vol_constraint) ** 2) * config.area_penalty  # lol

    def max_spatial_norm(geometry_params):
        domain_spatial_grad_norms = jnp.linalg.norm(domain_spatial_grad(geometry_params, coords), axis=-1)
        return jnp.max(domain_spatial_grad_norms)

    # Here we start optimization
    rprint('Starting optimization (may be slow because of compilation).')
    eta = config.eta  # Numerical damping coefficient.
    for i in range(config.num_iters):
        if i > 0 and i % config.decay_interval == 0:
            eta = eta * config.eta_decay
        _, _, occupied_pixels, _ = mshr.find_occupied_pixels(domain, geometry_params,
                                                            config.len_x, config.len_y,
                                                            config.nx, config.ny, center=True)


        # Get the function to compute vjp here. Will be more efficient than
        # evaluating this function three times.
        key, subkey = jax.random.split(key)
        cell_area, cell_area_vjp = jax.vjp(cell_area_estimate, subkey, geometry_params)
        obj_val, obj_ca_bar = jax.value_and_grad(objective)(cell_area)
        real_obj_val = objective_nofilter(cell_area)
        obj_val = obj_val.block_until_ready()
        total_energy_derivative = cell_area_vjp(obj_ca_bar)[1].block_until_ready()

        key, subkey = jax.random.split(key)
        cell_area, cell_area_vjp = jax.vjp(cell_area_estimate2, subkey, geometry_params)
        area_pen_val, area_pen_ca_bar = jax.value_and_grad(area_penalty)(cell_area)
        area_pen_val = area_pen_val.block_until_ready()
        area_pen_grad = cell_area_vjp(area_pen_ca_bar)[1].block_until_ready()

        quadrature_energy = obj_val
        quad_energies.append(quadrature_energy)

        # Compute norms to clip large gradients
        total_energy_derivative_norm = jnp.linalg.norm(total_energy_derivative)
        area_penalty_grad_norm = jnp.linalg.norm(area_pen_grad)

        if area_penalty_grad_norm > config.area_penalty_norm_bound * total_energy_derivative_norm:
            area_pen_grad = area_pen_grad \
                           * config.area_penalty_norm_bound \
                           * total_energy_derivative_norm / area_penalty_grad_norm

        grad_max_spatial_norm = jax.grad(max_spatial_norm)(geometry_params)

        # Rescale spatial norm to energy derivative norm
        grad_max_spatial_norm = grad_max_spatial_norm * total_energy_derivative_norm / jnp.linalg.norm(grad_max_spatial_norm)

        curr_real_area = jnp.mean(occupied_pixels)

        # Compute the spatial grad of the domain at each nodal point, to see if we get blowup.
        domain_spatial_grad_norms = jnp.linalg.norm(domain_spatial_grad(geometry_params, coords), axis=-1)

        #config.summary_writer.scalar('Fiber area estimate', curr_area_estimate, step=i)
        config.summary_writer.scalar('Area penalty grad norm', area_penalty_grad_norm, step=i)
        print(f'iter: {i:4} r_area: {curr_real_area:<6.4} eta: {eta:<6.4} '
              f'r_j: {real_obj_val:<10.6} '
              f'min_spatial_norm: {jnp.min(domain_spatial_grad_norms):<10.6} max_spatial_norm: {jnp.max(domain_spatial_grad_norms):<10.6} '
              f'area_pen: {area_penalty_grad_norm:<6.3} '
              f'max_sg_norm: {jnp.linalg.norm(grad_max_spatial_norm):<6.4} '
              f'energy_d_norm: {total_energy_derivative_norm:<6.4} '
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
            #visualize_pixel_domain(config, i, occupied_pixels, image_path)
            visualize_pixel_domain(config, i, cell_area, image_path)

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

        if config.oc_update:
            key, subkey = jax.random.split(key)
            cell_area, cell_area_vjp = jax.vjp(cell_area_estimate3, subkey, geometry_params)
            _, constraint_ca_bar = jax.value_and_grad(constraint)(cell_area)
            constraint_grad = cell_area_vjp(constraint_ca_bar)[1]

            def quick_constraint(geometry_params):
                _, _, occupied_pixels, _ = mshr.find_occupied_pixels(domain, geometry_params,
                                                                    config.len_x, config.len_y,
                                                                    config.nx, config.ny, center=True)
                return jnp.mean(occupied_pixels) - config.vol_constraint

            def slow_constraint(geometry_params):
                cell_area = cell_area_estimate3(subkey, geometry_params)
                return constraint(cell_area)

            # Bisection algorithm to find optimal lagrange multiplier.
            translated_gps = geometry_params + config.max_value
            l1, l2, move = 0, 1e1, config.move_limit
            dc = total_energy_derivative
            dv = constraint_grad
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = (l2 + l1) / 2.0

                # Optimality criteria update.
                # If dc and dv are 0, then the first term will be zero. We want it to
                # be 1 instead. Add the correction at the end.
                xnew = translated_gps * jnp.abs(lmid * dv / (1e-8 + dc)) ** eta + \
                        translated_gps * ((dc == 0.0) & (dv == 0.0))

                xnew = jnp.minimum(xnew, translated_gps + move)  # Cap from above by move limit.
                xnew = jnp.minimum(jnp.ones_like(xnew) * config.max_value * 2, xnew)  # Cap from above.

                xnew = jnp.maximum(translated_gps - move, xnew)  # Cap from below by move limit.
                xnew = jnp.maximum(jnp.zeros_like(xnew), xnew)  # Cap from below by 0.

                # If too much volume, increase lagrange multiplier.
                # Else, decrease.
                if config.slow_constraint:
                    constraint_val = slow_constraint(xnew - config.max_value)
                else:
                    constraint_val = quick_constraint(xnew - config.max_value)

                if constraint_val > 0:
                    l1 = lmid
                else:
                    l2 = lmid

                if (l1+l2 == 0):
                    raise Exception('div0, breaking lagrange multiplier')
            geometry_params = xnew - config.max_value
        else:
            if config.maximize:
                all_grads = total_energy_derivative
            else:
                all_grads = -total_energy_derivative

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

    rprint(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    varmint.app.run(main)
