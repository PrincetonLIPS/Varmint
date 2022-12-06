import os
import re
import sys
import time
import pickle

from functools import partial

import varmint

from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D, NeoHookean2DClamped
from varmint.physics.materials import Material
from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

from varmint.utils.mpi_utils import rprint, pytree_reduce, test_pytrees_equal
from varmint.utils.train_utils import update_ewa
from varmint.utils.movie_utils import create_movie, create_static_image, plot_ctrl

from shape_matching_geometry import generate_pore_shapes_geometry, generate_rectangular_radii
from generate_shape_targets import get_shape_target_generator

import jax
import jax.numpy as jnp
import numpy as onp

import optax
import haiku as hk

import matplotlib.pyplot as plt


FLAGS = varmint.flags.FLAGS
varmint.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/nma_v1/shape_matching/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/nma_v1/shape_matching')

varmint.config_flags.DEFINE_config_file('config', 'config/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.3
    _density = 1.25


def main(argv):
    args, dev_id, local_rank = varmint.initialize_experiment(verbose=True)
    config = args.config
    comm = varmint.MPI.COMM_WORLD

    if config.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)
    else:
        raise ValueError('Incorrect material model')

    # Construct geometry function along with initial geometry parameters.
    cell, radii_to_ctrl_fn, n_cells, get_central_pore_points, init_central_radii, init_mesh_perturb = \
        generate_pore_shapes_geometry(config, mat)
    init_radii = jnp.concatenate((
            generate_rectangular_radii((n_cells,), config.ncp),
    ))

    # We have a bunch of geometry parameters now:
    #   Outer radii, inner radii, mesh perturbations
    all_init_radii = (init_radii, init_central_radii, init_mesh_perturb)

    # Initialization of local-global transformations, reference control points, tractions.
    potential_energy_fn = cell.get_potential_energy_fn()
    l2g, g2l = cell.get_global_local_maps()
    ref_ctrl = radii_to_ctrl_fn(*all_init_radii)
    tractions = cell.tractions_from_dict({})

    # Set up material parameters based on defaults.
    # We could optimize these per patch if we wanted to.
    mat_params = (
        TPUMat.E * jnp.ones(ref_ctrl.shape[0]),
        TPUMat.nu * jnp.ones(ref_ctrl.shape[0]),
    )

    # Construct optimizer.
    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, dev_id=dev_id,
                                              **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    # Differentiable simulation function for given displacements and radii (decoder).
    def simulate(disps, radii, internal_radii, mesh_perturb):
        ref_ctrl = radii_to_ctrl_fn(radii, internal_radii, mesh_perturb)

        # The optimizer works in the global configuration.
        current_x = l2g(ref_ctrl, ref_ctrl)

        increment_dict = config.get_increment_dict(disps)
        current_x, all_xs, all_fixed_locs, _ = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        # Unflatten sequence to local configuration.
        ctrl_seq = cell.unflatten_sequence(
            all_xs, all_fixed_locs, ref_ctrl)
        final_x_local = g2l(current_x, all_fixed_locs[-1], ref_ctrl)

        return final_x_local, [ref_ctrl] + ctrl_seq

    # Initialize neural network (encoder).
    nn_fn = config.get_nn_fn(
            config.max_disp, config.n_layers, config.n_activations, config.n_disps)
    central_pore_points = get_central_pore_points(ref_ctrl)

    n_interior = central_pore_points.shape[0]
    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    dummy_displacements = central_pore_points.flatten()
    init_nn_params = nn_fn_t.init(config.jax_rng, dummy_displacements)

    # The following are two utility functions to help with the loss function.
    # We would like the shapes to be scale, rotation, and reindexing independent.

    def normalize_pore_shape(cps):
        """Function to ensure shape comparison is scale-independent."""

        center = jnp.mean(cps, axis=0)
        cps = cps - center
        norm = jnp.mean(jnp.linalg.norm(cps, axis=-1))

        return cps / norm

    def min_dist_rotation_reindexing(normed1, normed2, rotation=True, norm=None):
        """Factor out rotation (using Procrustes 2-D) and reindexing."""

        @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        def min_rotation_angle(cp1, cp2):
            # Find best angle of rotation from cp2 to cp1
            numer = jnp.sum(cp2[:, 0] * cp1[:, 1] - cp2[:, 1] * cp1[:, 0])
            denom = jnp.sum(cp2[:, 0] * cp1[:, 0] + cp2[:, 1] * cp1[:, 1])

            theta = jnp.arctan(numer / denom)

            rot_mat = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta),  jnp.cos(theta)]
            ])

            return (rot_mat @ cp2.T).T

        n_points = normed2.shape[0]
        indices = (jnp.arange(n_points).reshape(1, -1) + jnp.arange(n_points).reshape(-1, 1)) % n_points
        all_reindexed_normed2 = normed2[indices]

        if rotation:
            all_reindexed_normed2_rotated = min_rotation_angle(normed1, all_reindexed_normed2)
        else:
            all_reindexed_normed2_rotated = all_reindexed_normed2

        # Consider pairwise differences, take sum over all points, and find min index rotation.
        min_index = jnp.argmin(
            jnp.mean(
                jnp.linalg.norm(all_reindexed_normed2_rotated - normed1, axis=-1, ord=norm), axis=-1))
        min_dist = jnp.min(
            jnp.mean(
                jnp.linalg.norm(all_reindexed_normed2_rotated - normed1, axis=-1, ord=norm), axis=-1))

        return min_dist, all_reindexed_normed2_rotated[min_index]

    # Just visualize the neural network.
    rprint('NMA Neural Network:')
    rprint(hk.experimental.tabulate(nn_fn_t)(dummy_displacements))

    # Constructs the family of shapes we want to match.
    ds_save_path = os.path.join(
        args.exp_dir,
        f'sim-{args.exp_name}-dataset-omegaphi.pkl')
    shape_generator = get_shape_target_generator(
            config.shape_family, n_interior, config.shape_parameters,
            dataset_seed=config.dataset_seed, save_path=ds_save_path)
    fixed_target_shape = shape_generator()

    # For debugging. Only use a single shape throughout optimization.
    if config.debug_single_shape:
        rprint('Fixing target shape (DEBUGGING ONLY).')
        assert comm.Get_size() == 1, 'Fixed target shape meant for debugging only.'
        shape_generator = lambda: fixed_target_shape

    # Visualize dataset
    if comm.rank == 0:
        fig, ax = plt.subplots(config.num_ds_samples, 2)
        fig.set_size_inches(10, 5 * config.num_ds_samples)
        for i in range(config.num_ds_samples):
            target_shape = shape_generator()
            normalized_target_shape = normalize_pore_shape(target_shape)

            # Plot target shape
            ax[i][0].scatter(target_shape[:, 0], target_shape[:, 1])
            ax[i][0].scatter(target_shape[0, 0], target_shape[0, 1], c='red')
            ax[i][0].set_aspect('equal')

            ax[i][1].scatter(normalized_target_shape[:, 0], normalized_target_shape[:, 1])
            ax[i][1].scatter(normalized_target_shape[0, 0], normalized_target_shape[0, 1], c='red')
            ax[i][1].set_aspect('equal')

            target_image_path = os.path.join(
                args.exp_dir,
                f'sim-{args.exp_name}-fixed_target.png')
            fig.savefig(target_image_path)
    rprint('Generated shape dataset samples.')

    def loss_fn(all_params, cps):
        nn_params, (radii, internal_radii, mesh_perturb) = all_params

        # First normalize target
        normalized_target_cps = normalize_pore_shape(cps)

        if config.freeze_radii:
            radii = jax.lax.stop_gradient(radii)
            internal_radii = jax.lax.stop_gradient(internal_radii)

        # Encoder (with the option to freeze for debugging)
        if config.freeze_nn:
            mat_inputs = jnp.ones_like(central_pore_points.flatten()) * config.freeze_nn_val
        else:
            mat_inputs = nn_fn_t.apply(nn_params, normalized_target_cps.flatten())

        if not config.perturb_mesh:
            mesh_perturb = jax.lax.stop_gradient(mesh_perturb)

        # Decoder
        final_x_local, _ = simulate(mat_inputs, radii, internal_radii, mesh_perturb)

        # Get the resulting control points from middle pore and normalize shape.
        our_cps = get_central_pore_points(final_x_local)
        normalized_our_cps = normalize_pore_shape(our_cps)

        # If we want to penalize the smoothness of the outer and internal pore shapes.
        # Currently disabled.
        if config.radii_smoothness_penalty > 0.0:
            outer_smoothness_penalty = jnp.mean(jnp.square(jnp.diff(radii, axis=-1)))
            internal_smoothness_penalty = jnp.mean(jnp.square(jnp.diff(internal_radii)))
            total_smoothness_penalty = \
                    config.radii_smoothness_penalty * (outer_smoothness_penalty + internal_smoothness_penalty)
        else:
            total_smoothness_penalty = 0.0

        # Various types of loss functions we can use, depending on whether we want rotation, reindex invariance.
        # mse_rotation is enforces all of the invariances, and is the one we use.
        # We can also control the norm type. We use l1 for the experiments.
        if config.loss_type == 'mse':
            return jnp.mean(
                jnp.linalg.norm(normalized_target_cps - normalized_our_cps, axis=-1)
            ) + total_smoothness_penalty
        elif config.loss_type == 'mse_rotation':
            min_dist, _ = min_dist_rotation_reindexing(
                    normalized_target_cps, normalized_our_cps, norm=config.loss_norm)

            return min_dist + total_smoothness_penalty
        elif config.loss_type == 'mse_reindex':
            min_dist, _ = min_dist_rotation_reindexing(
                    normalized_target_cps, normalized_our_cps,
                    rotation=False, norm=config.loss_norm)

            return min_dist + total_smoothness_penalty
        elif config.loss_type == 'argmin':
            normalized_target_cps = normalized_target_cps.reshape(-1, 1, 2)
            normalized_our_cps = normalized_our_cps.reshape(1, -1, 2)
            dists = jnp.linalg.norm(normalized_target_cps - normalized_our_cps, axis=-1)

            return jnp.mean(jnp.argmin(dists, axis=-1)) + total_smoothness_penalty

    rprint(f'Starting NMA optimization...')

    # Scale the lr depending on the batch size.
    batch_size = comm.Get_size()
    lr = config.lr * batch_size

    nn_optimizer = optax.adam(lr)
    geometry_optimizer = optax.adam(lr * config.geometry_lr_multiplier)

    # The geometry parameters can use a different learning rate than the NN.
    param_labels = ('nn_radii', ('nn_radii', 'nn_radii', 'geometry'))
    optimizer = optax.multi_transform(
            {'nn_radii': nn_optimizer, 'geometry': geometry_optimizer},
            param_labels)

    all_losses = []
    all_ewa_losses = []
    curr_all_params = (init_nn_params, all_init_radii)

    # If reload is set, reload either the last checkpoint or the specified
    # args.load_iter. Otherwise start from scratch.
    load_iter = 0
    if args.reload:
        if args.load_iter < 0:
            # Load the latest checkpoint.
            all_ckpts = [f for f in os.listdir(args.exp_dir) if '.pkl' in f]

            if len(all_ckpts) == 0:
                rprint('No checkpoints found. Starting from scratch')
                load_iter = 0
            else:
                # Match the file name regex and extract the load iteration from file name.
                regex = r"sim-.+-params-([0-9]+).pkl"
                ckpt_nums = [int(re.match(regex, ckpt).group(1)) for ckpt in all_ckpts]
                load_iter = max(ckpt_nums)
                rprint(f'Loading from last iteration... iteration {load_iter}.')
        else:
            load_iter = args.load_iter

    if load_iter > 0:
        rprint('Loading parameters.')
        with open(os.path.join(args.exp_dir,
                               f'sim-{args.exp_name}-params-{load_iter}.pkl'), 'rb') as f:
            curr_all_params, all_losses, all_ewa_losses, opt_state, iter_num = pickle.load(f)
        assert load_iter == iter_num, 'Loaded iter_num didn\'t match load_iter.'
        rprint('\tDone.')
    else:
        iter_num = 0
        opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)
    ewa_loss = None

    comm.barrier()
    rprint(f'All processes at barrier.')
    for i in range(iter_num + 1, config.max_iter):
        iter_time = time.time()
        target_disps = onp.zeros((batch_size, n_interior, 2))
        for j in range(batch_size):
            target = shape_generator()
            target_disps[j] = target
        loss, grad_loss = loss_val_and_grad(curr_all_params, target_disps[comm.rank])
        avg_loss = pytree_reduce(loss, scale=1./batch_size)
        avg_grad_loss = pytree_reduce(grad_loss, scale=1./batch_size)
        step_time = time.time() - iter_time

        if comm.rank == 0:
            config.summary_writer.scalar('avg_loss', avg_loss, i)
            config.summary_writer.scalar('step_time', step_time, i)
            config.summary_writer.flush()

        ewa_loss = update_ewa(ewa_loss, avg_loss, config.ewa_weight)
        all_losses.append(avg_loss)
        all_ewa_losses.append(ewa_loss)

        rprint(f'Iteration {i} Loss: {avg_loss} '
               f'EWA Loss: {ewa_loss} '
               f'Time: {step_time}')

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)

        # Do appropriate clipping to bounds.
        curr_nn_params, (curr_radii, curr_internal_radii, curr_mesh_perturb) = curr_all_params
        curr_radii = jnp.clip(curr_radii, *config.radii_range)
        curr_internal_radii = jnp.clip(curr_internal_radii, *config.internal_radii_clip)
        curr_mesh_perturb = jnp.clip(curr_mesh_perturb, config.cell_length * config.mesh_perturb_range[0],
                                                        config.cell_length * config.mesh_perturb_range[1])
        curr_all_params = curr_nn_params, (curr_radii, curr_internal_radii, curr_mesh_perturb)

        if i % config.save_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(curr_all_params)

            if comm.rank == 0:
                # Pickle parameters
                rprint('Saving parameters.')
                with open(os.path.join(args.exp_dir,
                                       f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump((curr_all_params, all_losses, all_ewa_losses, opt_state, i), f)
                rprint('\tDone.')

        if i % config.eval_every == 0:
            # Generate video
            if comm.rank == 0:
                rprint(f'Generating image and video with optimization so far.')

                # Saving reference configuration
                curr_ref_ctrl = radii_to_ctrl_fn(curr_radii, curr_internal_radii, curr_mesh_perturb)
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(10, 10)

                plot_ctrl(ax, cell.element, curr_ref_ctrl)
                ax.set_aspect('equal')

                target_image_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-optimized-{i}-inspect-ref-config.png')
                fig.savefig(target_image_path)
                plt.close(fig)

                fig, ax = plt.subplots(config.num_eval, 4)
                fig.set_size_inches(20, 5 * config.num_eval)

                for trial in range(config.num_eval):
                    curr_nn_params, (curr_radii, curr_internal_radii, curr_mesh_perturb) = curr_all_params
                    target = shape_generator()
                    normalized_target_cps = normalize_pore_shape(target)
                    if config.freeze_nn:
                        mat_inputs = jnp.ones_like(central_pore_points.flatten()) * config.freeze_nn_val
                    else:
                        mat_inputs = nn_fn_t.apply(curr_nn_params, normalized_target_cps.flatten())
                    curr_ref_ctrl = radii_to_ctrl_fn(curr_radii, curr_internal_radii, curr_mesh_perturb)

                    # Run the simulation for eval.
                    final_x_local, ctrl_seq = \
                            simulate(mat_inputs, curr_radii, curr_internal_radii, curr_mesh_perturb)
                    our_cps = get_central_pore_points(final_x_local)
                    normalized_our_cps = normalize_pore_shape(our_cps)
                    _, min_dist_curve = min_dist_rotation_reindexing(
                            normalized_target_cps, normalized_our_cps)

                    # Plot target shape
                    ax[trial][0].scatter(normalized_target_cps[:, 0], normalized_target_cps[:, 1], c='blue', label='target')
                    ax[trial][0].scatter(normalized_target_cps[0, 0], normalized_target_cps[0, 1], c='red')

                    ax[trial][0].scatter(min_dist_curve[:, 0], min_dist_curve[:, 1], c='orange', label='ours')
                    ax[trial][0].scatter(min_dist_curve[0, 0], min_dist_curve[0, 1], c='purple')

                    ax[trial][0].set_aspect('equal')
                    ax[trial][0].legend()

                    ax[trial][1].bar(jnp.arange(config.n_disps), mat_inputs)

                    plot_ctrl(ax[trial][2], cell.element, final_x_local)
                    ax[trial][2].set_aspect('equal')

                    plot_ctrl(ax[trial][3], cell.element, curr_ref_ctrl)
                    ax[trial][3].set_aspect('equal')

                    # Visualize incremental displacement movie.
                    vid_path = os.path.join(
                            args.exp_dir,
                            f'sim-{args.exp_name}-optimized-{i}-trial-{trial}.mp4')
                    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

                config.summary_writer.plot(f'static_target', plt, step=i, close_plot=False)
                config.summary_writer.flush()

                target_image_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-optimized-{i}-static-trials.png')
                fig.savefig(target_image_path)
                plt.close(fig)

                # Plot losses
                loss_curve_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-loss.png')
                plt.plot(all_losses, label='loss')
                plt.plot(all_ewa_losses, label='EWA loss')
                plt.legend()
                plt.savefig(loss_curve_path)
                plt.close()
        comm.barrier()


if __name__ == '__main__':
    varmint.app.run(main)
