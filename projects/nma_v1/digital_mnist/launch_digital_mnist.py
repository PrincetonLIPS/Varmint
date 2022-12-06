import os
import re
import sys
import time
import pickle

import varmint

from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D, NeoHookean2DClamped
from varmint.physics.materials import Material
from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver
import varmint.geometry.bsplines as bsplines

from varmint.utils.mpi_utils import rprint, pytree_reduce, test_pytrees_equal
from varmint.utils.train_utils import update_ewa
from varmint.utils.movie_utils import plot_ctrl

from digital_mnist_geometry import generate_digital_mnist_shape, generate_rectangular_radii
from digital_mnist_patches import get_all_points_dict, draw_mpl_patches
from digital_mnist_movie_utils import create_movie_mnist

import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import numpy as onp

import optax
import haiku as hk

import matplotlib.pyplot as plt


FLAGS = varmint.flags.FLAGS
varmint.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/nma_v1/digital_mnist/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/nma_v1/digital_mnist')

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
    cell, radii_to_ctrl_fn, n_cells = \
        generate_digital_mnist_shape(config, mat)
    init_radii = jnp.concatenate((
            generate_rectangular_radii((n_cells,), config.ncp),
    ))

    # Initialization of local-global transformations, reference control points, tractions.
    potential_energy_fn = cell.get_potential_energy_fn()
    l2g, g2l = cell.get_global_local_maps()
    ref_ctrl = radii_to_ctrl_fn(init_radii)
    tractions = cell.tractions_from_dict({})

    # Initialize the color controls
    n_patches = ref_ctrl.shape[0]
    init_color_controls = jnp.zeros((n_patches, config.ncp, config.ncp, 1))
    color_eval_pts = bsplines.mesh(
            jnp.linspace(1e-4, 1-1e-4, 5 * config.ncp),
            jnp.linspace(1e-4, 1-1e-4, 5 * config.ncp))

    # Color is controlled via a BSpline surface over the material.
    color_eval_pts = color_eval_pts.reshape((-1, 2))
    color_eval_fn = jax.jit(jax.vmap(cell.element.get_map_fn(color_eval_pts)), device=jax.devices()[dev_id])

    all_points_dict = get_all_points_dict(5, config.cell_size, config.border_size)

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
    def simulate(disps, radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
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
            config.max_disp, config.n_disps)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = jnp.zeros((1, 28, 28, 1))
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    if config.init_from_ckpt:
        with open(config.nn_checkpoint, 'rb') as f:
            pretrained_nn_params = pickle.load(f)

        with open(config.material_checkpoint, 'rb') as f:
            curr_all_params, _, _, _ = pickle.load(f)
            _, curr_radii, curr_color_controls = curr_all_params

    def eval_color_score(pts, other_pts, other_pts_colors):
        width = 1.0

        pts = pts.reshape(-1, 1, 2)
        other_pts = other_pts.reshape(1, -1, 2)

        inv_dists = 1. / jnp.maximum(jnp.linalg.norm(pts - other_pts, axis=-1), 1e-5)
        softmax = jax.nn.softmax(config.softmax_temp * inv_dists)
        return jnp.mean(jnp.sum(other_pts_colors * softmax, axis=-1))

    # For each digit, manually specify the segments it must have.
    digits_to_segments = [
            (['1', '2', '3', '4', '5', '7'], ['6']),  # 0
            (['3', '4'], ['1', '2', '5', '6', '7']),  # 1
            (['1', '4', '5', '6', '7'], ['2', '3']),  # 2
            (['3', '4', '5', '6', '7'], ['1', '2']),  # 3
            (['2', '3', '4', '6'], ['1', '5', '7']),  # 4
            (['2', '3', '5', '6', '7'], ['1', '4']),  # 5
            (['1', '2', '3', '5', '6', '7'], ['4']),  # 6
            (['3', '4', '7'], ['1', '2', '5', '6']),  # 7
            (['1', '2', '3', '4', '5', '6', '7'], []),  # 8
            (['2', '3', '4', '6', '7'], ['1', '5']),  # 9
    ]

    def loss_fn(all_params, ds_element):
        nn_params, radii, color_params = all_params
        color_params = jax.nn.sigmoid(color_params)
        if config.freeze_colors:
            color_params = jax.lax.stop_gradient(color_params)
        if config.freeze_radii:
            radii = jax.lax.stop_gradient(radii)

        # Encoder
        if config.freeze_nn:
            mat_inputs = jnp.ones(config.n_disps) * config.freeze_nn_val
        else:
            mat_inputs = nn_fn_t.apply(
                    nn_params, ds_element['image'].reshape(1, 28, 28, 1)).squeeze()

        # Decoder
        final_x_local, _ = simulate(mat_inputs, radii)

        other_pts_locs = color_eval_fn(final_x_local).reshape(-1, 2)
        other_pts_colors = color_eval_fn(color_params).flatten()

        digit = ds_element['label']
        pos_segments, neg_segments = digits_to_segments[digit]
        pos_segment_pts = [all_points_dict[s] for s in pos_segments]

        # Encourage color in the appropriate segments.
        pos_segment_score = \
                sum(1 - eval_color_score(s, other_pts_locs, other_pts_colors) \
                        for s in pos_segment_pts)

        # Penalize color if it is in a segment it shouldn't be.
        if digit != 8:
            neg_segment_pts = [all_points_dict[s] for s in neg_segments]
            neg_segment_score = \
                    sum(eval_color_score(s, other_pts_locs, other_pts_colors) \
                            for s in neg_segment_pts)
        else:
            neg_segment_score = 0.0

        # Radii smoothness if desired.
        if config.radii_smoothness_penalty > 0.0:
            outer_smoothness_penalty = jnp.mean(jnp.square(jnp.diff(radii, axis=-1)))
            total_smoothness_penalty = \
                    config.radii_smoothness_penalty * outer_smoothness_penalty
        else:
            total_smoothness_penalty = 0.0

        # It's not actually a score, it's a loss. Minimize this.
        return neg_segment_score + pos_segment_score

    rprint(f'Starting NMA optimization...')

    batch_size = comm.Get_size()
    lr = config.lr * batch_size

    optimizer = optax.adam(lr)

    all_losses = []
    all_ewa_losses = []

    # Collect all parameters. Can pre-initialize NN and geometry, or start from
    # scratch. Pre-initializing can help do better, but from scratch you can get
    # good results too.
    if config.init_from_ckpt:
        curr_all_params = (pretrained_nn_params, curr_radii, curr_color_controls)
    else:
        curr_all_params = (init_nn_params, init_radii, init_color_controls)

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
            curr_all_params, all_losses, opt_state, iter_num = pickle.load(f)
        assert load_iter == iter_num, 'Loaded iter_num didn\'t match load_iter.'
        rprint('\tDone.')
    else:
        iter_num = 0
        opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)

    # In order for the different MPI ranks not to clash with file operations, do a pre-load.
    if comm.rank == 0:
        _ = tfds.load("mnist:3.*.*", split='train', data_dir=args.source_root).cache().repeat()
        _ = tfds.load("mnist:3.*.*", split='test',  data_dir=args.source_root).cache().repeat()
    comm.barrier()

    train_ds = tfds.load("mnist:3.*.*", split='train', data_dir='mnist_dataset').cache().repeat()
    train_ds = train_ds.shuffle(1000, seed=72)
    train_iterator = iter(tfds.as_numpy(train_ds))

    test_ds = tfds.load("mnist:3.*.*", split='test', data_dir='mnist_dataset').cache().repeat()
    test_iterator = iter(tfds.as_numpy(test_ds))

    ewa_loss = None

    comm.barrier()
    rprint(f'All processes at barrier.')
    for i in range(iter_num + 1, config.max_iters):
        iter_time = time.time()

        ds_elements = []
        for _ in range(batch_size):
            ds_elements.append(next(train_iterator))

        loss, grad_loss = loss_val_and_grad(curr_all_params, ds_elements[comm.rank])
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

        if comm.rank == 0:
            rprint(f'Iteration {i} Loss: {avg_loss} '
                   f'EWA Loss: {ewa_loss} '
                   f'Time: {step_time}')

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)
        curr_nn_params, curr_radii, curr_color_controls = curr_all_params
        curr_radii = jnp.clip(curr_radii, *config.radii_range)
        curr_all_params = curr_nn_params, curr_radii, curr_color_controls

        if i % config.save_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(curr_all_params)

            if comm.rank == 0:
                # Pickle parameters
                rprint('Saving parameters.')
                with open(os.path.join(args.exp_dir,
                                       f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump((curr_all_params, all_losses, opt_state, i), f)
                rprint('\tDone.')

        if i % config.eval_every == 0:
            # Generate video
            if comm.rank == 0:
                rprint(f'Generating image and video with optimization so far.')

                # Saving figure in reference configuration.
                curr_ref_ctrl = radii_to_ctrl_fn(curr_radii)
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(10, 10)

                plot_ctrl(ax, cell.element, curr_ref_ctrl)
                color_locs = color_eval_fn(curr_ref_ctrl)
                colors = color_eval_fn(jax.nn.sigmoid(curr_color_controls))
                ax.scatter(color_locs[..., 0], color_locs[..., 1], c=colors, s=7)

                draw_mpl_patches(ax, config.cell_size, config.border_size)
                ax.set_aspect('equal')

                target_image_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-optimized-{i}-inspect-ref-config.png')
                fig.savefig(target_image_path)
                plt.close(fig)

                fig, ax = plt.subplots(config.num_trials, 2)
                fig.set_size_inches(2 * 10, config.num_trials * 10)
                for trial in range(config.num_trials):
                    ds_element = next(test_iterator)
                    curr_nn_params, curr_radii, curr_color_controls = curr_all_params
                    mat_inputs = nn_fn_t.apply(
                            curr_nn_params, ds_element['image'].reshape(1, 28, 28, 1)).squeeze()
                    curr_ref_ctrl = radii_to_ctrl_fn(curr_radii)

                    final_x_local, ctrl_seq = simulate(mat_inputs, curr_radii)

                    ax[trial][0].imshow(ds_element['image'])
                    plot_ctrl(ax[trial][1], cell.element, final_x_local)
                    color_locs = color_eval_fn(final_x_local)
                    colors = color_eval_fn(jax.nn.sigmoid(curr_color_controls))
                    ax[trial][1].scatter(color_locs[..., 0], color_locs[..., 1], c=colors, s=7)

                    draw_mpl_patches(ax[trial][1], config.cell_size, config.border_size, alpha=1.0)
                    ax[trial][1].set_aspect('equal')

                    # Visualize incremental displacement movie.
                    vid_path = os.path.join(
                            args.exp_dir,
                            f'sim-{args.exp_name}-optimized-{i}-trial-{trial}.mp4')
                    create_movie_mnist(
                            config, cell.element, ctrl_seq,
                            vid_path, curr_color_controls, color_eval_fn)

                target_image_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-optimized-{i}-static-trials-config.png')
                fig.savefig(target_image_path)
                plt.close(fig)

                config.summary_writer.plot(f'static_target', plt, step=i, close_plot=False)
                config.summary_writer.flush()

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
