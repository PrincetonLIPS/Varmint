from absl import app
from absl import flags

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["JAX_PLATFORMS"] = "gpu,cpu"
import re
import sys
import time
import pickle

from functools import partial

from ml_collections import config_flags

import varmint.utils.experiment_utils as eutils
from varmint.utils.mpi_utils import *
from pore_shape_targets import get_shape_target_generator

import tensorflow_datasets as tfds

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

from jax.config import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
#import mpi4jax

from construct_digital_mnist_shape import generate_digital_mnist_shape, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D, NeoHookean2DClamped
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image, plot_ctrl
from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

import varmint.geometry.bsplines as bsplines

import optax
import haiku as hk

import matplotlib.pyplot as plt
import matplotlib as mpl
import varmint.utils.jaxboard as jaxboard

import digital_mnist_patches
from digital_mnist_movie_utils import create_movie_mnist


FLAGS = flags.FLAGS
eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/nma_v1/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/nma_v1/digital_mnist')

config_flags.DEFINE_config_file('config', 'config/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.3
    _density = 1.25


def main(argv):
    comm = MPI.COMM_WORLD
    rprint(f'Initializing MPI with JAX.', comm=comm)
    local_rank = find_local_rank(comm)
    dev_id = local_rank % len(jax.devices())

    if local_rank == 0:
        print(f'Node {MPI.Get_processor_name()} reporting with {len(jax.devices())} devices: {jax.devices()}', flush=True)

    args = FLAGS
    eutils.prepare_experiment_directories(args, comm)
    # args.seed and args.exp_dir should be set.

    config = args.config
    eutils.save_args(args, comm)
    npr.seed(config.seed)

    if comm.rank == 0:
        logdir = args.exp_dir
        summary_writer = jaxboard.SummaryWriter(logdir)

    if config.mat_model == 'NeoHookean2D':
        mat = NeoHookean2D(TPUMat)
        linear_mat = LinearElastic2D(TPUMat)
    elif config.mat_model == 'NeoHookean2DClamped':
        mat = NeoHookean2DClamped(TPUMat)
        linear_mat = LinearElastic2D(TPUMat)
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)
        linear_mat = LinearElastic2D(TPUMat)
    else:
        raise ValueError('Incorrect material model')

    cell, radii_to_ctrl_fn, n_cells = \
        generate_digital_mnist_shape(config, mat)

    init_radii = np.concatenate(
        (
            generate_rectangular_radii((n_cells,), config.ncp),
            #generate_circular_radii((n_cells,), config.ncp),
        )
    )
    rprint(f'radii: {init_radii.shape}', comm=comm)

    potential_energy_fn = cell.get_potential_energy_fn()
    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(), device=jax.devices()[dev_id])

    potential_energy_fn = jax.jit(potential_energy_fn, device=jax.devices()[dev_id])
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn, device=jax.devices()[dev_id])
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn, device=jax.devices()[dev_id])

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(init_radii)
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    # Initialize the color controls
    n_patches = ref_ctrl.shape[0]
    init_color_controls = np.zeros((n_patches, config.ncp, config.ncp, 1))
    color_eval_pts = bsplines.mesh(
            np.linspace(1e-4, 1-1e-4, 5 * config.ncp),
            np.linspace(1e-4, 1-1e-4, 5 * config.ncp))

    color_eval_pts = color_eval_pts.reshape((-1, 2))
    color_eval_fn = jax.jit(jax.vmap(cell.element.get_map_fn(color_eval_pts)), device=jax.devices()[dev_id])

    all_points_dict = digital_mnist_patches.get_all_points_dict(
            5, config.cell_size, config.border_size)

    mat_params = (
        TPUMat.E * np.ones(ref_ctrl.shape[0]),
        TPUMat.nu * np.ones(ref_ctrl.shape[0]),
    )

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, dev_id=dev_id,
                                              **config.solver_parameters)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn()

    def _radii_to_ref_and_init_x(radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)
        return ref_ctrl, init_x

    radii_to_ref_and_init_x = jax.jit(_radii_to_ref_and_init_x, device=jax.devices()[dev_id])
    fixed_locs_from_dict = jax.jit(cell.fixed_locs_from_dict, device=jax.devices()[dev_id])

    def simulate(disps, radii):
        ref_ctrl, current_x = radii_to_ref_and_init_x(radii)

        increment_dict = config.get_increment_dict(disps)
        current_x, all_xs, all_fixed_locs, _ = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    nn_fn = config.get_nn_fn(
            config.max_disp, config.n_disps)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.zeros((1, 28, 28, 1))
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    with open(config.nn_checkpoint, 'rb') as f:
        pretrained_nn_params = pickle.load(f)

    with open(config.material_checkpoint, 'rb') as f:
        curr_all_params, _, _, _ = pickle.load(f)
        _, curr_radii, curr_color_controls = curr_all_params

    #last_layer_w_init = np.zeros((10, config.n_disps)) + 1e-5
    #pretrained_nn_params['linear_3'] = {}
    #pretrained_nn_params['linear_3']['w'] = last_layer_w_init

    #init_nn_params = pretrained_nn_params

    def eval_color_score(pts, other_pts, other_pts_colors):
        width = 1.0

        pts = pts.reshape(-1, 1, 2)
        other_pts = other_pts.reshape(1, -1, 2)

        inv_dists = 1. / np.maximum(np.linalg.norm(pts - other_pts, axis=-1), 1e-5)
        softmax = jax.nn.softmax(config.softmax_temp * inv_dists)
        return np.mean(np.sum(other_pts_colors * softmax, axis=-1))

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
        digit = ds_element['label']
        color_params = jax.nn.sigmoid(color_params)
        if config.freeze_colors:
            color_params = jax.lax.stop_gradient(color_params)
        if config.freeze_radii:
            radii = jax.lax.stop_gradient(radii)
        if config.freeze_nn:
            mat_inputs = np.ones(config.n_disps) * config.freeze_nn_val
        else:
            mat_inputs = nn_fn_t.apply(
                    nn_params, ds_element['image'].reshape(1, 28, 28, 1)).squeeze()
            #mat_inputs = nn_params[digit]

        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(
                mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        other_pts_locs = color_eval_fn(final_x_local).reshape(-1, 2)
        other_pts_colors = color_eval_fn(color_params).flatten()

        pos_segments, neg_segments = digits_to_segments[digit]
        pos_segment_pts = [all_points_dict[s] for s in pos_segments]

        pos_segment_score = \
                sum(1 - eval_color_score(s, other_pts_locs, other_pts_colors) \
                        for s in pos_segment_pts)

        if digit != 8:
            neg_segment_pts = [all_points_dict[s] for s in neg_segments]
            neg_segment_score = \
                    sum(eval_color_score(s, other_pts_locs, other_pts_colors) \
                            for s in neg_segment_pts)

        if config.radii_smoothness_penalty > 0.0:
            outer_smoothness_penalty = np.mean(np.square(np.diff(radii, axis=-1)))
            total_smoothness_penalty = \
                    config.radii_smoothness_penalty * outer_smoothness_penalty
        else:
            total_smoothness_penalty = 0.0

        if digit != 8:
            return neg_segment_score + pos_segment_score
        else:
            return pos_segment_score

    rprint(f'Starting NMA optimization...', comm=comm)

    mpi_size = comm.Get_size()
    lr = config.lr * mpi_size

    optimizer = optax.adam(lr)

    all_losses = []

    #with open(config.material_checkpoint, 'rb') as f:
    #    curr_all_params, _, _, _ = pickle.load(f)
    #    _, curr_radii, curr_color_controls = curr_all_params
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
                rprint('No checkpoints found. Starting from scratch', comm=comm)
                load_iter = 0
            else:
                # Match the file name regex and extract the load iteration from file name.
                regex = r"sim-.+-params-([0-9]+).pkl"
                ckpt_nums = [int(re.match(regex, ckpt).group(1)) for ckpt in all_ckpts]
                load_iter = max(ckpt_nums)
                rprint(f'Loading from last iteration... iteration {load_iter}.', comm=comm)
        else:
            load_iter = args.load_iter

    if load_iter > 0:
        rprint('Loading parameters.', comm=comm)
        with open(os.path.join(args.exp_dir,
                               f'sim-{args.exp_name}-params-{load_iter}.pkl'), 'rb') as f:
            curr_all_params, all_losses, opt_state, iter_num = pickle.load(f)
        assert load_iter == iter_num, 'Loaded iter_num didn\'t match load_iter.'
        rprint('\tDone.', comm=comm)
    else:
        iter_num = 0
        opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)

    split = 'train'
    train_ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    train_ds = train_ds.shuffle(1000, seed=72)
    train_iterator = iter(tfds.as_numpy(train_ds))

    split = 'test'
    test_ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    test_iterator = iter(tfds.as_numpy(test_ds))

    ewa_loss = None
    ewa_weight = 0.95

    comm.barrier()
    rprint(f'All processes at barrier.', comm=comm)
    for i in range(iter_num + 1, 100000):
        iter_time = time.time()

        ds_elements = []
        for _ in range(mpi_size):
            ds_elements.append(next(train_iterator))

        loss, grad_loss = loss_val_and_grad(curr_all_params, ds_elements[comm.rank])
        avg_loss = pytree_reduce(loss, comm, scale=1./mpi_size)
        avg_grad_loss = pytree_reduce(grad_loss, comm, scale=1./mpi_size)
        step_time = time.time() - iter_time

        if comm.rank == 0:
            summary_writer.scalar('avg_loss', avg_loss, i)
            summary_writer.scalar('step_time', step_time, i)
            summary_writer.flush()

        all_losses.append(avg_loss)

        if ewa_loss == None:
            ewa_loss = avg_loss
        else:
            ewa_loss = ewa_loss * ewa_weight + avg_loss * (1 - ewa_weight)
        if comm.rank == 0:
            rprint(f'Iteration {i} Loss: {avg_loss} '
                   f'EWA Loss: {ewa_loss} '
                   f'Time: {step_time}', comm=comm)

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)
        curr_nn_params, curr_radii, curr_color_controls = curr_all_params

        if config.freeze_pretrain:
            pass
            #init_nn_params['linear_3']['w'] = curr_nn_params['linear_3']['w']
            #curr_nn_params = init_nn_params

        #rprint(f'NN params: {jax.tree_util.tree_map(lambda x: np.linalg.norm(x), curr_nn_params)}', comm=comm)

        if i % config.save_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(curr_all_params, comm)

            if comm.rank == 0:
                # Pickle parameters
                rprint('Saving parameters.', comm=comm)
                with open(os.path.join(args.exp_dir,
                                       f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump((curr_all_params, all_losses, opt_state, i), f)
                rprint('\tDone.', comm=comm)

        if i % config.eval_every == 0:
            # Generate video
            if comm.rank == 0:
                rprint(f'Generating image and video with optimization so far.', comm=comm)

                curr_radii = np.clip(curr_radii, config.radii_range[0], config.radii_range[1])
                curr_all_params = curr_nn_params, curr_radii, curr_color_controls

                # Saving figure after every iteration.
                curr_ref_ctrl = radii_to_ctrl_fn(curr_radii)
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(10, 10)

                plot_ctrl(ax, cell.element, curr_ref_ctrl)
                color_locs = color_eval_fn(curr_ref_ctrl)
                colors = color_eval_fn(jax.nn.sigmoid(curr_color_controls))
                ax.scatter(color_locs[..., 0], color_locs[..., 1], c=colors, s=7)

                digital_mnist_patches.draw_mpl_patches(ax, config.cell_size, config.border_size)
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

                    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = \
                            simulate(mat_inputs, curr_radii)
                    final_x_local = g2l(
                            optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii))

                    ax[trial][0].imshow(ds_element['image'])
                    plot_ctrl(ax[trial][1], cell.element, final_x_local)
                    color_locs = color_eval_fn(final_x_local)
                    colors = color_eval_fn(jax.nn.sigmoid(curr_color_controls))
                    ax[trial][1].scatter(color_locs[..., 0], color_locs[..., 1], c=colors, s=7)

                    digital_mnist_patches.draw_mpl_patches(ax[trial][1], config.cell_size, config.border_size, alpha=1.0)
                    ax[trial][1].set_aspect('equal')

                    all_velocities = np.zeros_like(all_displacements)
                    all_fixed_vels = np.zeros_like(all_fixed_locs)

                    # Visualize incremental displacement movie.
                    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                        all_displacements, all_velocities, all_fixed_locs,
                        all_fixed_vels, radii_to_ctrl_fn(curr_radii))

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

                summary_writer.plot(f'static_target', plt, step=i, close_plot=False)
                summary_writer.flush()

                # Plot losses
                loss_curve_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-loss.png')
                plt.plot(all_losses)
                plt.savefig(loss_curve_path)
                plt.close()
        comm.barrier()

if __name__ == '__main__':
    app.run(main)
