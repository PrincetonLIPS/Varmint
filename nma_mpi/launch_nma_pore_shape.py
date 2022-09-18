from absl import app
from absl import flags

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "gpu,cpu"
import re
import sys
import time
import pickle

from functools import partial

from ml_collections import config_flags

import experiment_utils as eutils
from mpi_utils import *
from pore_shape_targets import get_shape_target_generator

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

from jax.config import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
#import mpi4jax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from construct_pore_shape_shape import generate_pore_shapes_geometry, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.utils.movie_utils import create_movie, create_static_image, plot_ctrl
from varmintv2.solver.optimization_speed import SparseNewtonIncrementalSolver

import optax
import haiku as hk

import matplotlib.pyplot as plt
import matplotlib as mpl
import jaxboard


FLAGS = flags.FLAGS
eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/nma_mpi/experiments',
            source_root='n/fs/mm-iga/Varmint/nma_mpi')

config_flags.DEFINE_config_file('config', 'config/pore_shapes/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
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
    elif config.mat_model == 'LinearElastic2D':
        mat = LinearElastic2D(TPUMat)
        linear_mat = LinearElastic2D(TPUMat)
    else:
        raise ValueError('Incorrect material model')

    cell, radii_to_ctrl_fn, n_cells, get_central_pore_points, init_central_radii, init_mesh_perturb = \
        generate_pore_shapes_geometry(config, mat)

    init_radii = np.concatenate(
        (
            generate_rectangular_radii((n_cells,), config.ncp),
            #generate_circular_radii((n_cells,), config.ncp),
        )
    )
    all_init_radii = (init_radii, init_central_radii)
    rprint(f'radii: {init_radii.shape}', comm=comm)

    potential_energy_fn = cell.get_potential_energy_fn()
    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(), device=jax.devices()[dev_id])

    potential_energy_fn = jax.jit(potential_energy_fn, device=jax.devices()[dev_id])
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn, device=jax.devices()[dev_id])
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn, device=jax.devices()[dev_id])

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(*all_init_radii)
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    if config.mat_model == 'NeoHookean2D':
        mat_params = (
            TPUMat.shear * np.ones(ref_ctrl.shape[0]),
            TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
        )
        linear_mat_params = (
            TPUMat.lmbda * np.ones(ref_ctrl.shape[0]),
            TPUMat.mu * np.ones(ref_ctrl.shape[0]),
        )
    elif config.mat_model == 'LinearElastic2D':
        mat_params = (
            TPUMat.lmbda * np.ones(ref_ctrl.shape[0]),
            TPUMat.mu * np.ones(ref_ctrl.shape[0]),
        )
        linear_mat_params = (
            TPUMat.lmbda * np.ones(ref_ctrl.shape[0]),
            TPUMat.mu * np.ones(ref_ctrl.shape[0]),
        )
    else:
        raise ValueError('Incorrect material model')

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, dev_id=dev_id,
                                              **config.solver_parameters)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn()

    def _radii_to_ref_and_init_x(radii, central_radii):
        ref_ctrl = radii_to_ctrl_fn(radii, central_radii)
        init_x = l2g(ref_ctrl, ref_ctrl)
        return ref_ctrl, init_x
    
    radii_to_ref_and_init_x = jax.jit(_radii_to_ref_and_init_x, device=jax.devices()[dev_id])
    fixed_locs_from_dict = jax.jit(cell.fixed_locs_from_dict, device=jax.devices()[dev_id])

    def simulate(disps, radii, internal_radii):
        ref_ctrl, current_x = radii_to_ref_and_init_x(radii, internal_radii)

        increment_dict = config.get_increment_dict(disps)
        current_x, all_xs, all_fixed_locs = optimize(
                current_x, increment_dict, tractions, ref_ctrl, mat_params)

        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    nn_fn = config.get_nn_fn(
            config.max_disp, config.n_layers, config.n_activations, config.n_disps)
    central_pore_points = get_central_pore_points(ref_ctrl)

    def normalize_pore_shape(cps):
        center = np.mean(cps, axis=0)
        cps = cps - center
        norm = np.mean(np.linalg.norm(cps, axis=-1))

        return cps / norm

    def min_dist_rotation_reindexing(normed1, normed2):
        """Factor out rotation (using Procrustes 2-D) and reindexing."""

        @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        def min_rotation_angle(cp1, cp2):
            # Find best angle of rotation from cp2 to cp1
            numer = np.sum(cp2[:, 0] * cp1[:, 1] - cp2[:, 1] * cp1[:, 0])
            denom = np.sum(cp2[:, 0] * cp1[:, 0] + cp2[:, 1] * cp1[:, 1])

            theta = np.arctan(numer / denom)

            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])

            return (rot_mat @ cp2.T).T

        n_points = normed2.shape[0]
        indices = (np.arange(n_points).reshape(1, -1) + np.arange(n_points).reshape(-1, 1)) % n_points
        all_reindexed_normed2 = normed2[indices]

        all_reindexed_normed2_rotated = min_rotation_angle(normed1, all_reindexed_normed2)

        # Consider pairwise differences, take sum over all points, and find min index rotation.
        min_index = np.argmin(
            np.mean(
                np.linalg.norm(all_reindexed_normed2_rotated - normed1, axis=-1), axis=-1))
        min_dist = np.min(
            np.mean(
                np.linalg.norm(all_reindexed_normed2_rotated - normed1, axis=-1), axis=-1))

        return min_dist, all_reindexed_normed2_rotated[min_index]

    n_interior = central_pore_points.shape[0]
    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = central_pore_points.flatten()
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    rprint('NMA Neural Network:', comm=comm)
    rprint(hk.experimental.tabulate(nn_fn_t)(dummy_displacements), comm=comm)

    shape_generator = get_shape_target_generator(
            config.shape_family, n_interior, config.shape_parameters)
    fixed_target_shape = shape_generator()

    if config.debug_single_shape:
        rprint('Fixing target shape (DEBUGGING ONLY).', comm=comm)
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
    rprint('Generated shape dataset samples.', comm=comm)

    def loss_fn(all_params, cps):
        normalized_target_cps = normalize_pore_shape(cps)

        nn_params, (radii, internal_radii) = all_params
        if config.freeze_radii:
            radii = jax.lax.stop_gradient(radii)
            internal_radii = jax.lax.stop_gradient(internal_radii)
        if config.freeze_nn:
            mat_inputs = np.ones_like(central_pore_points.flatten()) * config.freeze_nn_val
        else:
            mat_inputs = nn_fn_t.apply(nn_params, normalized_target_cps.flatten())

        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(
                mat_inputs, radii, internal_radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii, internal_radii))
        our_cps = get_central_pore_points(final_x_local)
        normalized_our_cps = normalize_pore_shape(our_cps)

        if config.loss_type == 'mse':
            return np.mean(np.linalg.norm(normalized_target_cps - normalized_our_cps, axis=-1))
        elif config.loss_type == 'mse_rotation':
            min_dist, _ = min_dist_rotation_reindexing(
                    normalized_target_cps, normalized_our_cps)
            return min_dist
        elif config.loss_type == 'argmin':
            normalized_target_cps = normalized_target_cps.reshape(-1, 1, 2)
            normalized_our_cps = normalized_our_cps.reshape(1, -1, 2)

            dists = np.linalg.norm(normalized_target_cps - normalized_our_cps, axis=-1)
            return np.mean(np.argmin(dists, axis=-1))

    rprint(f'Starting NMA optimization...', comm=comm)

    mpi_size = comm.Get_size()
    lr = config.lr * mpi_size

    optimizer = optax.adam(lr)

    all_losses = []
    curr_all_params = (init_nn_params, all_init_radii)

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

    ewa_loss = None
    ewa_weight = 0.95

    comm.barrier()
    rprint(f'All processes at barrier.', comm=comm)
    for i in range(iter_num + 1, 100000):
        iter_time = time.time()
        target_disps = onp.zeros((mpi_size, n_interior, 2))
        for j in range(mpi_size):
            target = shape_generator()
            target_disps[j] = target
        loss, grad_loss = loss_val_and_grad(curr_all_params, target_disps[comm.rank])
        avg_loss = pytree_reduce(comm, loss, scale=1./mpi_size)
        avg_grad_loss = pytree_reduce(comm, grad_loss, scale=1./mpi_size)
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
        curr_nn_params, (curr_radii, curr_internal_radii) = curr_all_params
        curr_radii = np.clip(curr_radii, config.radii_range[0], config.radii_range[1])
        curr_internal_radii = np.clip(curr_internal_radii, config.internal_radii_clip[0], config.internal_radii_clip[1])
        curr_all_params = curr_nn_params, (curr_radii, curr_internal_radii)

        if i % config.save_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(comm, curr_all_params)

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
                fig, ax = plt.subplots(config.num_eval, 4)
                fig.set_size_inches(20, 5 * config.num_eval)

                for trial in range(config.num_eval):
                    curr_nn_params, (curr_radii, curr_internal_radii) = curr_all_params
                    target = shape_generator()
                    normalized_target_cps = normalize_pore_shape(target)
                    if config.freeze_nn:
                        mat_inputs = np.ones_like(central_pore_points.flatten()) * config.freeze_nn_val
                    else:
                        mat_inputs = nn_fn_t.apply(curr_nn_params, normalized_target_cps.flatten())
                    curr_ref_ctrl = radii_to_ctrl_fn(curr_radii, curr_internal_radii)

                    optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = \
                            simulate(mat_inputs, curr_radii, curr_internal_radii)
                    final_x_local = g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii, curr_internal_radii))
                    our_cps = get_central_pore_points(final_x_local)
                    normalized_our_cps = normalize_pore_shape(our_cps)
                    _, min_dist_curve = min_dist_rotation_reindexing(
                            normalized_target_cps, normalized_our_cps)

                    all_velocities = np.zeros_like(all_displacements)
                    all_fixed_vels = np.zeros_like(all_fixed_locs)

                    # Plot target shape
                    ax[trial][0].scatter(normalized_target_cps[:, 0], normalized_target_cps[:, 1], c='blue', label='target')
                    ax[trial][0].scatter(normalized_target_cps[0, 0], normalized_target_cps[0, 1], c='red')

                    ax[trial][0].scatter(min_dist_curve[:, 0], min_dist_curve[:, 1], c='orange', label='ours')
                    ax[trial][0].scatter(min_dist_curve[0, 0], min_dist_curve[0, 1], c='purple')

                    ax[trial][0].set_aspect('equal')
                    ax[trial][0].legend()

                    ax[trial][1].bar(np.arange(config.n_disps), mat_inputs)

                    plot_ctrl(ax[trial][2], cell.element,
                              g2l(optimized_curr_g_pos, all_fixed_locs[-1],
                                  radii_to_ctrl_fn(curr_radii, curr_internal_radii)))
                    ax[trial][2].set_aspect('equal')

                    plot_ctrl(ax[trial][3], cell.element, curr_ref_ctrl)
                    ax[trial][3].set_aspect('equal')

                    # Visualize incremental displacement movie.
                    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                        all_displacements, all_velocities, all_fixed_locs,
                        all_fixed_vels, radii_to_ctrl_fn(curr_radii, curr_internal_radii))

                    vid_path = os.path.join(
                            args.exp_dir,
                            f'sim-{args.exp_name}-optimized-{i}-trial-{trial}.mp4')
                    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

                summary_writer.plot(f'static_target', plt, step=i, close_plot=False)
                summary_writer.flush()

                target_image_path = os.path.join(
                    args.exp_dir,
                    f'sim-{args.exp_name}-optimized-{i}-static-trials.png')
                fig.savefig(target_image_path)
                plt.close(fig)

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
