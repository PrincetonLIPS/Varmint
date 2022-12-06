import os
import sys
import time
import pickle

import varmint

from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

from varmint.utils.mpi_utils import rprint, pytree_reduce, test_pytrees_equal
from varmint.utils.train_utils import update_ewa

from rotation_geometry import construct_cell2D, generate_rectangular_radii
from rotation_plotting import create_movie_nma, create_static_image_nma

import optax
import haiku as hk

import jax
import jax.numpy as jnp
import numpy as onp

import matplotlib.pyplot as plt


FLAGS = varmint.flags.FLAGS
varmint.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/nma_v1/rotation/experiments',
            source_root='/n/fs/mm-iga/Varmint/projects/nma_v1/rotation')

varmint.config_flags.DEFINE_config_file('config', 'config/single_rotation.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.30
    _density = 1.25


def main(argv):
    args, dev_id, local_rank = varmint.initialize_experiment(verbose=True)
    config = args.config
    comm = varmint.MPI.COMM_WORLD

    mat = NeoHookean2D(TPUMat)

    # Construct geometry function along with initial radii parameters.
    cell, radii_to_ctrl_fn, n_cells, init_mesh_perturb = \
        construct_cell2D(input_str=config.grid_str, patch_ncp=config.ncp,
                         quad_degree=config.quad_deg, spline_degree=config.spline_deg,
                         material=mat)
    init_radii = jnp.concatenate((
            generate_rectangular_radii((n_cells,), config.ncp),
    ))

    # Initialization of local-global transformations, reference control points, tractions.
    potential_energy_fn = cell.get_potential_energy_fn()
    l2g, g2l = cell.get_global_local_maps()
    ref_ctrl = radii_to_ctrl_fn(jnp.array(init_radii))
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
    def simulate(disps, radii):
        ref_ctrl = radii_to_ctrl_fn(radii)

        # The optimizer works in the global configuration.
        current_x = l2g(ref_ctrl, ref_ctrl)

        increment_dict = config.get_increment_dict(disps)
        current_x, all_xs, all_fixed_locs, solved_increment = \
                optimize(current_x, increment_dict, tractions, ref_ctrl, mat_params)

        # Unflatten sequence to local configuration.
        ctrl_seq = cell.unflatten_sequence(
            all_xs, all_fixed_locs, ref_ctrl)
        final_x_local = g2l(current_x, all_fixed_locs[-1], ref_ctrl)

        return final_x_local, [ref_ctrl] + ctrl_seq

    # Initialize neural network (encoder).
    nn_fn = config.get_nn_fn(
            config.max_disp, config.n_layers, config.n_activations, config.n_disps)
    nn_fn_t = hk.without_apply_rng(hk.transform(nn_fn))
    init_nn_params = nn_fn_t.init(config.jax_rng, jnp.array([0.0]))

    # Gather all NMA parameters into a pytree.
    curr_all_params = (init_nn_params, init_radii)

    # To emulate rotation, our objective function is the rotation of two points about their center.
    p1 = jnp.sum(jnp.abs(radii_to_ctrl_fn(init_radii) - config.left_point), axis=-1) < 1e-14
    p2 = jnp.sum(jnp.abs(radii_to_ctrl_fn(init_radii) - config.right_point), axis=-1) < 1e-14
    ps = [p1, p2]

    def angle_to_points(angle):
        rot = jnp.array([
            [jnp.cos(angle[0]), -jnp.sin(angle[0])],
            [jnp.sin(angle[0]),  jnp.cos(angle[0])],
        ])

        r = config.right_point
        l = config.left_point
        c = config.center_point

        return rot @ (l - c) + c, rot @ (r - c) + c

    def loss_fn(all_params, angle):
        target_l, target_r = angle_to_points(angle)

        nn_params, radii = all_params
        
        # Encoder
        mat_inputs = nn_fn_t.apply(nn_params, angle)

        # Decoder
        final_x_local, _ = simulate(mat_inputs, radii)

        # We want our identified points at specified locations determined by angle.
        return jnp.sum(jnp.abs(final_x_local[p1] - target_l)) / ref_ctrl[p1].shape[0] + \
                jnp.sum(jnp.abs(final_x_local[p2] - target_r)) / ref_ctrl[p2].shape[0]

    all_losses = []
    all_ewa_losses = []

    # Reload parameters if needed.
    if args.reload:
        rprint('Loading parameters.')
        with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{args.load_iter}.pkl'), 'rb') as f:
            curr_all_params, all_losses, all_ewa_losses = pickle.load(f)
        rprint('\tDone.')

    # Scale the lr depending on the batch size.
    batch_size = comm.Get_size()
    lr = config.lr * batch_size

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)
    ewa_loss = None

    comm.barrier()
    rprint(f'All processes at barrier. Starting training over {batch_size} MPI ranks.')
    for i in range(args.load_iter + 1, config.max_iter):
        # Do one training step, averaging gradients over MPI ranks.
        iter_time = time.time()
        target_angles = onp.random.uniform(*config.angle_range, size=(batch_size, 1))

        loss, grad_loss = loss_val_and_grad(curr_all_params, target_angles[comm.rank])
        avg_loss = pytree_reduce(loss, scale=1./batch_size)
        avg_grad_loss = pytree_reduce(grad_loss, scale=1./batch_size)
        ewa_loss = update_ewa(ewa_loss, avg_loss, config.ewa_weight)
        rprint(f'Iteration {i} Loss: {avg_loss} EWA Loss: {ewa_loss} Time: {time.time() - iter_time}')

        all_losses.append(avg_loss)
        all_ewa_losses.append(ewa_loss)

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)

        # Clip radii
        curr_nn_params, curr_radii = curr_all_params
        curr_radii = jnp.clip(curr_radii, *config.radii_range)
        curr_all_params = curr_nn_params, curr_radii

        if i % config.eval_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(curr_all_params)

            # Generate video
            if comm.rank == 0:
                # Generate two samples, one from max angle and one from min.
                for is_max, test_angle in enumerate(config.angle_range):
                    rprint(f'Generating image and video with optimization so far.')
                    curr_nn_params, curr_radii = curr_all_params

                    # Convert test angle to points.
                    test_pts = angle_to_points(jnp.array([test_angle]))

                    # Simulate with the test input.
                    mat_inputs = nn_fn_t.apply(curr_nn_params, jnp.array([test_angle]))
                    _, ctrl_seq = simulate(mat_inputs, curr_radii)

                    # Save static image and deformation sequence video.
                    angle_label = 'max' if is_max == 1 else 'min'
                    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref_config-{i}-{angle_label}-angle.png')
                    create_static_image_nma(cell.element, ctrl_seq[0], image_path, test_pts, ps=ps)

                    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}-{angle_label}-angle.png')
                    create_static_image_nma(cell.element, ctrl_seq[-1], image_path, test_pts, ps=ps)

                    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}-{angle_label}-angle.mp4')
                    create_movie_nma(cell.element, ctrl_seq, vid_path, test_pts, ps=ps)

                # Export graph of losses
                loss_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-loss.png')
                plt.title('Loss over iterations')
                plt.plot(all_losses, label='loss')
                plt.plot(all_ewa_losses, label='EWA loss')
                plt.legend()
                plt.savefig(loss_path)
                plt.close()

        if i % config.save_every == 0:
            # Pickle parameters
            if comm.rank == 0:
                rprint('Saving parameters.')
                with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump((curr_all_params, all_losses, all_ewa_losses), f)
                rprint('\tDone.')

        comm.barrier()


if __name__ == '__main__':
    varmint.app.run(main)

