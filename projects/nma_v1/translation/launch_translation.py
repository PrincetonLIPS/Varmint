from absl import app
from absl import flags

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import time
import pickle

from ml_collections import config_flags

import numpy.random as npr
import numpy as onp
import jax.numpy as jnp
import jax

from jax.config import config
config.update("jax_enable_x64", True)

from mpi4py import MPI

from translation_geometry import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from translation_plotting import create_movie_nma, create_static_image_nma

from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

from varmint.utils import experiment_utils as eutils
from varmint.utils.mpi_utils import *
from varmint.utils.train_utils import update_ewa

import optax
import haiku as hk

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/nma_v1/translation/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/nma_v1/translation')

config_flags.DEFINE_config_file('config', 'config/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.30
    _density = 1.25


def main(argv):
    args, dev_id, local_rank = eutils.initialize_experiment(verbose=True)
    config = args.config
    comm = MPI.COMM_WORLD

    mat = NeoHookean2D(TPUMat)

    # Construct geometry function along with initial radii parameters.
    cell, radii_to_ctrl_fn, n_cells = \
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

    # Differentiable simulation function for a given displacements and radii (decoder).
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
            config.max_disp, config.n_layers, config.n_activations, config.n_disps, config.start_point)
    nn_fn_t = hk.without_apply_rng(hk.transform(nn_fn))
    init_nn_params = nn_fn_t.init(config.jax_rng, config.start_point)

    # Gather all NMA parameters into a pytree.
    curr_all_params = (init_nn_params, init_radii)

    # Target point
    p1 = jnp.sum(jnp.abs(radii_to_ctrl_fn(init_radii) - config.start_point), axis=-1) < 1e-14

    def loss_fn(all_params, target):
        nn_params, radii = all_params

        # Encoder
        mat_inputs = nn_fn_t.apply(nn_params, target)

        # Decoder
        final_x_local, _ = simulate(mat_inputs, radii)

        # We want our identified point (p1) at a specified location (target).
        return jnp.sum(jnp.abs(final_x_local[p1] - target)) / ref_ctrl[p1].shape[0]

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
        target_disps = jnp.array(onp.random.uniform(*config.target_range, size=(batch_size, 2)))

        loss, grad_loss = loss_val_and_grad(curr_all_params, target_disps[comm.rank])
        avg_loss = pytree_reduce(loss, scale=1./batch_size)
        avg_grad_loss = pytree_reduce(grad_loss, scale=1./batch_size)
        ewa_loss = update_ewa(ewa_loss, avg_loss, config.ewa_weight)
        rprint(f'Iteration {i} Loss: {avg_loss} EWA Loss: {ewa_loss} Time: {time.time() - iter_time}')

        all_losses.append(avg_loss)
        all_ewa_losses.append(ewa_loss)

        updates, opt_state = optimizer.update(avg_grad_loss, opt_state)
        curr_all_params = optax.apply_updates(curr_all_params, updates)

        if i % config.eval_every == 0:
            # Verify that the parameters have not deviated between different MPI ranks.
            test_pytrees_equal(curr_all_params)

            # Generate video
            if comm.rank == 0:
                rprint(f'Generating image and video with optimization so far.')
                curr_nn_params, curr_radii = curr_all_params

                # Sample a random test point.
                test_disps = onp.random.uniform(*config.target_range, size=(2,))

                # Simulate with the test input.
                mat_inputs = nn_fn_t.apply(curr_nn_params, test_disps)
                _, ctrl_seq = simulate(mat_inputs, curr_radii)

                # Save static image and deformation sequence video.
                image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-ref_config-{i}.png')
                create_static_image_nma(cell.element, ctrl_seq[0], image_path, test_disps)

                image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
                create_static_image_nma(cell.element, ctrl_seq[-1], image_path, test_disps)

                vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
                create_movie_nma(cell.element, ctrl_seq, vid_path, test_disps, p1=p1)

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
    app.run(main)
