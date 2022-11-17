from absl import app
from absl import flags

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import time
import pickle

from ml_collections import config_flags

import experiment_utils as eutils
from mpi_utils import *

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

from construct_nma_shape import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.utils.movie_utils import create_movie_mnist, create_static_image_nma
from varmintv2.solver.optimization_speed import SparseNewtonIncrementalSolver

import varmintv2.geometry.bsplines as bsplines

import optax
import haiku as hk

import matplotlib.pyplot as plt

import get_mnist


FLAGS = flags.FLAGS
eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/nma_mpi/experiments',
            source_root='n/fs/mm-iga/Varmint/nma_mpi')

config_flags.DEFINE_config_file('config', 'config/default.py')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
    _density = 1.25


def main(argv):
    comm = MPI.COMM_WORLD
    rprint(f'Initializing MPI with JAX. Visible JAX devices: {jax.devices()}', comm=comm)
    rprint(f'There are {len(jax.devices())} devices available.', comm=comm)
    local_rank = find_local_rank(comm)
    dev_id = local_rank % len(jax.devices())
    print(f'Using GPU {dev_id} on node {MPI.Get_processor_name()}.')

    args = FLAGS
    eutils.prepare_experiment_directories(args, comm)
    # args.seed and args.exp_dir should be set.

    config = args.config
    eutils.save_args(args, comm)
    npr.seed(config.seed)

    mat = NeoHookean2D(TPUMat)

    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=config.grid_str, patch_ncp=config.ncp,
                         quad_degree=config.quad_deg, spline_degree=config.spline_deg,
                         material=mat)

    init_radii = np.concatenate(
        (
            generate_rectangular_radii((n_cells,), config.ncp),
        )
    )

    num_examples = 1
    some_mnist = None
    if comm.rank == 0:
        _, test = get_mnist.mnist()
        test = test.reshape((-1, 28, 28))
        some_mnist = test[1:(num_examples+1)]
    comm.bcast(some_mnist, root=0)

    potential_energy_fn = cell.get_potential_energy_fn()
    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(), device=jax.devices()[dev_id])

    potential_energy_fn = jax.jit(potential_energy_fn, device=jax.devices()[dev_id])
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn, device=jax.devices()[dev_id])
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn, device=jax.devices()[dev_id])

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(np.array(init_radii))
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    # Initialize the color controls
    n_patches = ref_ctrl.shape[0]
    init_color_controls = np.zeros((n_patches, config.ncp, config.ncp, 1))
    color_eval_pts = bsplines.mesh(np.linspace(1e-8, 1-1e-8, 10), np.linspace(1e-8, 1-1e-8, 10))
    color_eval_pts = color_eval_pts.reshape((-1, 2))
    color_eval_fn = jax.jit(jax.vmap(cell.element.get_map_fn(color_eval_pts)), device=jax.devices()[dev_id])

    mat_params = (
        TPUMat.shear * np.ones(ref_ctrl.shape[0]),
        TPUMat.bulk * np.ones(ref_ctrl.shape[0]),
    )

    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_fn, max_iter=1000,
                                              step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, dev_id=dev_id)

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

        increment_dict = {
            '99': np.array([0.0, 0.0]),
            '98': np.array([0.0, 0.0]),
            '97': np.array([0.0, 0.0]),
            '96': np.array([0.0, 0.0]),
            '2': np.array([-disps[0], 0.0]),
            '3': np.array([0.0, -disps[1]]),
            '4': np.array([-disps[2], 0.0]),
            '5': np.array([0.0, -disps[3]]),
        }

        current_x, all_xs, all_fixed_locs = optimize(current_x, increment_dict, tractions, ref_ctrl, mat_params)

        #return current_x, (None, None, None)
        return current_x, (np.stack(all_xs, axis=0), np.stack(all_fixed_locs, axis=0), None)

    def tanh_clip(x):
        return np.tanh(x) * 4.0
    def nn_fn(input):
        mlp = hk.Sequential([
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(10), jax.nn.softplus,
            hk.Linear(4),  tanh_clip,
        ])

        return mlp(input)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.array([0.0])
    #init_nn_params = nn_fn_t.init(rng, dummy_displacements)
    init_control_params = np.zeros((num_examples, 4))

    @jax.jit
    def colors_in_im(pts, mnist_im):
        # point will be in the square [5.0, 20.0]^2
        pts = 28.0 * (pts - 5.0) / 15.0

        # The image must be flipped in the first axis
        mnist_im = mnist_im[::-1, :].T

        return jax.scipy.ndimage.map_coordinates(mnist_im, pts.T, order=1).T[..., np.newaxis]

    @jax.jit
    @jax.vmap
    def pt_mask(pt):
        return np.logical_and(np.all(pt > 5.0, axis=-1, keepdims=True),
                              np.all(pt < 20.0, axis=-1, keepdims=True))

    @jax.jit
    @jax.vmap
    def points_in_pores(local_ctrl):
        left = local_ctrl[0, 0, :]
        top = local_ctrl[1, 0, :]
        right = local_ctrl[2, 0, ::-1]
        bottom = local_ctrl[3, 0, ::-1]
        left_right = np.linspace(left, right, 5)

        @jax.vmap
        def transform(top, bottom, left_right):
            vec1 = top - bottom
            vec2 = left_right[-1] - left_right[0]

            vec1_3d = np.concatenate((vec1, np.array([0])))
            vec1_3d_norm = np.linalg.norm(vec1_3d)
            vec2_3d = np.concatenate((vec2, np.array([0])))
            vec2_3d_norm = np.linalg.norm(vec2_3d)

            theta = np.arcsin(np.cross(vec2_3d / vec2_3d_norm, vec1_3d / vec1_3d_norm)[2])
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            norm_ratio = np.linalg.norm(vec1) / np.linalg.norm(vec2)

            return norm_ratio * (left_right[:] - left_right[0]) @ R.T + bottom

        res = transform(top, bottom, left_right)
        color_eval_pts = bsplines.mesh(np.linspace(0.0+1e-8, 1.0-1e-8, 15),
    				   np.linspace(0.0+1e-8, 1.0-1e-8, 15))
        color_eval_pts = color_eval_pts.reshape((-1, 2))
        color_eval_fn = cell.element.get_map_fn(color_eval_pts)

        return color_eval_fn(res)

    def im_from_points(all_points, all_colors, temp=1.0):
        """From a list of points (N, 2), colors (N,), and point mask (N,) produce
        a 28x28 image rendered from the points."""

        pixel_locs = \
                np.stack(np.meshgrid(np.linspace(5, 20, 28), np.linspace(5, 20, 28)), axis=-1)
        pixel_locs = pixel_locs.reshape(-1, 1, 2)
        all_inv_dists = 1. / np.sqrt(np.sum(np.square(all_points - pixel_locs), axis=-1))
        all_dists_softmax = jax.nn.softmax(temp * all_inv_dists, axis=-1)

        pixel_colors = np.sum(all_colors.reshape(1, -1) * all_dists_softmax, axis=-1)
        return pixel_colors.reshape(28, 28)

    def loss_fn(all_params, index):
        control_params, radii, color_params = all_params
        picture = some_mnist[index]

        # Only optimize color to test.
        control_params = jax.lax.stop_gradient(control_params)
        #radii = jax.lax.stop_gradient(radii)

        #color_params = jax.nn.sigmoid(color_params)  # Want between 0 and 1.
        colors = color_eval_fn(color_params)

        #mat_inputs = nn_fn_t.apply(nn_params, np.array([0.0]))
        #mat_inputs = np.array([0.0, 0.0, 0.0, 0.0])
        #control_params = np.tanh(control_params)
        mat_inputs = control_params[index]
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        # Get points in pores. They should correspond to 0 pixel value.
        final_ctrl_reshaped = final_x_local.reshape((-1, 4) + final_x_local.shape[1:])

        pore_locs = points_in_pores(final_ctrl_reshaped).reshape(-1, 2)
        color_locs = color_eval_fn(final_x_local).reshape(-1, 2)
        all_locs = np.concatenate((pore_locs, color_locs), axis=0)

        pore_locs_colors = np.zeros(pore_locs.shape[0])
        color_locs_colors = colors.flatten()
        all_colors = np.concatenate((pore_locs_colors, color_locs_colors), axis=0)

        rendered_image = im_from_points(all_locs, all_colors, temp=100)

        #im_colors_at_locs = colors_in_im(color_locs, picture)
        #im_colors_at_pores = colors_in_im(pore_locs, picture)

        ## Only consider locations within picture frame
        #valid_colors = pt_mask(color_locs)
        #valid_pore_points = pt_mask(pore_locs)

        #return np.sum(np.square(im_colors_at_locs - colors) * valid_colors) + \
        #        np.sum(np.square(im_colors_at_pores) * valid_pore_points)

        return np.sum(np.square(rendered_image - picture))

    print(f'Starting NMA optimization on device {dev_id}')

    curr_all_params = (init_control_params, init_radii, init_color_controls)
    if args.reload:
        print('Loading parameters.')
        with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{args.load_iter}.pkl'), 'rb') as f:
            curr_all_params = pickle.load(f)
        print('\tDone.')
        iter_num = args.load_iter
        processed_iter_num = args.load_iter
    else:
        iter_num = 0
        processed_iter_num = 0

    mpi_size = comm.Get_size()
    lr = 0.1 * mpi_size

    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(curr_all_params)

    loss_val_and_grad = jax.value_and_grad(loss_fn)

    def pytree_reduce(comm, pytree, scale=1.0, token=None):
        raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
        #reduce_sum, token = mpi4jax.allreduce(raveled, op=MPI.SUM, comm=comm, token=token)
        reduce_sum = comm.allreduce(raveled.block_until_ready(), op=MPI.SUM)
        token = None

        return unravel(reduce_sum * scale), token

    def test_pytrees_equal(comm, pytree, token=None):
        if comm.rank == 0:
            print('Testing if parameters have deviated.')
            vtime = time.time()
        raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
        #all_params, token = mpi4jax.gather(raveled, root=0, comm=comm, token=token)
        all_params = comm.gather(raveled.block_until_ready(), root=0)
        token = None
        if comm.rank == 0:
            for i in range(mpi_size-1):
                assert np.allclose(all_params[i], all_params[i+1])
            print(f'\tVerified in {time.time() - vtime} s.')

        return token

    ewa_loss = None
    ewa_weight = 0.95

    def simulate_element(params, im):
        loss, grad_loss = loss_val_and_grad(params, im)
        return loss, grad_loss

    import scipy.optimize

    raveled, unravel = jax.flatten_util.ravel_pytree(curr_all_params)
    def raveled_loss_fn(p, im=0):
        return loss_fn(unravel(p), im)

    def raveled_grad_fn(p, im=0):
        return onp.array(jax.flatten_util.ravel_pytree(jax.grad(loss_fn)(unravel(p), im))[0])

    def callback_fn(p):
        val = raveled_loss_fn(p)
        print(f'iteration value {val}.')

    # curr_all_params = (init_control_params, init_radii, init_color_controls)
    control_params_lower_bounds = np.ones_like(init_control_params) * -4.0
    control_params_upper_bounds = np.ones_like(init_control_params) * 4.0

    radii_lower_bounds = np.ones_like(init_radii) * 0.1
    radii_upper_bounds = np.ones_like(init_radii) * 0.9

    color_lower_bounds = np.ones_like(init_color_controls) * 0.0
    color_upper_bounds = np.ones_like(init_color_controls) * 1.0

    lbs = jax.flatten_util.ravel_pytree((control_params_lower_bounds,
                                         radii_lower_bounds,
                                         color_lower_bounds))[0]

    ubs = jax.flatten_util.ravel_pytree((control_params_upper_bounds,
                                         radii_upper_bounds,
                                         color_upper_bounds))[0]
    bounds = zip(lbs, ubs)
    results = scipy.optimize.minimize(raveled_loss_fn, onp.array(raveled), method='L-BFGS-B',
                                      jac=raveled_grad_fn, bounds=bounds,
                                      callback=callback_fn, options={'maxiter': 100})
    curr_all_params = unravel(results.x)
    print(f'BFGS finished optimization. Result: {results.success}')

    # Verify that the parameters have not deviated between different MPI ranks.
    token = None
    token = test_pytrees_equal(comm, curr_all_params, token=token)

    # Generate video
    if comm.rank == 0:
        rprint(f'Generating image and video with optimization so far.', comm=comm)

        curr_control_params, curr_radii, curr_color_params = curr_all_params
        #mat_inputs = nn_fn_t.apply(curr_nn_params, np.array([0.0]))

        index = onp.random.randint(0, num_examples)
        #mat_inputs = np.tanh(curr_control_params[index])
        mat_inputs = curr_control_params[index]
        optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(mat_inputs, curr_radii)

        all_velocities = np.zeros_like(all_displacements)
        all_fixed_vels = np.zeros_like(all_fixed_locs)

        image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-bfgs.png')
        vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-bfgs.mp4')
        #create_static_image_nma(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path, test_pts)
        ctrl_seq, _ = cell.unflatten_dynamics_sequence(
            all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
        create_movie_mnist(config, cell.element, ctrl_seq, vid_path, curr_color_params)

        # Pickle parameters
        rprint('Saving parameters.', comm=comm)
        with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-bfgs.pkl'), 'wb') as f:
            pickle.dump(curr_all_params, f)
        rprint('\tDone.', comm=comm)
        #token = mpi4jax.barrier(comm=comm, token=token)
        comm.barrier()

if __name__ == '__main__':
    app.run(main)
