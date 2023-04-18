from absl import app
from absl import flags

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import time
import os
import argparse

import optax

from geometry.cantilever_geometry import construct_beam

from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

import varmint.utils.analysis_utils as autils
import varmint.utils.experiment_utils as eutils

from varmint.utils.mpi_utils import *

from varmint.solver.cholesky_solver import SparseCholeskyLinearSolver

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax
import varmint.utils.filtering as filtering

import haiku as hk

from ml_collections import config_dict
from ml_collections import config_flags

import matplotlib.pyplot as plt

from symmetria import Symmetria
from siren import get_siren_network


eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/symgroups/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/symgroups')

config = config_dict.ConfigDict({
    'jax_seed': 42,

    'sym_group': 1,
    'enforce_sym': False,

    # Number of repeats across each axis.
    'sym_xreps': 10,
    'sym_yreps': 10,

    'num_verts': 5000,
    'graph_method': 'mesh',
    'embed_dims': 0,
    'layers': [30, 30, 30],

    'siren': True,
    'siren_n_layers': 3,
    'siren_n_activations': 64,
    'siren_first_omega_0': 10.,
    'siren_hidden_omega_0': 10.,

    # Neural field match opt params
    'n_match_iters': 10000,
    'match_lr': 0.001,

    'ncp': 2,
    'quaddeg': 3,
    'splinedeg': 1,

    'nx': 200,
    'ny': 200,
    'width': 75.0,
    'height': 25.0,
    'disp': 3.0,
    'volf': 0.5,
    'f1': 2.5,
    'f2': 10.0,

    'init_simp_exponent': 1,
    'max_simp_exponent': 4,
    'simp_exponent_update': 0.01,

    'simp_eta': 0.5,
    'simp_move_limit': 0.2,

    'optim_type': 'grad',
    'weight_lr': 0.0001,
    'grad_clip': 0.1,
    'volume_penalty': 0.0,
    'maximize': False,

    'solver_parameters': {},

    'max_iters': 2000,
    'change_tol': 0.0005,
    'vis_every': 1,
})

config_flags.DEFINE_config_dict('config', config)


class SteelMat(Material):
    _E = 200.0
    _nu = 0.30
    _density = 8.0


def get_network(input_dims, layers, activation, key):
    def network(x):
        stack = []
        for layer in layers:
            stack.append(hk.Linear(layer))
            stack.append(activation)
        stack.append(hk.Linear(2, with_bias=False))

        mlp = hk.Sequential(stack)
        return mlp(x)

    base_fn_t = hk.transform(network)
    base_fn_t = hk.without_apply_rng(base_fn_t)

    weights = base_fn_t.init(key, np.ones(input_dims))

    return jax.vmap(base_fn_t.apply, in_axes=(None, 0)), weights


def generate_point_load_fn(ref_ctrl, g2l, point):
    # point is an array of shape (2,)
    index = np.sum((ref_ctrl - point) ** 2, axis=-1) < 1e-8
    num_indices = np.sum(index)

    def point_load_fn(current_x, fixed_locs, ref_ctrl, force):
        def_ctrl = g2l(current_x, fixed_locs, ref_ctrl)

        # index can contain multiple entries. We want to divide by number of
        # occurrences to get the force correctly.
        return -np.sum((def_ctrl[index] - ref_ctrl[index]) * force) / num_indices

    return point_load_fn


def construct_simulation(config):
    mat = LinearElastic2D(SteelMat)

    cell, ref_ctrl = \
        construct_beam(config.width, config.height, config.nx, config.ny,
                       quad_degree=config.quaddeg, material=mat)

    l2g, g2l = cell.get_global_local_maps()

    potential_energy_fn = cell.get_potential_energy_fn()

    # Add a point load at the top right corner.
    point_load_fn = \
        generate_point_load_fn(ref_ctrl, g2l, np.array([config.width, 0.0]))

    # Magnitude of point load.
    point_force = np.array([0.0, -100.0])
    #point_force = np.array([0.0, 0.0])

    # Use this objective function in the solver instead of the standard potential energy.
    def potential_energy_with_point_load(current_x, fixed_locs, tractions, ref_ctrl, mat_params):
        return potential_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, mat_params) \
                + point_load_fn(current_x, fixed_locs, ref_ctrl, point_force)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    optimizer = SparseCholeskyLinearSolver(cell, potential_energy_with_point_load,
                                           **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    def simulate(mat_params):
        init_x = l2g(ref_ctrl, ref_ctrl)

        increment_dict = {
            '1': np.array([0.0, 0.0]),
            #'2': np.array([0.0, -1.0]),
        }

        tractions_dict = {
            #'A': np.array([0.0, -0.0]),
        }

        current_x = optimize(init_x, increment_dict, tractions_dict, ref_ctrl, mat_params)

        fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, increment_dict)
        tractions = cell.tractions_from_dict({})
        strain_energy = strain_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, mat_params)
        return current_x, fixed_locs, strain_energy

    return cell, simulate, ref_ctrl, optimizer


def construct_symmetry_basis(key, config):
    pxx = onp.linspace(0, config.sym_xreps, config.nx)
    pxy = onp.linspace(0, config.sym_yreps, config.ny)
    px_grid = onp.stack(onp.meshgrid(pxx, pxy), axis=-1)

    if config.enforce_sym:
        # Construct the symmetry basis here.
        S = Symmetria.plane_group(config.sym_group)
        basis = S.sg.basic_basis

        embedder = S.get_orbifold_map(
            num_verts = config.num_verts,
            graph_method = config.graph_method,
        )

        embedded_px = embedder(px_grid.reshape(-1,2))[0]
        embed_dims = embedded_px.shape[1]
    else:
        embed_dims = 2

        # Translate pixels to -1, 1 for SIREN network.
        embedded_px = 2 * px_grid - np.array([1.0, 1.0])

    key, subkey = jax.random.split(key)
    if config.siren:
        netfunc, init_weights = get_siren_network(
            embed_dims, 
            config.siren_n_layers,
            config.siren_n_activations,
            subkey,
            first_omega_0=config.siren_first_omega_0,
            hidden_omega_0=config.siren_hidden_omega_0,
        )
    else:
        netfunc, init_weights = get_network(
            embed_dims,
            config.layers,
            np.sin,
            subkey,
        )

    return key, embedded_px, netfunc, init_weights


def main(argv):
    args, dev_id, local_rank = eutils.initialize_experiment(verbose=True)
    config = args.config

    # running target simulation
    cell, sim_fn, _ref_ctrl, optimizer = construct_simulation(config)

    # construct symmetry basis
    key, embedded_px, netfunc, init_weights = construct_symmetry_basis(config.jax_rng, config)

    def weights_to_pixels(weights, offset):
        outputs = netfunc(weights, embedded_px) + np.array([offset, 0.0])
        outputs = jax.nn.softmax(outputs)[..., 0]  # Softmax but take only first element.
        return outputs.reshape((config.ny, config.nx))

    @jax.vmap
    def domain_to_sym_basis(point):
        return point * np.array([config.sym_xreps, config.sym_yreps]) / \
                np.array([config.width, config.height])

    ########
    def x_to_mp(x, p):
        E = 1e-9 + x ** p * (SteelMat._E - 1e-9)
        nu_ini = SteelMat._nu * np.ones(_ref_ctrl.shape[0])
        return (E[::-1].reshape(config.nx * config.ny), nu_ini)
    l2g, g2l = cell.get_global_local_maps()

    def constraint(ele_d):
        return np.mean(ele_d) - config.volf

    def constraint_sq(ele_d):
        return constraint(ele_d) ** 2

    # For scaling objective function, get initial strain energy.
    uniform_gray = config.volf * np.ones((config.ny, config.nx))
    mat_params = x_to_mp(uniform_gray, 1)
    _, _, se_p = sim_fn(mat_params)
    J_0 = se_p  # Get this for scaling

    def objective(ele_d, p):
        mat_params = x_to_mp(ele_d, p)
        _, _, se_p = sim_fn(mat_params)

        return se_p / J_0

    def simulate_model(ele_d, output_path):
        mat_params = x_to_mp(ele_d, p)
        final_x, fixed_locs, _ = sim_fn(mat_params)

        plt.savefig(os.path.join(args.exp_dir, f"compare_SE-{output_path}.png"))
        plt.close()

        plt.plot(np.linspace(0,len(losses),len(losses)), losses, '-b')
        plt.savefig(os.path.join(args.exp_dir, f"loss-{output_path}.png"))
        plt.close()

        rprint('Saving results of sim...')
        image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-{output_path}.png')
        vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-{output_path}.mp4')
        fig = plt.figure()
        ax = fig.gca()
        create_static_image(cell.element, g2l(final_x, fixed_locs, _ref_ctrl), ax)
        ax.set_aspect('equal')
        fig.savefig(image_path)
        plt.close(fig)

    losses = []
    frames = []
    iter_times = []

    weight_optimizer = optax.adam(config.weight_lr)
    opt_state = weight_optimizer.init(init_weights)

    rprint(f'*Starting Optimization....*')
    # Run the optimality criteria optimization algorithm from the 88 lines paper.
    val_grad_O = jax.value_and_grad(objective)
    val_grad_C2 = jax.value_and_grad(constraint_sq)

    weights = init_weights
    offset = 0.0
    p = config.init_simp_exponent
    alpha = 0.1

    change = 1.0
    for loop in range(config.max_iters):
        if change < config.change_tol:
            break

        x, weights_to_pixels_vjp = jax.vjp(weights_to_pixels, weights, offset)

        iter_time = time.time()
        c, dc = val_grad_O(x, p)
        v, dv = val_grad_C2(x)

        penalized_sensitivity = \
                jax.tree_util.tree_map(lambda x, y: x + alpha * y, dc, dv)

        loop += 1
        losses.append(c)
        frames.append(x.copy())

        if loop % config.vis_every == 0:
            if len(iter_times) > 0:
                avg_iter_time = sum(iter_times) / len(iter_times)
            else:
                avg_iter_time = 0
            iter_times = []
            rprint(f'Iter: {loop}, Obj: {c:.4f}, Volume Penalty: {v:.4}, SIMP p: {p:.2f}, alpha: {alpha:.2f}, Avg iter time: {avg_iter_time:.2f}s')

            plt.imshow(1 - x, extent=(0, config.width, 0, config.height), cmap='gray',vmin=0, vmax=1)
            plt.savefig(os.path.join(args.exp_dir, f"gray-{loop}.png"))
            plt.close()

            plt.plot(losses)
            plt.savefig(os.path.join(args.exp_dir, f"losses.png"))
            plt.close()

        p = min(p + config.simp_exponent_update, config.max_simp_exponent)
        alpha = min(alpha + 0.2, 100)

        penalized_obj_grad = weights_to_pixels_vjp(penalized_sensitivity)[0]
        penalized_obj_grad = jax.tree_util.tree_map(lambda t: np.clip(t, -config.grad_clip, config.grad_clip), penalized_obj_grad)
        updates, opt_state = weight_optimizer.update(penalized_obj_grad, opt_state)
        weights = optax.apply_updates(weights, updates)

        # Fold the offset into the last layer
        #assert weights['linear_3']['w'].shape[1] == 2
        #weights['linear_3']['w'] /= -offset
        #offset = -1.0

        iter_times.append(time.time() - iter_time)

    rprint(f'final loss: {losses[-1]}')

    scriptFile_g = open(os.path.join(args.exp_dir, f'all_Xs.txt'), "w")
    onp.savetxt(scriptFile_g, onp.asarray(frames).reshape(-1,ini_x[0].shape[0]),"%f")
    scriptFile_g.close()

    plt.imshow(1 - frames[-1],extent=(0, config.width, 0, config.height), cmap='gray',vmin=0, vmax=1)
    plt.savefig(os.path.join(args.exp_dir, f"final-gray.png"))
    plt.close()


if __name__ == '__main__':
    app.run(main)
