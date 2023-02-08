from absl import app
from absl import flags

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import time
import os
import argparse

import optax

from geometry.compress_geometry import construct_beam

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


eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/symgroups/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/symgroups')

config = config_dict.ConfigDict({
    'jax_seed': 42,

    'sym_group': 1,

    # Number of repeats across each axis.
    'sym_xreps': 10,
    'sym_yreps': 10,

    'num_verts': 5000,
    'graph_method': 'mesh',
    'embed_dims': 0,
    'layers': [30, 30, 30],

    # Neural field match opt params
    'n_match_iters': 10000,
    'match_lr': 0.001,

    'ncp': 2,
    'quaddeg': 3,
    'splinedeg': 1,

    'nx': 200,
    'ny': 200,
    'width': 25.0,
    'height': 25.0,
    'disp': 3.0,
    'volf': 0.5,
    'f1': 2.5,
    'f2': 10.0,
    'simp_exponent': 3,
    'simp_eta': 0.5,
    'simp_move_limit': 0.2,

    'optim_type': 'grad',
    'weight_lr': 0.1,
    'volume_penalty': 10000.0,
    'maximize': True,

    'solver_parameters': {},

    'max_iters': 10000,
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
        stack.append(hk.Linear(1))

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


def construct_simulation(config, geo_params, numx, numy, disp_t, patch_ncp):
    mat = LinearElastic2D(SteelMat)

    cell, ref_ctrl = \
        construct_beam(geo_params[0], geo_params[1], numx, numy,
                       quad_degree=config.quaddeg, material=mat)

    l2g, g2l = cell.get_global_local_maps()

    potential_energy_fn = cell.get_potential_energy_fn()

    # Add a point load at the top right corner.
    point_load_fn = \
        generate_point_load_fn(ref_ctrl, g2l, np.array([0.0, config.height]))

    # Magnitude of point load.
    point_force = np.array([0.0, -0.0])

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
            '2': np.array([0.0, -1.0]),
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


def main(argv):
    args, dev_id, local_rank = eutils.initialize_experiment(verbose=True)
    config = args.config

    # running target simulation
    patch_ncp = 2
    nx = config.nx
    ny = config.ny

    gps_i = np.array([config.width, config.height])
    disp_i = config.disp

    DISP = disp_i / 20 * np.arange(20 + 1)
    cell, sim_fn, _ref_ctrl, optimizer = \
            construct_simulation(config, gps_i, nx, ny, disp_i, patch_ncp)

    # Construct the symmetry basis here.
    S = Symmetria.plane_group(config.sym_group)
    basis = S.sg.basic_basis

    pxx = onp.linspace(0, config.sym_xreps, config.nx)
    pxy = onp.linspace(0, config.sym_yreps, config.ny)
    px_grid = onp.stack(onp.meshgrid(pxx, pxy), axis=-1)

    embedder = S.get_orbifold_map(
        num_verts = config.num_verts,
        graph_method = config.graph_method,
    )

    embedded_px = embedder(px_grid.reshape(-1,2))[0]
    embed_dims = embedded_px.shape[1]

    key, subkey = jax.random.split(config.jax_rng)
    netfunc, init_weights = get_network(
        embed_dims,
        config.layers,
        np.sin,
        subkey,
    )

    def weights_to_pixels(weights):
        outputs = netfunc(weights, embedded_px)
        #outputs = outputs - np.min(outputs)
        #outputs = outputs / np.max(outputs)
        outputs = jax.nn.sigmoid(outputs / 100.0)
        return outputs.reshape((config.ny, config.nx))

    def match_weights_to_pixels(weights, pixels):
        # Do an optimization loop to find best weights
        # to match pixel values.
        def loss(weights):
            return np.mean((weights_to_pixels(weights) - pixels) ** 2)

        loss_val_grad = jax.jit(jax.value_and_grad(loss))
        current_weights = weights

        for i in range(config.n_match_iters):
            val, grad = loss_val_grad(current_weights)
            if i == 0:
                init_loss = val
            current_weights = jax.tree_util.tree_map(
                    lambda x, y: x - config.match_lr * y, current_weights, grad)
        final_loss = val
        return current_weights, init_loss, final_loss

    @jax.vmap
    def domain_to_sym_basis(point):
        return point * np.array([config.sym_xreps, config.sym_yreps]) / \
                np.array([config.width, config.height])

    ########
    vol_frac = config.volf
    e_min = 1e-9
    e_0 = 1.0
    p = config.simp_exponent
    x_e = np.ones((ny,nx))
    E_ini = e_min + x_e ** p * (e_0 - e_min)
    nu_ini = SteelMat._nu * np.ones(_ref_ctrl.shape[0])

    target_mat_params = (E_ini[::-1].reshape(nx*ny), nu_ini)

    all_Es = np.array([])
    all_Es = np.append(all_Es, E_ini[::-1].reshape(nx*ny))

    all_Es_t = np.array([])
    all_Es_t = np.append(all_Es_t, E_ini[::-1].reshape(nx*ny))

    l2g, g2l = cell.get_global_local_maps()

    def constraint(ele_d):
        return filtering.mean_density(ele_d, config.f1, config.f2) - vol_frac

    def constraint_sq(ele_d):
        return (filtering.mean_density(ele_d, config.f1, config.f2) - vol_frac) ** 2

    def objective(ele_d):
        ele_d = filtering.physical_density(ele_d, config.f1, config.f2)
        E_ele = e_min + ele_d ** p * (e_0 - e_min)
        mat_params = (E_ele[::-1].reshape(nx*ny), nu_ini)
        _, _, se_p = sim_fn(mat_params)

        # TODO(doktay): Convert back to se_p[-1] when full SE is computed.
        return se_p

    def penalized_objective(ele_d):
        se_p = objective(ele_d)
        penalty = constraint_sq(ele_d)

        if config.maximize:
            return -se_p + penalty * config.volume_penalty
        else:
            return se_p + penalty * config.volume_penalty

    def simulate_model(ele_d, output_path):
        ele_d = filtering.physical_density(ele_d, config.f1, config.f2)
        E_ele = e_min + ele_d ** p * (e_0 - e_min)
        mat_params = (E_ele[::-1].reshape(nx*ny), nu_ini)
        final_x, fixed_locs, _ = sim_fn(mat_params)

        # TODO(doktay): plt.plot(DISP, se_t, 'o-b', DISP, se_o, 'o--r')
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

    weight_optimizer = optax.sgd(config.weight_lr)
    opt_state = weight_optimizer.init(init_weights)
    scale = 1.0

    rprint(f'*Starting Optimization....*')
    # Run the optimality criteria optimization algorithm from the 88 lines paper.
    val_grad_O = jax.value_and_grad(objective)
    val_grad_C = jax.value_and_grad(constraint)
    val_grad_C2 = jax.value_and_grad(penalized_objective)

    x = vol_frac * np.ones((ny,nx))
    weights = init_weights

    change = 1.0
    for loop in range(config.max_iters):
        if change < config.change_tol:
            break

        if config.optim_type == 'oc':
            weights, init_loss, final_loss = match_weights_to_pixels(weights, x)
            print(f'optimized from {init_loss} to {final_loss}')
        x, weights_to_pixels_vjp = jax.vjp(weights_to_pixels, weights)

        iter_time = time.time()
        c, dc = val_grad_O(x)
        v, dv = val_grad_C(x)
        v2, dv2 = val_grad_C2(x)

        # Numpy magic to average gradients across orbits. Hack only for P1 group.
        x_stride = config.nx // config.sym_xreps
        y_stride = config.ny // config.sym_yreps

        grads_per_orbit = np.mean(dc.reshape(y_stride, config.sym_yreps,
                                  x_stride, config.sym_xreps), axis=(1, 3))
        dc = np.tile(grads_per_orbit, (config.sym_yreps, config.sym_xreps))

        grads_per_orbit = np.mean(dv.reshape(y_stride, config.sym_yreps,
                                  x_stride, config.sym_xreps), axis=(1, 3))
        dv = np.tile(grads_per_orbit, (config.sym_yreps, config.sym_xreps))

        grads_per_orbit = np.mean(dv2.reshape(y_stride, config.sym_yreps,
                                  x_stride, config.sym_xreps), axis=(1, 3))
        dv2 = np.tile(grads_per_orbit, (config.sym_yreps, config.sym_xreps))

        loop += 1
        losses.append(c)
        frames.append(x.copy())

        if loop % config.vis_every == 0:
            if len(iter_times) > 0:
                avg_iter_time = sum(iter_times) / len(iter_times)
            else:
                avg_iter_time = 0
            iter_times = []
            rprint(f'Iter: {loop}, Obj: {c:.4f}, Volume: {v+vol_frac:.4}, Full obj: {v2:.4f}, Avg iter time: {avg_iter_time:.2f}s')

            plt.imshow(1 - x, extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
            plt.savefig(os.path.join(args.exp_dir, f"gray-{loop}.png"))
            plt.close()
            
            plt.plot(losses)
            plt.savefig(os.path.join(args.exp_dir, f"losses.png"))
            plt.close()

            #plt.imshow(1 - nn_fitted_x, extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
            #plt.savefig(os.path.join(args.exp_dir, f"gray-{loop}-nn-fitted.png"))
            #plt.close()

        if config.optim_type == 'grad':
            penalized_obj_grad = weights_to_pixels_vjp(dv2)[0]
            updates, opt_state = weight_optimizer.update(penalized_obj_grad, opt_state)
            weights = optax.apply_updates(weights, updates)
        elif config.optim_type == 'oc':
            # Bisection algorithm to find optimal lagrange multiplier.
            l1, l2, move = 0, 1e9, config.simp_move_limit
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = (l2 + l1) / 2.0
                eta = config.simp_eta  # Numerical damping coefficient.

                # Optimality criteria update.
                xnew = x * np.abs(dc / (lmid * dv)) ** eta

                xnew = np.minimum(xnew, x + move)  # Cap from above by move limit.
                xnew = np.minimum(np.ones_like(xnew), xnew)  # Cap from above by 1.

                xnew = np.maximum(x - move, xnew)  # Cap from below by move limit.
                xnew = np.maximum(np.zeros_like(xnew), xnew)  # Cap from below by 0.

                # If too much volume, increase lagrange multiplier.
                # Else, decrease.
                if constraint(xnew) > 0:
                    l1 = lmid
                else:
                    l2 = lmid

                if (l1+l2 == 0):
                    raise Exception('div0, breaking lagrange multiplier')
            change = np.max(np.abs(xnew - x))
            x = xnew
        iter_times.append(time.time() - iter_time)

    rprint(f'final loss: {losses[-1]}')

    scriptFile_g = open(os.path.join(args.exp_dir, f'all_Xs.txt'), "w")
    onp.savetxt(scriptFile_g, onp.asarray(frames).reshape(-1,ini_x[0].shape[0]),"%f")
    scriptFile_g.close()

    plt.imshow(1 - frames[-1],extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
    plt.savefig(os.path.join(args.exp_dir, f"final-gray.png"))
    plt.close()

    rprint('Simulating final model...')
    simulate_model(x, 'final')

    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_final.txt'), "w")
    # TODO(doktay): onp.savetxt(scriptFile_se, se_o,"%f")
    scriptFile_se.close()
    scriptFile_in = open(os.path.join(args.exp_dir, f'increments_final.txt'), "w")
    onp.savetxt(scriptFile_in, DISP,"%f")
    scriptFile_in.close()
    scriptFile_ls = open(os.path.join(args.exp_dir, f'all_loss.txt'), "w")
    onp.savetxt(scriptFile_ls, losses,"%f")
    scriptFile_ls.close()

    rprint("Done!")


if __name__ == '__main__':
    app.run(main)
