from absl import app
from absl import flags

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import time
import os
import argparse

import optax

from geometry.small_beam_geometry import construct_beam
from geometry.topopt_mmb_geometry import construct_mmb_beam

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

from ml_collections import config_dict
from ml_collections import config_flags

import matplotlib.pyplot as plt


eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/projects/symgroups/experiments',
            source_root='n/fs/mm-iga/Varmint/projects/symgroups')

config = config_dict.ConfigDict({
    'ncp': 2,
    'quaddeg': 3,
    'splinedeg': 1,

    'nx': 300,
    'ny': 100,
    'width': 75.0,
    'height': 25.0,
    'disp': 3.0,
    'volf': 0.5,
    'f1': 5.0,
    'f2': 40.0,

    'solver_parameters': {},

    'max_iters': 1000,
    'change_tol': 0.0005,
})

config_flags.DEFINE_config_dict('config', config)


class SteelMat(Material):
    _E = 200.0
    _nu = 0.30
    _density = 8.0


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
    point_force = np.array([0.0, -1.0])

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
            #'2': np.array([0.0, -disp_t]),
            '3': np.array([0.0, 0.0]),
        }

        tractions_dict = {
            'A': np.array([0.0, -0.0]),
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

    ########
    vol_frac = config.volf
    e_min = 1e-9
    e_0 = 1.0
    p = 3
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

    def objective(ele_d):
        ele_d = filtering.physical_density(ele_d, config.f1, config.f2)
        E_ele = e_min + ele_d ** p * (e_0 - e_min)
        mat_params = (E_ele[::-1].reshape(nx*ny), nu_ini)
        _, _, se_p = sim_fn(mat_params)

        # TODO(doktay): Convert back to se_p[-1] when full SE is computed.
        return np.linalg.norm(se_p)

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

    rprint(f'*Starting Optimization....*')
    # Run the optimality criteria optimization algorithm from the 88 lines paper.
    val_grad_O = jax.value_and_grad(objective)
    val_grad_C = jax.value_and_grad(constraint)

    x = vol_frac * np.ones((ny,nx))
    change = 1.0
    for loop in range(config.max_iters):
        if change < config.change_tol:
            break

        #if loop % 100 == 0:  
        #    simulate_model(x, f'iteration-{loop}')

        iter_time = time.time()
        c, dc = val_grad_O(x)
        v, dv = val_grad_C(x)

        if loop % 10 == 0:
            if len(iter_times) > 0:
                avg_iter_time = sum(iter_times) / len(iter_times)
            else:
                avg_iter_time = 0
            iter_times = []
            rprint(f'Iter: {loop}, Obj: {c:.2f}, Constr: {v:.2E}, Largest elem chg: {change:.4f}, Avg iter time: {avg_iter_time:.2f}s')
            plt.imshow(1 - x, extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
            plt.savefig(os.path.join(args.exp_dir, f"gray-{loop}.png"))
            plt.close()

        loop += 1
        losses.append(c)
        frames.append(x.copy())

        # Bisection algorithm to find optimal lagrange multiplier.
        l1, l2, move = 0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = (l2 + l1) / 2.0
            eta = 0.5  # Numerical damping coefficient.

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
