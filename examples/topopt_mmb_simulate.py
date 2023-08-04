from absl import app
from absl import flags

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import time
import os
import argparse

import optax

from geometry.topopt_mmb_geometry import construct_mmb_beam
from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils import create_movie, create_static_image

import varmint.utils.analysis_utils as autils
import varmint.utils.experiment_utils as eutils

from varmint.utils.mpi_utils import *

from varmint.solver.incremental_loader import SparseNewtonIncrementalSolver

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax
import varmint.utils.filtering as filtering

from ml_collections import config_dict
from ml_collections import config_flags

import matplotlib.pyplot as plt


eutils.prepare_experiment_args(
    None, exp_root='/n/fs/mm-iga/Varmint/experiments',
            source_root='n/fs/mm-iga/Varmint/')

config = config_dict.ConfigDict({
    'ncp': 2,
    'quaddeg': 3,
    'splinedeg': 1,

    'nx': 60,
    'ny': 20,
    'width': 75.0,
    'height': 25.0,
    'disp': 3.0,
    'volf': 0.5,
    'f1': 1.0,
    'f2': 4.0,

    'solver_parameters': {
        'tol': 1e-8,
    }
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

    cell, get_ref_ctrl_fn = \
        construct_mmb_beam(geo_params, numx, numy, patch_ncp, quad_degree=config.quaddeg,
                           spline_degree=config.splinedeg, material=mat)

    l2g, g2l = cell.get_global_local_maps()
    ref_ctrl = get_ref_ctrl_fn()

    potential_energy_fn = cell.get_potential_energy_fn()

    # Add a point load at the top right corner.
    point_load_fn = \
        generate_point_load_fn(ref_ctrl, g2l, np.array([config.width, config.height]))

    # Magnitude of point load.
    point_force = np.array([0.0, -10.0])

    # Use this objective function in the solver instead of the standard potential energy.
    def potential_energy_with_point_load(current_x, fixed_locs, tractions, ref_ctrl, mat_params):
        return potential_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, mat_params) \
                + point_load_fn(current_x, fixed_locs, ref_ctrl, point_force)

    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    optimizer = SparseNewtonIncrementalSolver(cell, potential_energy_with_point_load,
                                              **config.solver_parameters)
    optimize = optimizer.get_optimize_fn()

    def simulate(mat_params):
        init_x = l2g(ref_ctrl, ref_ctrl)

        increment_dict = {
            '1': np.array([0.0, 0.0]),
            #'2': np.array([0.0, -disp_t]),
            #'3': np.array([0.0, 0.0]),
        }

        tractions_dict = {
            'A': np.array([0.0, -0.0]),
        }

        current_x, all_xs, all_fixed_locs, solved_increment = \
                optimize(init_x, increment_dict, tractions_dict, ref_ctrl, mat_params)
        if solved_increment < 1.0:
            # TODO(doktay): Fix along with returning full strain energy curve.
            raise RuntimeError("Could not complete solve.")

        fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, increment_dict)
        tractions = cell.tractions_from_dict({})
        strain_energy = strain_energy_fn(current_x, fixed_locs, tractions, ref_ctrl, mat_params)
        return current_x, (np.stack(all_xs, axis=0),
                           np.stack(all_fixed_locs, axis=0),
                           strain_energy)

    return cell, simulate, get_ref_ctrl_fn


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
    cell, sim_fn, get_ref_ctrl_fn = \
            construct_simulation(config, gps_i, nx, ny, disp_i, patch_ncp)
    _ref_ctrl = get_ref_ctrl_fn()

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

    rprint('Simulating target model...')
    sim_time = time.time()
    rprint(f'Initial E is {e_0}')
    final_x, (all_displacements, all_fixed_locs, se_t) = sim_fn(target_mat_params)
    rprint(f'Finished target simulation in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    # TODO(doktay): plt.plot(DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"target_SE.png"))
    plt.close()

    rprint('Saving results of target simulation...')
    image_path = os.path.join(args.exp_dir, f'sim-target.png')
    vid_path = os.path.join(args.exp_dir, f'sim-target.mp4')
    fig = plt.figure()
    ax = fig.gca()
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn()), ax)
    ax.set_aspect('equal')
    fig.savefig(image_path)
    plt.close(fig)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn())
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_target.txt'), "w")
    # TODO(doktay): onp.savetxt(scriptFile_se, se_t,"%f")
    scriptFile_se.close()
    scriptFile_in = open(os.path.join(args.exp_dir, f'increments_target.txt'), "w")
    onp.savetxt(scriptFile_in, DISP,"%f")
    scriptFile_in.close()

    plt.imshow(1 - x_e, cmap='gray',extent=(0, gps_i[0], 0, gps_i[1]), vmin=0, vmax=1)
    plt.savefig(os.path.join(args.exp_dir, f"target-gray.png"))
    plt.close()
    ############################################
    clip_min = 0.001 * np.ones((ny,nx))
    #clip_min = clip_min.at[-1,-1].set(1.0)
    clip_max = np.ones((ny,nx))
    ini_x = vol_frac * np.ones((ny,nx))

    losses = []
    frames = []

    def constraint(ele_d):
        return filtering.mean_density(ele_d, config, use_filter=True) - vol_frac

    def objective(ele_d):
        ele_d = filtering.physical_density(ele_d, config, use_filter=True)
        E_ele = e_min + ele_d ** p * (e_0 - e_min)
        mat_params = (E_ele[::-1].reshape(nx*ny), nu_ini)
        _, (_, _, se_p) = sim_fn(mat_params)

        # TODO(doktay): Convert back to se_p[-1] when full SE is computed.
        return np.linalg.norm(se_p)

    def test_optimization(x, objective_fn, losses, frames):
        loop, loss = 0, 1
        lr = 1e-2

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(x)
        val_grad_obj = jax.value_and_grad(objective_fn)
        while loss > 1e-1 and loop < 500:
            loss, grad_loss = val_grad_obj(x)
            iter_time = time.time()
            if loop % 10 == 0:
               rprint(f'Iter: {loop}, Time: {time.time()-iter_time:.2f}, Obj: {loss:.2f}, Grad: {np.sum(grad_loss):.2f}')

            losses.append(loss)
            frames.append(x.copy())
            loop += 1
            updates, opt_state = optimizer.update(grad_loss, opt_state)
            x = optax.apply_updates(x, updates)
            x = np.clip(x, clip_min, clip_max)
        return x, losses, frames

    def lm_optimization(x, objective_fn, constraint_fn, losses, frames):
        loop, change = 0, 1
        l, lr = 1e4, 1e-2
        lagrangian = lambda x, l : objective_fn(x) + l*np.abs(constraint_fn(x))

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(x)
        compute_loss = lambda x : lagrangian(x, l)
        val_grad_lagr = jax.value_and_grad(compute_loss)
        while change > 1e-5 and loop < 400:
            iter_time = time.time()
            loss, grad_loss = val_grad_lagr(x)
            c = objective_fn(x)
            v = constraint_fn(x)
            if loop % 10 == 0:
               rprint(f'Iter: {loop}, Loss: {loss:.2f}, Obj: {c:.2f}, Constr: {v:.2E}, Largest elem chg: {change:.4f}, Grad: {np.sum(grad_loss):.2f}')

            losses.append(loss)
            frames.append(x.copy())
            loop += 1
            updates, opt_state = optimizer.update(grad_loss, opt_state)
            change = np.max(np.abs(np.clip(optax.apply_updates(x, updates), clip_min, clip_max) - x))
            x = optax.apply_updates(x, updates)
            x = np.clip(x, clip_min, clip_max)

        return x, losses, frames
 
    def lmhb_optimization(x, objective_fn, constraint_fn, losses, frames):
        loop, change = 0, 1
        val_grad_O = jax.value_and_grad(objective_fn)
        val_grad_C = jax.value_and_grad(constraint_fn)
        while change > 0.0005 and loop < 300:
            c, dc = val_grad_O(x)
            v, dv = val_grad_C(x)
            iter_time = time.time()

            if loop % 10 == 0:
                rprint(f'Iter: {loop}, Obj: {c:.2f}, Constr: {v:.2E}, Largest elem chg: {change:.4f}')
                plt.imshow(1 - x, extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
                plt.savefig(os.path.join(args.exp_dir, f"gray-{loop}.png"))
                plt.close()

            loop += 1
            losses.append(c)
            frames.append(x.copy())

            l1, l2, move = 0, 1e9, 0.2
            while (l2 - l1) / (l1 + l2) > 1e-3:
                lmid = (l2 + l1) / 2.0
                la = np.multiply(x, np.sqrt(np.multiply(np.abs(dc), np.abs(1/dv))/lmid))
                #la = np.multiply(x, np.sqrt(np.multiply(-dc, 1/dv)/lmid))
                lb = np.min(np.stack((la, x+move)), axis=0)
                lc = np.min(np.stack((np.ones_like(lb), lb)), axis=0)
                ld = np.max(np.stack((x-move, lc)), axis=0)
                xnew = np.max(np.stack((np.zeros_like(ld), ld)), axis=0)
                if constraint_fn(xnew) > 0:
                    l1 = lmid
                else:
                    l2 = lmid
                if (l1+l2 == 0):
                    raise Exception('div0, breaking lagrange multiplier')
            change = np.max(np.abs(xnew - x))
            x = xnew

        return x, losses, frames

 
 
    rprint(f'*Starting Optimization....*')
    x, losses, frames = lmhb_optimization(ini_x, objective, constraint, losses, frames)
    rprint(f'final loss: {losses[-1]}')

    scriptFile_g = open(os.path.join(args.exp_dir, f'all_Xs.txt'), "w")
    onp.savetxt(scriptFile_g, onp.asarray(frames).reshape(-1,ini_x[0].shape[0]),"%f")
    scriptFile_g.close()

    plt.imshow(1 - frames[-1],extent=(0, gps_i[0], 0, gps_i[1]), cmap='gray',vmin=0, vmax=1)
    plt.savefig(os.path.join(args.exp_dir, f"final-gray.png"))
    plt.close()

    rprint('Simulating final model...')
    x = filtering.physical_density(x, config, use_filter=True)
    E_ele = e_min + x ** p * (e_0 - e_min)
    mat_params = (E_ele[::-1].reshape(nx*ny), nu_ini)

    plt.imshow(1 - E_ele, cmap='gray',vmin=0, vmax=e_0)
    plt.savefig(os.path.join(args.exp_dir, f"E_final-gray.png"))
    plt.close()

    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, se_o) = sim_fn(mat_params)
    rprint(f'Finished final sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    # TODO(doktay): plt.plot(DISP, se_t, 'o-b', DISP, se_o, 'o--r')
    plt.savefig(os.path.join(args.exp_dir, f"compare_SE.png"))
    plt.close()

    plt.plot(np.linspace(0,len(losses),len(losses)), losses, '-b')
    plt.savefig(os.path.join(args.exp_dir, f"loss.png"))
    plt.close()

    rprint('Saving results of final sim...')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-final.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-final.mp4')
    fig = plt.figure()
    ax = fig.gca()
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn()), ax)
    ax.set_aspect('equal')
    fig.savefig(image_path)
    plt.close(fig)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn())
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

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
