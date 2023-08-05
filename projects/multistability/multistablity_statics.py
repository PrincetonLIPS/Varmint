from comet_ml import Experiment
import time
import os
import argparse

from varmint.geometry.multistable2d import construct_multistable2D
from varmint.geometry.elements import Patch2D
from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmint.physics.materials import Material
from varmint.utils.movie_utils_v0 import create_movie, create_static_image

import jax.experimental.host_callback as hcb

import varmint.utils.analysis_utils as autils
import varmint.utils.experiment_utils_args as eutils

from varmint.solver.optimization import SparseNewtonSolverHCB, SparseNewtonSolverHCBRestart

import scipy.optimize

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

import matplotlib.pyplot as plt

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(
    parser, exp_root='/Users/mehran/Desktop/Varmint-main/projects/multistability/experiments')

# Geometry parameters.
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=8)
parser.add_argument('-s', '--splinedeg', type=int, default=3)

parser.add_argument('--simtime', type=float, default=50.0)
parser.add_argument('--dt', type=float, default=0.5)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--comet', dest='comet', action='store_true')

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--strategy', choices=['ilu_preconditioning', 'superlu', 'lu'],
                    default='ilu_preconditioning')


class TPUMat(Material):
    _E = 0.07
    _nu = 0.46
    _density = 1.25


def construct_simulation(geo_params, numx, numy, disp_t, patch_ncp):
    mat = NeoHookean2D(TPUMat)

    multiplier = 1.0
    cell, get_ref_ctrl_fn = \
        construct_multistable2D(geo_params, numx, numy, patch_ncp, quad_degree=args.quaddeg,
                               spline_degree=args.splinedeg, material=mat,
                               multiplier=multiplier)


    l2g, g2l = cell.get_global_local_maps()
    tractions = {}
    tractions = cell.tractions_from_dict(tractions)

    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    optimizer = SparseNewtonSolverHCBRestart(cell, potential_energy_fn, max_iter=1000,step_size=1.0, tol=1e-8, ls_backtrack=0.95)

    optimize = optimizer.get_optimize_fn()
    n_increments = 40

    
    increments = disp_t / n_increments * np.arange(n_increments+1)

    @jax.jit
    def simulate(geo_params, mat_params):

        ref_ctrl = get_ref_ctrl_fn(geo_params)
        init_x = l2g(ref_ctrl, ref_ctrl)

        def sim_increment(x_prev, increment):
            fixed_displacements = {
                '1': np.array([0.0, 0.0]),
                '2': np.array([0.0, -increment]),
                '3': np.array([0.0, 0.0]),
                '4': np.array([0.0, 0.0]),
            }
            fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
            new_x = optimize(x_prev, (fixed_locs, tractions, ref_ctrl, mat_params))
            strain_energy = strain_energy_fn(new_x, fixed_locs, tractions, ref_ctrl, mat_params)

            return new_x, (new_x, fixed_locs, strain_energy)
        
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = jax.lax.scan(sim_increment, init_x, increments)
        return final_x, (all_xs, all_fixed_locs, all_strain_energies)

    return cell, simulate, get_ref_ctrl_fn


if __name__ == '__main__':
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args)
    # args.seed and args.exp_dir should be set.
    eutils.save_args(args)
    npr.seed(args.seed)

    # running target simulation
    patch_ncp = 7
    nx = 1
    ny = 2
    all_gps = np.array([])
    gps_i = np.array([12.0,1.0,5.0,6.0,8.0,7.7,12.0,11.0,0.20,0.21])
    disp_i = 2.0 * (gps_i[2+ny] - gps_i[2] + gps_i[3+ny] - gps_i[3]) + 0.5


    all_gps = np.append(all_gps,gps_i)
    
    DISP = disp_i / 40 * np.arange(40 + 1)
    
    cell, sim_fn, get_ref_ctrl_fn = construct_simulation(gps_i, nx, ny, disp_i, patch_ncp)
    l2g, g2l = cell.get_global_local_maps()
    
    ref_ctrl = get_ref_ctrl_fn(gps_i)
    
    mat_params = (
        TPUMat.E * np.ones(ref_ctrl.shape[0]),
        TPUMat.nu * np.ones(ref_ctrl.shape[0]),
    )

    print('Simulating target geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, se_t) = sim_fn(gps_i, mat_params)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    plt.plot(DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"target_SEs.png"))
    plt.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-target.png')
    vid_path = os.path.join(args.exp_dir, f'sim-target.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn(gps_i)), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(gps_i))
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)
    ############################################
    print(f"Starting experiment {args.exp_name}.")
    gps_lower = np.array([1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
    gps_upper = np.array([10.0, 3.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 0.4,0.4])
    gps = np.array([12.0,1.0,5.0,5.0,6.8,6.5,11.0,11.0,0.21,0.20])
    disp_t = DISP[-1]
    all_gps = np.append(all_gps,gps)
    
    
    print('Simulating initial geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(gps,mat_params)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)
    

    plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"SEs-0.png"))
    plt.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-0.png')
    vid_path = os.path.join(args.exp_dir, f'sim-0.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn(gps)), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(gps))
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

    def loss_fn(gps):
        _, (_, _, se_p) = sim_fn(gps,mat_params)
        return np.linalg.norm(se_p - se_t)
        
    print('*** Starting Adjoint Optimization... ***')
    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    curr_gps = gps
    all_gpst = np.array([])
    all_gpst = np.append(all_gpst,curr_gps)
    lr = 0.01
    pt = 0.0
    st = 0.0
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    loss = 1
    count = 1
    all_loss = np.array([])
    all_count = np.array([])
    while loss > 0.01 and count < 3000:

        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_gps)
        print(f'Iteration {count} Loss: {loss} Grad Norm: {np.linalg.norm(grad_loss)} Time: {time.time() - iter_time}')

        all_loss = np.append(all_loss,loss)
        all_count = np.append(all_count,count)
        if count % 10 == 0 or loss <= 0.01:

            optimized_curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(curr_gps,mat_params)
            image_path = os.path.join(args.exp_dir, f'sim-optimized-{count}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-optimized-{count}.mp4')
            create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], get_ref_ctrl_fn(curr_gps)), image_path)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(curr_gps))
            create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

            plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
            plt.savefig(os.path.join(args.exp_dir, f"SEs-{count}.png"))
            plt.close()
            
            all_gps = np.append(all_gps,curr_gps)

            scriptFile_g = open(os.path.join(args.exp_dir, f'all_gps.txt'), "w")
            onp.savetxt(scriptFile_g, all_gps.reshape(-1,curr_gps.shape[0]),"%f")
            scriptFile_g.close()


        scriptFile_ls = open(os.path.join(args.exp_dir, f'all_loss.txt'), "w")
        onp.savetxt(scriptFile_ls, all_loss,"%f")
        scriptFile_ls.close()


        pt = b1 * pt + (1 - b1) * grad_loss
        st = b2 * st + (1 - b2) * grad_loss**2
        pht = pt/(1-b1**count)
        sht = st/(1-b2**count)
        curr_gps = curr_gps - lr * pht/(np.sqrt(sht)+eps)
        
        all_gpst = np.append(all_gpst,curr_gps)
        scriptFile_gt = open(os.path.join(args.exp_dir, f'all_gps_t.txt'), "w")
        onp.savetxt(scriptFile_gt, all_gpst.reshape(-1,curr_gps.shape[0]),"%f")
        scriptFile_gt.close()

        curr_gps = np.clip(curr_gps, gps_lower, gps_upper)
        count += 1

    plt.plot(all_count, all_loss, '-b')
    plt.savefig(os.path.join(args.exp_dir, f"loss.png"))
    plt.close()
