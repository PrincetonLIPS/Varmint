from comet_ml import Experiment
import time
import os
import argparse

import optax


from varmintv2.geometry.multistable2d_diff_var import construct_multistable2D
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie, create_static_image

import jax.experimental.host_callback as hcb

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

from varmintv2.solver.optimization_speed import SparseNewtonIncrementalSolver
from varmintv2.solver.optimization import SparseNewtonSolverHCBRestart

import scipy.optimize
from scipy.signal import argrelextrema

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
    parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


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

class TPOMat(Material):
    _E = 0.02
    _nu = 0.47
    _density = 1.14

class SteelMat(Material):
    _E = 200.0
    _nu = 0.3
    _density = 8.0


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

    ref_ctrl = get_ref_ctrl_fn(geo_params)

    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    #optimizer = SparseNewtonSolverHCB(cell, potential_energy_fn, max_iter=100,step_size=0.8, tol=1e-1)
    optimizer = SparseNewtonSolverHCBRestart(cell, potential_energy_fn, max_iter=1000,
                                             step_size=1.0, tol=1e-8, ls_backtrack=0.95)

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

    if args.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="gTBUDHLLNaIqxMWyjHKQgtlkW",
            project_name="general",
            workspace="denizokt",
        )
    else:
        experiment = None
    
    
    # running target simulation
    patch_ncp = 7
    nx = 1
    ny = 1
    all_gps = np.array([])
    gps_i = np.array([12.0,1.0,5.0,7.0,11.0,0.2])
    disp_i = 2.0 * (gps_i[2+ny] - gps_i[2])

    all_gps = np.append(all_gps,gps_i)
    
    DISP = disp_i / 40 * np.arange(40 + 1)

    cell, sim_fn, get_ref_ctrl_fn = construct_simulation(gps_i, nx, ny, disp_i, patch_ncp)

    _ref_ctrl = get_ref_ctrl_fn(gps_i)
    target_E = 0.06
    target_mat_params = (
        target_E * np.ones(_ref_ctrl.shape[0]),
        TPUMat._nu * np.ones(_ref_ctrl.shape[0]),
    )

    l2g, g2l = cell.get_global_local_maps()

    print('Simulating target geometry parameters.')
    sim_time = time.time()
    print(f'Target E is {target_E}')
    final_x, (all_displacements, all_fixed_locs, se_t) = sim_fn(gps_i, target_mat_params)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    plt.plot(DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"target_SEs-{args.exp_name}.png"))
    plt.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-target.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-target.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn(gps_i)), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(gps_i))
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)
    
    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_target.txt'), "w")
    onp.savetxt(scriptFile_se, se_t,"%f")
    scriptFile_se.close()
    scriptFile_in = open(os.path.join(args.exp_dir, f'increments_target.txt'), "w")
    onp.savetxt(scriptFile_in, DISP,"%f")
    scriptFile_in.close()

    print(f'Finished target simulation {args.exp_name}.')

    
    ############################################
    print(f"Starting experiment {args.exp_name}.")

    loca = DISP[argrelextrema(se_t, np.less)]
    h2_st = np.sort(np.diff(loca,prepend=0)/2.0)[::-1]
    thick = np.array([0.19,0.20,0.21,0.22])
    
    gps = np.array([12.0,1.00])
    gps = np.append(gps, 5.0*np.ones(ny))
    gps = np.append(gps, 5.0+h2_st)
    gps = np.append(gps, 11.0*np.ones(ny))
    gps = np.append(gps, thick[0:ny])
    

    clip_min = np.array([1.0, 0.4])
    clip_min = np.append(clip_min, 1.0*np.ones(3*ny))
    clip_min = np.append(clip_min, 0.1*np.ones(ny))

    clip_max = np.array([20.0, 3.0])
    clip_max = np.append(clip_max, 15.0*np.ones(3*ny))
    #clip_max = clip_max.at[-ny:].set(10.0)
    clip_max = np.append(clip_max, 0.4*np.ones(ny))

    all_gps = np.append(all_gps,gps)
    

    initial_E = 0.02
    initial_mat_params = (
        initial_E * np.ones(_ref_ctrl.shape[0]),
        TPUMat._nu * np.ones(_ref_ctrl.shape[0]),
    )

    print('Simulating initial geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(gps, initial_mat_params)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)
    

    plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}-0.png"))
    plt.close()


    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_optimized-{0}.txt'), "w")
    onp.savetxt(scriptFile_se, all_strain_energies,"%f")
    scriptFile_se.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-0.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-0.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn(gps)), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(gps))
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

    print(f'Finished initial simulation {args.exp_name}.')
    ############################################
 
    def loss_fn(all_params):
        gps, E = all_params
        mat_params = (
            E * np.ones(_ref_ctrl.shape[0]),
            TPUMat._nu * np.ones(_ref_ctrl.shape[0]),
        )

        _, (_, _, se_p) = sim_fn(gps, mat_params)
        return np.linalg.norm(se_p - se_t)
        
    print('Starting adjoint optimization')
    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    curr_params = (gps_i, initial_E)
    all_gpst = np.array([])
    all_gpst = np.append(all_gpst, curr_params[0])
    lr = 0.01
    pt = 0.0
    st = 0.0
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(curr_params)
    print(f'Initial E is {initial_E}')

    loss = 100
    count = 1
    all_loss = np.array([])
    all_count = np.array([])
    while loss > 0.02 and count < 2000:

        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_params)
        print(f'Iteration {count} Loss: {loss} GPS Grad Norm: {np.linalg.norm(grad_loss[0])} E Grad Norm: {np.linalg.norm(grad_loss[1])} Time: {time.time() - iter_time}')

        all_loss = np.append(all_loss,loss)
        all_count = np.append(all_count,count)
        if count % 10 == 0 or loss <= 0.02:
            mat_params = (
                curr_params[1] * np.ones(_ref_ctrl.shape[0]),
                TPUMat._nu * np.ones(_ref_ctrl.shape[0]),
            )

            print(f'Generating image and video with optimization so far.')
            optimized_curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(curr_params[0], mat_params)
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{count}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{count}.mp4')
            create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], get_ref_ctrl_fn(curr_params[0])), image_path)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(curr_params[0]))
            create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

            plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
            plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}-{count}.png"))
            plt.close()
            
            all_gps = np.append(all_gps,curr_params[0])

            scriptFile_g = open(os.path.join(args.exp_dir, f'all_gps.txt'), "w")
            onp.savetxt(scriptFile_g, all_gps.reshape(-1,curr_params[0].shape[0]),"%f")
            scriptFile_g.close()

            scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_optimized-{count}.txt'), "w")
            onp.savetxt(scriptFile_se, all_strain_energies,"%f")
            scriptFile_se.close()


        scriptFile_ls = open(os.path.join(args.exp_dir, f'all_loss.txt'), "w")
        onp.savetxt(scriptFile_ls, all_loss,"%f")
        scriptFile_ls.close()


        # pt = b1 * pt + (1 - b1) * grad_loss
        # st = b2 * st + (1 - b2) * grad_loss**2
        # pht = pt/(1-b1**count)
        # sht = st/(1-b2**count)
        # curr_gps = curr_gps - lr * pht/(np.sqrt(sht)+eps)
        #curr_gps = curr_gps - lr * grad_loss
        updates, opt_state = optimizer.update(grad_loss, opt_state)
        curr_params = optax.apply_updates(curr_params, updates)

        all_gpst = np.append(all_gpst,curr_params[0])
        scriptFile_gt = open(os.path.join(args.exp_dir, f'all_gps_t.txt'), "w")
        onp.savetxt(scriptFile_gt, all_gpst.reshape(-1,curr_params[0].shape[0]),"%f")
        scriptFile_gt.close()

        curr_gps, curr_E = curr_params
        curr_gps = np.clip(curr_gps, clip_min, clip_max)
        curr_E = np.clip(curr_E, 0.02, 0.07)
        print(f'New E is {curr_E}')
        curr_params = (curr_gps, curr_E)
        count += 1


    if loss <= 0.02:
       scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_optimized.txt'), "w")
       onp.savetxt(scriptFile_se, all_strain_energies,"%f")
       scriptFile_se.close()
    else:
        mat_params = (
            curr_params[1] * np.ones(_ref_ctrl.shape[0]),
            TPUMat._nu * np.ones(_ref_ctrl.shape[0]),
        )

        print(f'Generating image and video with BEST optimization...')
        b_gps = all_gpst.reshape(-1,curr_params[0].shape[0])[np.argmin(all_loss)]
        optimized_curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(b_gps, mat_params)
        image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-BEST-optimized-{np.argmin(loss)}.png')
        vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-BEST-optimized-{np.argmin(loss)}.mp4')
        create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], get_ref_ctrl_fn(b_gps)), image_path)
        ctrl_seq, _ = cell.unflatten_dynamics_sequence(
            all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(b_gps))
        create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)


        scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_optimized.txt'), "w")
        onp.savetxt(scriptFile_se, all_strain_energies,"%f")
        scriptFile_se.close()
        

        plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
        plt.savefig(os.path.join(args.exp_dir, f"compare_BEST-SEs-{args.exp_name}-{np.argmin(loss)}.png"))
        plt.close()
        
        b_loss = loss_fn(b_gps)
        
        all_count = np.append(all_count,count)
        all_loss = np.append(all_loss,b_loss)
        scriptFile_ls = open(os.path.join(args.exp_dir, f'all_loss.txt'), "w")
        onp.savetxt(scriptFile_ls, all_loss,"%f")
        scriptFile_ls.close()
        
        all_gpst = np.append(all_gpst,b_gps)
        scriptFile_gt = open(os.path.join(args.exp_dir, f'all_gps_t.txt'), "w")
        onp.savetxt(scriptFile_gt, all_gpst.reshape(-1,curr_params[0].shape[0]),"%f")
        scriptFile_gt.close()
        
        print (f'final loss = {b_loss}')


    plt.plot(all_count, all_loss, '-b')
    plt.savefig(os.path.join(args.exp_dir, f"loss.png"))
    plt.close()
    
