from comet_ml import Experiment
import time
import os
import argparse


from varmintv2.geometry.multistable2d_diff import construct_multistable2D
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie, create_static_image

import jax.experimental.host_callback as hcb

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolverHCB, SparseNewtonSolverHCBRestart

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

    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    #optimizer = SparseNewtonSolverHCB(cell, potential_energy_fn, max_iter=100,
    #                                  step_size=0.8, tol=1e-1)
    optimizer = SparseNewtonSolverHCBRestart(cell, potential_energy_fn, max_iter=1000,
                                      step_size=1.0, tol=1e-8, ls_backtrack=0.95)

    optimize = optimizer.get_optimize_fn()
    n_increments = 30

    disp_i = (2.0*numy) * (geo_params[2] - geo_params[1]) + 0.2
    increments = disp_i / n_increments * np.arange(n_increments+1)

    @jax.jit
    def simulate(geo_params):
        #hcb.id_print(np.array([12345]))
        #hcb.id_print(increments)
        #hcb.id_print(np.array([67890]))

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
            new_x = optimize(x_prev, (fixed_locs, tractions, ref_ctrl))
            strain_energy = strain_energy_fn(new_x, fixed_locs, tractions, ref_ctrl)

            return new_x, (new_x, fixed_locs, strain_energy)
        
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = jax.lax.scan(sim_increment, init_x, increments)
        return final_x, (all_xs, all_fixed_locs, all_strain_energies)

    return cell, simulate, get_ref_ctrl_fn


        
#def update(gps, nx, ny, disp_t, se_t, lr = 0.01):
#    return gps - lr * jax.grad(loss_fn)(gps, nx, ny, disp_t, se_t)


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
    ny = 2
    all_gps = np.array([])
    gps_i = np.array([12.0,5.0,6.5,11.0,1.0,0.20,0.21])
    disp_i = (2.0*ny) * (gps_i[2] - gps_i[1]) + 0.2
    all_gps = np.append(all_gps,gps_i)
    
    DISP = disp_i / 30 * np.arange(30 + 1)
    
    cell, sim_fn, get_ref_ctrl_fn = construct_simulation(gps_i, nx, ny, disp_i, patch_ncp)
    l2g, g2l = cell.get_global_local_maps()

    print('Simulating target geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, se_t) = sim_fn(gps_i)
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
    gps = np.array([12.0,5.0,6.0,11.0,1.0,0.20,0.21])
    #gps = np.array([12.004259, 4.949189, 6.071916, 11.000167, 1.004048, 0.150658, 0.142823])
    #gps = np.array([12.004366, 4.946748, 6.075079, 10.998824, 1.005234, 0.149361, 0.140450])  # solvable in forward but not backward
    #gps = np.array([12.00407298, 4.92747882, 6.09261675, 10.98618575, 1.01330177, 0.15192443, 0.14647966])  # not solvable in forward or backward
    disp_t = DISP[-1]
    all_gps = np.append(all_gps,gps)
    
    
    print('Simulating initial geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(gps)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)
    

    plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
    plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}-0.png"))
    plt.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-0.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-0.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1], get_ref_ctrl_fn(gps)), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(gps))
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

    print(f'Finished simulation {args.exp_name}.')

    def loss_fn(gps):
        _, (_, _, se_p) = sim_fn(gps)
        return np.linalg.norm(se_p - se_t)
        
    def loss_fn_f(gps):
        _, (_, _, se_p) = sim_fn_f(gps)
        return np.linalg.norm(se_p - se_t)

    print('Starting adjoint optimization')
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

    eps = 1e-8
    loss = 100
    count = 1
    while loss > 0.02:
        print('About to simulate GPS:')
        print('\t', curr_gps)

        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_gps)
        print(f'Iteration {count} Loss: {loss} Grad Norm: {np.linalg.norm(grad_loss)} Time: {time.time() - iter_time}')

        
        if count % 20 == 0 or loss <= 0.02:

            print(f'Generating image and video with optimization so far.')
            optimized_curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(curr_gps)
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{count}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{count}.mp4')
            create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], get_ref_ctrl_fn(curr_gps)), image_path)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, get_ref_ctrl_fn(curr_gps))
            create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

            plt.plot(DISP, all_strain_energies, 's--r', DISP, se_t, 'o-b')
            plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}-{count}.png"))
            plt.close()
            
            all_gps = np.append(all_gps,curr_gps)

            scriptFile_g = open(os.path.join(args.exp_dir, f'all_gps.txt'), "w")
            onp.savetxt(scriptFile_g, all_gps.reshape(-1,curr_gps.shape[0]),"%f")
            scriptFile_g.close()

        #pt = b1 * pt + (1 - b1) * grad_loss
        #st = b2 * st + (1 - b2) * grad_loss**2
        #pht = pt/(1-b1**count)
        #sht = st/(1-b2**count)
        #curr_gps = curr_gps - lr * pht/(np.sqrt(sht)+eps)
        curr_gps = curr_gps - lr * grad_loss

        #curr_gps = np.clip(curr_gps, np.array([1.0, 5.0, 1.0, 1.0, 0.1, 0.2, 0.21]), \
        #                             np.array([20.0, 5.0, 10.0, 20.0, 2.0, 0.2, 0.21]))

        all_gpst = np.append(all_gpst,curr_gps)
        scriptFile_gt = open(os.path.join(args.exp_dir, f'all_gps_t.txt'), "w")
        onp.savetxt(scriptFile_gt, all_gpst.reshape(-1,curr_gps.shape[0]),"%f")
        scriptFile_gt.close()


        curr_gps = curr_gps.at[5:].set(np.abs(curr_gps[5:]))
        #curr_gps = curr_gps.at[5:].set(np.clip(curr_gps[5:],0.05,1))
        count += 1


    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy_optimized.dat'), "w")
    onp.savetxt(scriptFile_se, all_strain_energies,"%f")
    scriptFile_se.close()

    final_loss = loss_fn(curr_gps)
    print(f'Final loss: {final_loss}')
