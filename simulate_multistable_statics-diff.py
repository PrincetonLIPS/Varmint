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

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolverHCB

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
parser.add_argument('-c', '--ncp', type=int, default=7)
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


def construct_simulation(geo_params, numx, numy, disp_t):
    mat = NeoHookean2D(TPUMat)

    multiplier = 1.0
    cell, get_ref_ctrl_fn = \
        construct_multistable2D(geo_params, numx, numy, patch_ncp=args.ncp, quad_degree=args.quaddeg,
                               spline_degree=args.splinedeg, material=mat,
                               multiplier=multiplier)


    l2g, g2l = cell.get_global_local_maps()
    tractions = {}
    tractions = cell.tractions_from_dict(tractions)

    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())
    optimizer = SparseNewtonSolverHCB(cell, potential_energy_fn, max_iter=100,
                                      step_size=0.8, tol=1e-1)
    optimize = optimizer.get_optimize_fn()

    n_increments = 20
    increments = disp_t / n_increments * np.arange(n_increments+1)

    @jax.jit
    def simulate(geo_params):
        ref_ctrl = get_ref_ctrl_fn(geo_params)
        init_x = l2g(ref_ctrl)

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

    return cell, simulate


        
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

    print(f"Starting experiment {args.exp_name}.")
    se_t = onp.loadtxt("strain_energy.dat",dtype=float)
    DISP = onp.loadtxt("increments.dat",dtype=float)
    gps = np.array([14.0,5.0,7.0,10.0,1.0,0.2])
    disp_t = DISP[-1]
    nx = 1
    ny = 1
    
    #N = 2
    #for _ in range(N):
    #    gps = update(gps, nx, ny, disp_t, se_t)
    
    cell, sim_fn = construct_simulation(gps, nx, ny, disp_t)
    l2g, g2l = cell.get_global_local_maps()

    print('Simulating initial geometry parameters.')
    sim_time = time.time()
    final_x, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(gps)
    print(f'Finished sim in {time.time() - sim_time} seconds.')

    all_velocities = np.zeros_like(all_displacements)
    all_fixed_vels = np.zeros_like(all_fixed_locs)

    plt.plot(DISP, all_strain_energies, '-b', DISP, se_t, '--r')
    plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}.png"))
    plt.close()

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_static_image(cell.element, g2l(final_x, all_fixed_locs[-1]), image_path)

    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels)
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

    print(f'Finished simulation {args.exp_name}.')

    def loss_fn(gps):
        _, (_, _, se_p) = sim_fn(gps)
        return np.linalg.norm(se_p - se_t)

    print('Starting adjoint optimization')
    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    curr_gps = gps
    lr = 0.0001
    for i in range(1, 101):
        iter_time = time.time()
        loss, grad_loss = loss_val_and_grad(curr_gps)
        print(f'Iteration {i} Loss: {loss} Grad Norm: {np.linalg.norm(grad_loss)} Time: {time.time() - iter_time}')

        curr_gps = curr_gps - lr * grad_loss

        if i > 0 and i % 5 == 0:
            print(f'Generating image and video with optimization so far.')
            optimized_curr_g_pos, (all_displacements, all_fixed_locs, all_strain_energies) = sim_fn(curr_gps)
            image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
            vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
            create_static_image(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1]), image_path)
            ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                all_displacements, all_velocities, all_fixed_locs, all_fixed_vels)
            create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)

            plt.plot(DISP, all_strain_energies, '-b', DISP, se_t, '--r')
            plt.savefig(os.path.join(args.exp_dir, f"compare_SEs-{args.exp_name}-{i}.png"))
            plt.close()

    final_loss = loss_fn(curr_gps)
    print(f'Final loss: {final_loss}')

    #scriptFile_ins = open('updated_inputs.dat', "w")
    #onp.savetxt(scriptFile_ins, gps,"%f")
