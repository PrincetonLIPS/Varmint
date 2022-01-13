from comet_ml import Experiment
import time
import os
import argparse


from varmintv2.geometry.multistable2d import construct_multistable2D
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie, create_static_image

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolver, DenseNewtonSolver, DenseNewtonSolverJittable

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
parser.add_argument('-q', '--quaddeg', type=int, default=10)
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

class SteelMat(Material):
    _E = 200.0
    _nu = 0.3
    _density = 8.0


def main():
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

    mat = NeoHookean2D(TPUMat)

    multiplier = 1.0
    cell, ref_ctrl = \
        construct_multistable2D(patch_ncp=args.ncp, quad_degree=args.quaddeg,
                               spline_degree=args.splinedeg, material=mat,
                               multiplier=multiplier)

    potential_energy_fn = cell.get_potential_energy_fn(ref_ctrl)
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(ref_ctrl))

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    #potential_energy_fn = jax.jit(potential_energy_fn)
    #grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    #hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()

    sim_time = time.time()
    curr_g_pos = l2g(ref_ctrl)
    print(f'Optimizing {curr_g_pos.size} degrees of freedom.')

    n_increments = 100
    strain_energies = []
    increments = []

    all_displacements = []
    all_velocities = []

    all_fixed_locs = []
    all_fixed_vels = []

    optimizer = DenseNewtonSolverJittable(cell, potential_energy_fn, max_iter=100, step_size=0.5)
    optimize = jax.jit(optimizer.optimize)
    for i in range(n_increments + 1):
        # Increment displacement a little bit.
        fixed_displacements = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([0.0, -8.0 / n_increments * i]),
            '3': np.array([0.0, 0.0]),
            '4': np.array([0.0, 0.0]),
        }

        tractions = {}

        fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
        tractions = cell.tractions_from_dict(tractions)

        all_fixed_locs.append(fixed_locs)
        all_fixed_vels.append(np.zeros_like(fixed_locs))


        # Solve for new state
        print(f'Starting optimization at iteration {i}.')
        opt_start = time.time()
        new_x, success = optimize(curr_g_pos, (fixed_locs, tractions))
        if not success:
            print(f'Optimization reached max iters.')
            break
        print(f'Optimization finished in {time.time() - opt_start} s')
                                          
        curr_g_pos = new_x
        all_displacements.append(curr_g_pos)
        all_velocities.append(np.zeros_like(curr_g_pos))
        strain_energy = strain_energy_fn(curr_g_pos, fixed_locs, tractions)
        strain_energies.append(strain_energy)
        increments.append(8.0 / n_increments * i)
        print(f'Total strain energy is: {strain_energy} J')

    print('Saving result in image.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_static_image(cell.element, g2l(curr_g_pos, fixed_locs), image_path)

    scriptFile_se = open(os.path.join(args.exp_dir, f'strain_energy.dat'), "w")
    onp.savetxt(scriptFile_se, strain_energies,"%f")
    scriptFile_in = open(os.path.join(args.exp_dir, f'increments.dat'), "w")
    onp.savetxt(scriptFile_in, increments,"%f")
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels)
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)
    
    plt.plot(increments, strain_energies)
    plt.savefig(os.path.join(args.exp_dir, f'strain_energy_graph-{args.exp_name}.png'))
    print(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    main()
