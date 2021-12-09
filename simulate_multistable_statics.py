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

import scipy.optimize

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

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


class WigglyMat(Material):
    _E = 0.03
    _nu = 0.48
    _density = 1.0


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

    WigglyMat._E = args.E
    mat = NeoHookean2D(WigglyMat)

    multiplier = 1.0
    cell, ref_ctrl = \
        construct_multistable2D(patch_ncp=args.ncp, quad_degree=args.quaddeg,
                               spline_degree=args.splinedeg, material=mat,
                               multiplier=multiplier)

    potential_energy_fn = cell.get_potential_energy_fn(ref_ctrl)
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(ref_ctrl))

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    potential_energy_fn = jax.jit(potential_energy_fn)
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()

    sim_time = time.time()
    curr_g_pos = l2g(ref_ctrl)

    n_increments = 30
    for i in range(n_increments + 1):
        # Increment displacement a little bit.
        fixed_displacements = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([0.0, -0.4 / n_increments * i]),
        }

        tractions = {}

        fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
        tractions = cell.tractions_from_dict(tractions)

        # Solve for new state
        print(f'Starting optimization at iteration {i}.')
        opt_start = time.time()
        results = scipy.optimize.minimize(potential_energy_fn, curr_g_pos,
                                          args=(fixed_locs, tractions),
                                          method='trust-ncg',
                                          jac=grad_potential_energy_fn,
                                          hess=hess_potential_energy_fn,
                                          #options={'maxiter': 10000}
                                          )
        print(f'Optimization succeeded: {results.success}.')
        print(f'Took {time.time() - opt_start} seconds.')
        if not results.success:
            print(f'Optimization failed with status {results.status}.')
            print(results.message)
            break

        curr_g_pos = results.x
        strain_energy = strain_energy_fn(curr_g_pos, fixed_locs, tractions)
        print(f'Total strain energy is: {strain_energy} J')

    print('Saving result in video.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.png')
    create_static_image(cell.element, g2l(curr_g_pos, fixed_locs), image_path)
    print(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    main()
