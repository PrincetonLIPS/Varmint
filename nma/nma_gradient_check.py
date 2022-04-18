import pickle
from comet_ml import Experiment
import time
import os
import argparse

import sys
import os
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from construct_nma_shape import construct_cell2D, generate_bertoldi_radii, generate_circular_radii, generate_rectangular_radii
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie_nma, create_static_image_nma

import jax.experimental.host_callback as hcb

from varmintv2.utils import analysis_utils as autils
from varmintv2.utils import experiment_utils as eutils

from varmintv2.solver.optimization import SparseNewtonSolverHCB, SparseNewtonSolverHCBRestart, SparseNewtonSolverHCBRestartPrecondition

import scipy.optimize

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax
from jax.flatten_util import ravel_pytree

import optax
import haiku as hk

import matplotlib.pyplot as plt

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(
    parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=5)
parser.add_argument('-s', '--splinedeg', type=int, default=2)

parser.add_argument('--simtime', type=float, default=50.0)
parser.add_argument('--dt', type=float, default=0.5)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--comet', dest='comet', action='store_true')

parser.add_argument('--load_iter', type=int, default=-1)


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


if __name__ == '__main__':
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args, reload=True)
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

    grid_str = "C0000 C0300 C0000\n"\
               "C2000 S0000 C0040\n"\
               "C0000 C0005 C0000\n"

    cell, radii_to_ctrl_fn, n_cells = \
        construct_cell2D(input_str=grid_str, patch_ncp=args.ncp,
                         quad_degree=args.quaddeg, spline_degree=args.splinedeg,
                         material=mat)

    init_radii = np.concatenate(
        (
            #generate_bertoldi_radii((n_cells,), args.ncp, 0.12, -0.06),
            #generate_circular_radii((1,), args.ncp),
            generate_rectangular_radii((n_cells,), args.ncp),
        )
    )
    potential_energy_fn = cell.get_potential_energy_fn()
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn())

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    potential_energy_fn = jax.jit(potential_energy_fn)
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()

    ref_ctrl = radii_to_ctrl_fn(np.array(init_radii))
    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, {})
    tractions = cell.tractions_from_dict({})

    optimizer = SparseNewtonSolverHCBRestartPrecondition(cell, potential_energy_fn, max_iter=1000,
                                                         step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10)

    x0 = l2g(ref_ctrl, ref_ctrl)
    optimize = optimizer.get_optimize_fn(x0, (fixed_locs, tractions, ref_ctrl))

    n_increments = 100

    @jax.jit
    def simulate(disps, radii):
        ref_ctrl = radii_to_ctrl_fn(radii)
        init_x = l2g(ref_ctrl, ref_ctrl)

        increments = disps / n_increments
        increments = increments[..., np.newaxis] * np.arange(n_increments + 1)
        increments = increments.T  # increments is (n_increments, n_boundaries)

        def sim_increment(x_prev, increment):
            #fixed_displacements = {
            #    '99': np.array([0.0, 0.0]),
            #    '98': np.array([0.0, 0.0]),
            #    '97': np.array([0.0, 0.0]),
            #    '96': np.array([0.0, 0.0]),
            #    '2': np.array([-increment[0], 0.0]),
            #    '3': np.array([0.0, -increment[1]]),
            #    '4': np.array([-increment[2], 0.0]),
            #    '5': np.array([0.0, -increment[3]]),
            #}
            fixed_displacements = {
                '99': np.array([0.0, 0.0]),
                '98': np.array([0.0, 0.0]),
                '97': np.array([0.0, 0.0]),
                '96': np.array([0.0, 0.0]),
                '2': np.array([-increment[0], 0.0]),
                #'3': np.array([0.0, -increment[1]]),
                '3': np.array([0.0, 0.0]),
                '4': np.array([-increment[2], 0.0]),
                '5': np.array([0.0, 0.0]),
                #'5': np.array([0.0, -increment[3]]),
            }

            fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
            new_x = optimize(x_prev, (fixed_locs, tractions, ref_ctrl))
            strain_energy = strain_energy_fn(new_x, fixed_locs, tractions, ref_ctrl)

            return new_x, (new_x, fixed_locs, strain_energy)
        
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = jax.lax.scan(sim_increment, init_x, increments)
        return final_x, (all_xs, all_fixed_locs, all_strain_energies)

    p1 = np.sum(np.abs(radii_to_ctrl_fn(init_radii) - np.array([7.5, 7.5])), axis=-1) < 1e-14

    test_pts = np.array([
        [6.0, 9.0],
    ])

    test_disps = np.array([6.0, 9.0])

    def clip_fn(x):
        return np.clip(x, -2.5, 2.5)

    def tanh_clip(x):
        return np.tanh(x) * 2.5
    def nn_fn(input):
        mlp = hk.Sequential([
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(30), jax.nn.softplus,
            hk.Linear(10), jax.nn.softplus,
            hk.Linear(4),   tanh_clip,
        ])

        return mlp(input)

    nn_fn_t = hk.transform(nn_fn)
    nn_fn_t = hk.without_apply_rng(nn_fn_t)
    rng = jax.random.PRNGKey(22)
    dummy_displacements = np.array([0.0, 0.0])
    init_nn_params = nn_fn_t.init(rng, dummy_displacements)

    def loss_fn(all_params, displacements):
        delta = displacements - np.array([7.5, 7.5])
        nn_params, radii = all_params
        mat_inputs = nn_fn_t.apply(nn_params, delta)
        final_x, (all_xs, all_fixed_locs, all_strain_energies) = simulate(mat_inputs, radii)
        final_x_local = g2l(final_x, all_fixed_locs[-1], radii_to_ctrl_fn(radii))

        # We want our identified point (p1) at a specified location (displacements).
        return np.sum(np.abs(final_x_local[p1] - displacements)) / ref_ctrl[p1].shape[0]

    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    print('Loading parameters.')
    with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{args.load_iter}.pkl'), 'rb') as f:
        curr_all_params = pickle.load(f)
    print('\tDone.')

    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    flat_params, unravel_tree = ravel_pytree(curr_all_params)

    # Generate a bunch of targets and random directions.
    # The directional derivative (grad @ dir) should match the finite difference.
    hs = [1e-4, 1e-6, 1e-8]
    for i in range(10):
        target_disps = onp.random.uniform(6.0, 9.0, size=2)
        random_dir = onp.random.randn(flat_params.size)
        random_dir = random_dir / onp.linalg.norm(random_dir)

        loss, loss_grad = loss_val_and_grad(curr_all_params, target_disps)
        flat_grad, _ = ravel_pytree(loss_grad)
        print(f'Grad with AD in random direction: {flat_grad @ random_dir}')
        for h in hs:
            perturbed_params = flat_params + random_dir * h
            d_loss, _ = loss_val_and_grad(unravel_tree(perturbed_params), target_disps)
            print(f'\th={h}: {(d_loss - loss) / h}')
        print('\n')