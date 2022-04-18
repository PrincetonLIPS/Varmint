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

import optax
import haiku as hk

import matplotlib.pyplot as plt
import multiprocessing as mp

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

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--reload', dest='reload', action='store_true')
parser.add_argument('--load_iter', type=int, default=-1)

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


def worker(args, in_q, out_q, dev_id):
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

    print(f'Starting NMA optimization on device {dev_id}')
    
    if dev_id == 0:
        curr_all_params = (init_nn_params, init_radii)
        if args.reload:
            print('Loading parameters.')
            with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{args.load_iter}.pkl'), 'rb') as f:
                curr_all_params = pickle.load(f)
            print('\tDone.')
            iter_num = args.load_iter
            processed_iter_num = args.load_iter
        else:
            iter_num = 0
            processed_iter_num = 0

        lr = 0.0001 * jax.device_count()

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(curr_all_params)
        num_from_iter = 0
        results_from_iter = []
    else:
        curr_all_params = None
        optimizer = None
        opt_state = None

    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn), device=jax.devices()[dev_id // args.tasks_per_device])

    target_disps = onp.array([
        [6.0, 6.0],
        [6.0, 9.0],
        [9.0, 6.0],
        [9.0, 9.0],
        [7.5, 8.5],
        [7.5, 6.5],
        [8.5, 7.5],
        [6.5, 7.5],
    ])

    ewa_loss = None
    ewa_weight = 0.95
    while True:
        def simulate_element(params, disp):
            loss, grad_loss = loss_val_and_grad(params, disp)
            out_q.put((loss, grad_loss))

        def process_batch(i, ewa_loss, iter_time, opt_state, curr_all_params):
            losses = [res[0] for res in results_from_iter]
            lgrads = [res[1] for res in results_from_iter]

            # Combine the losses from parallelism
            loss = np.mean(np.stack(losses, axis=0), axis=0)
            grad_loss = jax.tree_map(
                lambda *g: np.mean(np.stack(g, axis=0), axis=0), *lgrads
            )

            if ewa_loss == None:
                ewa_loss = loss
            else:
                ewa_loss = ewa_loss * ewa_weight + loss * (1 - ewa_weight)
            print(f'Iteration {i} Loss: {loss} EWA Loss: {ewa_loss} Radii Grad Norm: {np.linalg.norm(grad_loss[1])} Time: {time.time() - iter_time}')

            updates, opt_state = optimizer.update(grad_loss, opt_state)
            curr_all_params = optax.apply_updates(curr_all_params, updates)
            curr_nn_params, curr_radii = curr_all_params
            curr_radii = np.clip(curr_radii, 0.1, 0.9)
            curr_all_params = (curr_nn_params, curr_radii)

            if i % 10 == 0:
                print(f'Generating image and video with optimization so far.')
                delta = test_disps - np.array([7.5, 7.5])
                curr_nn_params, curr_radii = curr_all_params
                mat_inputs = nn_fn_t.apply(curr_nn_params, delta)

                optimized_curr_g_pos, (all_displacements, all_fixed_locs, _) = simulate(mat_inputs, curr_radii)

                all_velocities = np.zeros_like(all_displacements)
                all_fixed_vels = np.zeros_like(all_fixed_locs)

                image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.png')
                vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}-optimized-{i}.mp4')
                create_static_image_nma(cell.element, g2l(optimized_curr_g_pos, all_fixed_locs[-1], radii_to_ctrl_fn(curr_radii)), image_path, test_pts)
                ctrl_seq, _ = cell.unflatten_dynamics_sequence(
                    all_displacements, all_velocities, all_fixed_locs, all_fixed_vels, radii_to_ctrl_fn(curr_radii))
                create_movie_nma(cell.element, ctrl_seq, vid_path, test_pts, comet_exp=None)

                # Pickle parameters
                print('Saving parameters.')
                with open(os.path.join(args.exp_dir, f'sim-{args.exp_name}-params-{i}.pkl'), 'wb') as f:
                    pickle.dump(curr_all_params, f)
                print('\tDone.')
            
            return ewa_loss, iter_time, opt_state, curr_all_params


        # If dev_id = 0, then this is the parameter server. Otherwise, it is a measly worker.
        if dev_id == 0:
            if processed_iter_num == iter_num:
                iter_num += 1
                # Prepare the targets for all workers (including our own)
                iter_time = time.time()

                target_disps = onp.random.uniform(6.0, 9.0, size=(jax.device_count() * args.tasks_per_device, 2))
                for disps in target_disps:
                    in_q.put((curr_all_params, disps))
        
            try:
                loss, grad_loss = out_q.get(False)
                results_from_iter.append((loss, grad_loss))
                num_from_iter += 1
            except:
                time.sleep(0.1)

            if num_from_iter == target_disps.shape[0]:
                processed_iter_num += 1
                ewa_loss, iter_time, opt_state, curr_all_params = process_batch(processed_iter_num, ewa_loss, iter_time, opt_state, curr_all_params)
                num_from_iter = 0
                results_from_iter = []

            elif num_from_iter > target_disps.shape[0]:
                print(f"Something is wrong!!!!!")
            
        try:
            params, disps = in_q.get(False)
            simulate_element(params, disps)
        except:
            time.sleep(0.1)


if __name__ == '__main__':
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args, reload=args.reload)
    # args.seed and args.exp_dir should be set.

    eutils.save_args(args)
    npr.seed(args.seed)

    args.tasks_per_device = 1

    if args.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="gTBUDHLLNaIqxMWyjHKQgtlkW",
            project_name="general",
            workspace="denizokt",
        )
    else:
        experiment = None

    mp.set_start_method('spawn')
    in_q = mp.Queue()
    out_q = mp.Queue()
    workers = []
    for dev_id in range(jax.device_count() * args.tasks_per_device):
        p = mp.Process(target=worker, args=(args, in_q, out_q, dev_id))
        p.start()
        workers.append(p)
    
    workers[0].join()