from comet_ml import Experiment
import time
import os
import argparse


from varmintv2.geometry.multistable2d import construct_multistable2D
from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D, LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie
from varmintv2.utils.geometry_utils import verify_constraints

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

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

parser.add_argument('--simtime', type=float, default=5000.0)
parser.add_argument('--dt', type=float, default=50.0)

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


def simulate(ref_ctrl, ref_vels, cell: Geometry,
             dt, T, strategy, friction=0.0): # 1e-7):
    def friction_force(q, qdot, ref_ctrl, fixed_pos,
                       fixed_vel, tractions): return -friction * qdot

    flatten, unflatten = cell.get_global_local_maps()
    full_lagrangian = cell.get_lagrangian_fn()

    # Initially in the ref config with zero momentum.
    q = flatten(ref_ctrl)
    p = flatten(ref_vels)
    print(f'Simulation has {q.shape[0]} degrees of freedom.')

    stepper = HamiltonianStepper(
        full_lagrangian, geometry=cell, F=friction_force, save=True)
    step, optimizer = stepper.construct_stepper(q.shape[0], strategy=strategy)

    fixed_locs_fn, fixed_vels_fn = cell.get_fixed_locs_fn(ref_ctrl)
    traction_fn = cell.get_traction_fn()

    QQ = [q]
    PP = [p]
    TT = [0.0]
    all_fixed = [fixed_locs_fn(TT[-1])]
    all_fixed_vels = [fixed_vels_fn(TT[-1])]

    while TT[-1] < T:
        t0 = time.time()
        fixed_locs = all_fixed[-1]
        fixed_vels = all_fixed_vels[-1]

        success = False
        this_dt = dt
        while True:
            new_q, new_p = step(
                QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs, fixed_vels, traction_fn(TT[-1]))

            success = np.all(np.isfinite(new_q))
            if success:
                break
            else:
                this_dt = this_dt / 2.0
                print('\tFailed to converge. dt now %f' % (this_dt))

        QQ.append(new_q)
        PP.append(new_p)
        TT.append(TT[-1] + this_dt)
        all_fixed.append(fixed_locs_fn(TT[-1]))
        all_fixed_vels.append(fixed_locs_fn(TT[-1]))
        t1 = time.time()
        print(f'stepped to time {TT[-1]} in {t1-t0} seconds')

    print(optimizer.stats)
    return QQ, PP, TT, all_fixed, all_fixed_vels


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

    multiplier = 10.0
    cell, init_ctrl = \
        construct_multistable2D(patch_ncp=args.ncp, quad_degree=args.quaddeg,
                               spline_degree=args.splinedeg, material=mat,
                               multiplier=multiplier)

    @cell.register_dirichlet_bc('1')
    def group_1_movement(t):
        return t / args.simtime * np.array([0.0, 0.0])

    @cell.register_dirichlet_bc('2')
    def group_2_movement(t):
        return t / args.simtime * np.array([0.0, -0.4 * multiplier])

    @cell.register_dirichlet_bc('3')
    def group_3_movement(t):
        return t / args.simtime * np.array([0.0, 0.0])

    @cell.register_dirichlet_bc('4')
    def group_4_movement(t):
        return t / args.simtime * np.array([0.0, 0.0])

    dt = np.float64(args.dt)
    T = args.simtime

    sim_time = time.time()

    QQ, PP, TT, all_fixed, all_fixed_vels = \
        simulate(init_ctrl, np.zeros_like(init_ctrl), cell, dt, T, args.strategy)

    print(f'Total simulation time: {time.time() - sim_time}')
    sim_dir = os.path.join(args.exp_dir, 'sim_ckpt')
    os.mkdir(sim_dir)

    # Turn this into a sequence of control point sets.
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        QQ, PP, all_fixed, all_fixed_vels)

    print('Saving result in video.')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=experiment)
    print(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    main()
