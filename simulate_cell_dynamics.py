from comet_ml import Experiment
import time
import os
import argparse
import analysis_utils as autils
import experiment_utils as eutils
from varmint.movie_utils import create_movie
from varmint.cell2d import Cell2D, CellShape, register_dirichlet_bc, register_traction_bc
from varmint.discretize import HamiltonianStepper
from varmint.constitutive import NeoHookean2D
from varmint.materials import Material
from varmint.patch2d import Patch2D
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
parser.add_argument('--strategy', choices=['ilu_preconditioning'],
                    default='ilu_preconditioning')


class WigglyMat(Material):
    _E = 0.03
    _nu = 0.48
    _density = 1.0


def simulate(ref_ctrl, ref_vels, cell, dt, T, strategy, friction=1e-7):
    def friction_force(q, qdot, ref_ctrl, fixed_pos,
                       fixed_vel, tractions): return -friction * qdot

    flatten, unflatten = cell.get_dynamics_flatten_unflatten()
    full_lagrangian = cell.get_lagrangian_fun()

    # Initially in the ref config with zero momentum.
    q, p = flatten(ref_ctrl, ref_vels)
    print(f'Simulation has {q.shape[0]} degrees of freedom.')

    stepper = HamiltonianStepper(
        full_lagrangian, friction_force, save=True, cell=cell)
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


def sim_radii(cell, radii, dt, T, strategy):
    # Construct reference shape.
    ref_ctrl = cell.radii_to_ctrl_fn(radii)

    # Simulate the reference shape.
    QQ, PP, TT, all_fixed, all_fixed_vels = simulate(
        ref_ctrl, np.zeros_like(ref_ctrl), cell, dt, T, strategy)

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

    cell_shape = CellShape(
        num_cp=args.ncp,
        quad_degree=args.quaddeg,
        spline_degree=args.splinedeg,
    )

    grid_str = "C0500 C0500 C0500 C0500 C0500 C0500\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0000 C0000 C0000 C0000 C0000 C0000\n"\
               "C0001 C0001 C0001 C0001 C0001 C0001\n"

    cell = Cell2D(cell_shape=cell_shape, material=mat, instr=grid_str)

    @register_dirichlet_bc('1', cell)
    def group_1_movement(t):
        return t / args.simtime * np.array([0.0, 0.0])

    @register_dirichlet_bc('5', cell)
    def group_2_movement(t):
        return t / args.simtime * np.array([0.0, -3.0])

    @register_traction_bc('B', cell)
    def group_A_traction(t):
        return 1e-1 * np.array([1.0, 0.0])

    @register_traction_bc('D', cell)
    def group_A_traction(t):
        return 1e-1 * np.array([-1.0, 0.0])

    init_radii = None

    dt = np.float64(args.dt)
    T = args.simtime

    radii = np.concatenate(
        (
            cell.generate_bertoldi_radii((cell.n_cells,), 0.12, -0.06),
            # cell.generate_random_radii((cell.n_cells,)),
            # cell.generate_circular_radii((cell.n_cells,)),
        )
    )

    sim_time = time.time()
    QQ, PP, TT, all_fixed, all_fixed_vels = sim_radii(
        cell, radii, dt, T, args.strategy)
    print(f'Total simulation time: {time.time() - sim_time}')
    sim_dir = os.path.join(args.exp_dir, 'sim_ckpt')
    os.mkdir(sim_dir)
    #autils.save_dynamics_simulation(sim_dir, QQ, PP, TT, init_radii, cell)

    # Turn this into a sequence of control point sets.
    ref_ctrl = cell.radii_to_ctrl_fn(radii)
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        QQ, PP, all_fixed, all_fixed_vels)

    print('Saving result in video.')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_movie(cell.patch, ctrl_seq, vid_path, comet_exp=experiment)


if __name__ == '__main__':
    main()
