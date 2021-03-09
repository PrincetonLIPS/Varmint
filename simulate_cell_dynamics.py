import jax

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import numpy.random as npr

from varmint.patch2d      import Patch2D
from varmint.materials    import Material
from varmint.constitutive import NeoHookean2D
from varmint.discretize   import HamiltonianStepper
from varmint.cell2d       import Cell2D, CellShape
from varmint.movie_utils  import create_movie

import experiment_utils as eutils
import analysis_utils as autils

import argparse
import os
import time


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-x', '--nx', type=int, default=3)
parser.add_argument('-y', '--ny', type=int, default=1)
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=10)
parser.add_argument('-s', '--splinedeg', type=int, default=3)

parser.add_argument('--simtime', type=float, default=0.5)
parser.add_argument('--dt', type=float, default=0.005)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--optimizer', choices=['levmar', 'scipy-lm', 'newtoncg', 'newtoncg-python', 'newtoncg-scipy', 'trustncg-scipy'],
                    default='levmar')


class WigglyMat(Material):
  _E = 0.003
  _nu = 0.48
  _density = 1.0


def simulate(ref_ctrl, ref_vels, cell, dt, T, friction=1e-7):
  friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

  flatten, unflatten = cell.get_dynamics_flatten_unflatten()
  full_lagrangian = cell.get_lagrangian_fun()

  stepper = HamiltonianStepper(full_lagrangian, friction_force)
  step = stepper.construct_stepper(optimkind=args.optimizer)

  # Initially in the ref config with zero momentum.
  q, p = flatten(ref_ctrl, ref_vels)

  QQ = [q]; PP = [p]; TT = [0.0]
  while TT[-1] < T:
    t0 = time.time()
    fixed_locs = ref_ctrl

    success = False
    this_dt = dt
    while True:
      new_q, new_p = step(QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)

      success = np.all(np.isfinite(new_q))
      if success:
        break
      else:
        this_dt = this_dt / 2.0
        print('\tFailed to converge. dt now %f' % (this_dt))

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1] + this_dt)
    t1 = time.time()
    print(f'stepped to time {TT[-1]} in {t1-t0} seconds')

  return QQ, PP, TT


def sim_radii(cell, radii, dt, T):
  # Construct reference shape.
  ref_ctrl = cell.radii_to_ctrl(radii)

  # Simulate the reference shape.
  QQ, PP, TT = simulate(ref_ctrl, np.zeros_like(ref_ctrl), cell, dt, T)

  return QQ, PP, TT


if __name__ == '__main__':
  args = parser.parse_args()
  eutils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.

  eutils.save_args(args)
  npr.seed(args.seed)

  WigglyMat._E = args.E
  mat = NeoHookean2D(WigglyMat)

  cell_shape = CellShape(
    num_x=args.nx,
    num_y=args.ny,
    num_cp=args.ncp,
    quad_degree=args.quaddeg,
    spline_degree=args.splinedeg,
  )

  cell = Cell2D(cell_shape=cell_shape, fixed_side='left', material=mat)
  init_radii = cell.generate_random_radii(args.seed)

  dt = np.float64(args.dt)
  T  = args.simtime

  QQ, PP, TT = sim_radii(cell, init_radii, dt, T)
  sim_dir = os.path.join(args.exp_dir, 'sim_ckpt')
  os.mkdir(sim_dir)
  autils.save_dynamics_simulation(sim_dir, QQ, PP, TT, init_radii, cell)

  # Turn this into a sequence of control point sets.
  ref_ctrl = cell.radii_to_ctrl(init_radii)
  ctrl_seq, _ = cell.unflatten_dynamics_sequence(QQ, PP, ref_ctrl)

  print('Saving result in video.')
  vid_path = os.path.join(args.exp_dir, 'sim.mp4')
  create_movie(cell.patch, ctrl_seq, vid_path)
