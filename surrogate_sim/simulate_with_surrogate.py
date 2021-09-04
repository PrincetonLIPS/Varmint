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
from varmint.discretize   import SurrogateStepper, SurrogateInitStepper
from varmint.cell2d       import Cell2D, CellShape
from varmint.movie_utils  import create_movie

import experiment_utils as eutils
import analysis_utils as autils

import dataset

import argparse
import os
import time


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(parser, exp_root=None)

parser.add_argument('--dsroot', default='/n/fs/mm-iga/Varmint/slurm_experiments/')
parser.add_argument('--dsname', default='newdataset')
parser.add_argument('--surrogate', default='tanhsimple')

parser.add_argument('--stepper', choices=['init', 'surrogate'], default='surrogate')
parser.add_argument('--optimizer', choices=['levmar', 'scipy-lm', 'newtoncg', 'newtoncg-python', 'newtoncg-scipy', 'trustncg-scipy'],
                    default='levmar')


class WigglyMat(Material):
  _E = 0.003
  _nu = 0.48
  _density = 1.0


def simulate(ref_ctrl, ref_vels, cell, dt, T, predict_fun, radii, optimizer, stepper, friction=1e-7):
  friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

  flatten, unflatten = cell.get_dynamics_flatten_unflatten()
  full_lagrangian = cell.get_lagrangian_fun()

  if stepper == 'init':
    stepper = SurrogateInitStepper(full_lagrangian, friction_force)
    step = stepper.construct_stepper(predict_fun, radii, optimkind=optimizer)
  elif stepper == 'surrogate':
    stepper = SurrogateStepper(full_lagrangian, friction_force)
    step = stepper.construct_stepper(predict_fun, radii)
  else:
    raise ValueError(f'Invalid stepper type {stepper}.')

  # Initially in the ref config with zero momentum.
  q, p = flatten(ref_ctrl, ref_vels)

  QQ = [q]; PP = [p]; TT = [0.0]
  while TT[-1] < T:
    t0 = time.time()
    fixed_locs = ref_ctrl

    success = False
    this_dt = dt
    new_q, new_p = step(QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)

    success = np.all(np.isfinite(new_q))
    if not success:
      print(f'Surrogate time stepping diverged... Aborting and returning what we have.')
      return QQ, PP, TT

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1] + this_dt)
    t1 = time.time()
    print(f'stepped to time {TT[-1]} in {t1-t0} seconds')

  return QQ, PP, TT


def main():
  args = parser.parse_args()
  args.dsdir = os.path.join(args.dsroot, args.dsname)
  args.exp_root = os.path.join(args.dsdir, 'surrogate_sims')
  args.exp_name = f'{args.surrogate}-{args.exp_name}'
  eutils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.

  eutils.save_args(args)
  npr.seed(args.seed)

  # Get the arguments for any of the dataset runs
  simckpt = dataset.get_any_ckpt(args.dsdir)
  simargs = simckpt.args

  WigglyMat._E = simargs['E']
  mat = NeoHookean2D(WigglyMat)

  cell_shape = CellShape(
    num_x=simargs['nx'],
    num_y=simargs['ny'],
    num_cp=simargs['ncp'],
    quad_degree=simargs['quaddeg'],
    spline_degree=simargs['splinedeg'],
  )

  cell = Cell2D(cell_shape=cell_shape, fixed_side='left', material=mat)
  init_radii = cell.generate_random_radii(args.seed)  # Do NOT use the seed from simargs.

  dt = np.float64(simargs['dt'])
  T  = simargs['simtime']

  ref_ctrl = cell.radii_to_ctrl(init_radii)
  ref_vels = np.zeros_like(ref_ctrl)

  # Get surrogate
  surrogate = eutils.read_surrogate_experiment(args.dsdir, args.surrogate)

  QQ, PP, TT = simulate(ref_ctrl, ref_vels, cell, dt, T,
                        surrogate.predict_fun, init_radii, args.optimizer, args.stepper)

  sim_dir = os.path.join(args.exp_dir, 'sim_ckpt')
  os.mkdir(sim_dir)
  autils.save_dynamics_simulation(sim_dir, QQ, PP, TT, init_radii, cell)

  # Turn this into a sequence of control point sets.
  ref_ctrl = cell.radii_to_ctrl(init_radii)
  ctrl_seq, _ = cell.unflatten_dynamics_sequence(QQ, PP, ref_ctrl)

  print('Saving result in video.')
  vid_path = os.path.join(args.exp_dir, 'sim.mp4')
  create_movie(cell.patch, ctrl_seq, vid_path)
  print(f'Experiment dir at: {args.exp_dir}')


if __name__ == '__main__':
  main()
