import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import string

from varmint.patch2d      import Patch2D
from varmint.shape2d      import Shape2D
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D
from varmint.bsplines     import default_knots
from varmint.lagrangian   import generate_patch_lagrangian
from varmint.discretize   import get_hamiltonian_stepper
from varmint.levmar       import get_lmfunc
from varmint.cellular2d   import index_array_from_ctrl, generate_quad_lattice
from varmint.cell2d       import Cell2D, CellShape
from varmint.movie_utils  import create_movie

import experiment_utils as exputils

import json
import logging
import random
import argparse
import time
import os

#from varmint.grad_graph import grad_graph
#import jax.profiler
#server = jax.profiler.start_server(9999)


parser = argparse.ArgumentParser()
exputils.prepare_experiment_args(parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-x', '--nx', type=int, default=3)
parser.add_argument('-y', '--ny', type=int, default=1)
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=10)
parser.add_argument('-s', '--splinedeg', type=int, default=3)

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--optimizer', choices=['levmar', 'scipy-lm', 'newtoncg', 'newtoncg-python', 'newtoncg-scipy', 'trustncg-scipy'],
                    default='levmar')


class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0


class CollapsingMat(Material):
  _E = 0.00005
  _nu = 0.48
  _density = 1.0

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

def save_optimization(args, old_q, old_p, t, dt, ref_ctrl, fixed_locs, new_q, new_p):
  save_dir = os.path.join(args.exp_dir, 'optckpts', str(float(t)))
  print(f'saving to dir {save_dir}')

  assert not os.path.exists(save_dir)
  os.makedirs(save_dir)

  metadata = {}
  metadata['time'] = float(t)
  metadata['dt'] = float(dt)
  metadata['global_params'] = vars(args)

  with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)

  onp.save(os.path.join(save_dir, 'old_q.npy'), onp.asarray(old_q))
  onp.save(os.path.join(save_dir, 'old_p.npy'), onp.asarray(old_p))
  onp.save(os.path.join(save_dir, 'ref_ctrl.npy'), onp.asarray(ref_ctrl))
  onp.save(os.path.join(save_dir, 'fixed_locs.npy'), onp.asarray(fixed_locs))
  onp.save(os.path.join(save_dir, 'new_q.npy'), onp.asarray(new_q))
  onp.save(os.path.join(save_dir, 'new_p.npy'), onp.asarray(new_p))

if __name__ == '__main__':
  args = parser.parse_args()
  exputils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.
  npr.seed(args.seed)

  mat = NeoHookean2D(WigglyMat)

  cell_shape = CellShape(
    num_x=args.nx,
    num_y=args.ny,
    num_cp=args.ncp,
    quad_degree=args.quaddeg,
    spline_degree=args.splinedeg,
  )

  cell = Cell2D(cell_shape=cell_shape, init_radii='random',
                fixed_side='left', material=mat)
  flatten, unflatten = cell.get_dynamics_flatten_unflatten()
  full_lagrangian = cell.get_lagrangian_fun()

  dt = np.float64(0.005)
  T  = 0.5

  friction = 1e-7
  friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

  #q, p = flatten(cell.ref_ctrl, np.zeros_like(cell.ref_ctrl))
  #scale_args = (q, p, dt, (cell.ref_ctrl, cell.ref_ctrl))
  stepper, residual_fun, diagD = \
          get_hamiltonian_stepper(full_lagrangian, friction_force,
                                  optimkind=args.optimizer,# scale_args=scale_args,
                                  return_residual=True)

  def simulate(ref_ctrl):

    # Initially in the ref config with zero momentum.
    q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

    QQ = [ q ]
    PP = [ p ]
    TT = [ 0.0 ]

    while TT[-1] < T:

      t0 = time.time()
      fixed_locs = ref_ctrl

      success = False
      this_dt = dt
      while True:
        new_q, new_p = stepper(QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)
        #resids_before = residual_fun(QQ[-1] * diagD, (QQ[-1], PP[-1], this_dt, (ref_ctrl, fixed_locs)))
        #resids_after  = residual_fun(new_q * diagD,  (QQ[-1], PP[-1], this_dt, (ref_ctrl, fixed_locs)))

        #resid_grad = jax.jacfwd(residual_fun)(QQ[-1] * diagD, (QQ[-1], PP[-1], this_dt, (ref_ctrl, fixed_locs)))
        #print(f'Shape of resids: {resids_before.shape}')
        #print(f'Shape of resid grads: {resid_grad.shape}')
        #print(f'Rank of Jacobian: {np.linalg.matrix_rank(resid_grad)}')
        #print(f'Rank of GN Matrix: {np.linalg.matrix_rank(resid_grad.T @ resid_grad)}')
        #print(f'Condition number of GN Matrix: {np.linalg.cond(resid_grad.T @ resid_grad)}')
        #print(f'Before residual norm: {np.linalg.norm(resids_before)}')
        #print(f'After norm: {np.linalg.norm(resids_after)}')

        success = np.all(np.isfinite(new_q))
        if success:
          if args.save:
            save_optimization(args, QQ[-1], PP[-1], TT[-1],
                              this_dt, ref_ctrl, fixed_locs, new_q, new_p)
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

  def radii_to_ctrl(radii):
    return cell.radii_to_ctrl(radii)

  def sim_radii(radii):

    # Construct reference shape.
    ref_ctrl = radii_to_ctrl(radii)

    # Simulate the reference shape.
    QQ, PP, TT = simulate(ref_ctrl)

    # Turn this into a sequence of control point sets.
    ctrl_seq = [
      unflatten(
        qt[0],
        np.zeros_like(qt[0]),
        ref_ctrl, # + displacement(qt[1]),
      )[0] \
      for qt in zip(QQ, TT)
    ]

    return ctrl_seq

  def loss(radii):
    ctrl_seq = sim_radii(radii)

    return -np.mean(ctrl_seq[-1]), ctrl_seq


  val, ctrl_seq = loss(cell.init_radii)

  print('Saving result in video.')
  vid_path = os.path.join(args.exp_dir, 'sim.mp4')
  create_movie(cell.patch, ctrl_seq, vid_path)

  quit()

  valgrad_loss = jax.value_and_grad(loss, has_aux=True)

  radii = cell.init_radii

  lr = 1.0
  for ii in range(5):
    (val, ctrl_seq), gradmo = valgrad_loss(radii)
    print()
    #print(radii)
    print(val)
    print(gradmo)

    create_movie(cell.patch, ctrl_seq, 'cell5-%d.mp4' % (ii+1))

    radii = np.clip(radii - lr * gradmo, 0.05, 0.95)
