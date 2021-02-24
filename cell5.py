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

  friction = 1e-7

  # Create patch parameters.c
  quad_deg   = args.quaddeg
  spline_deg = args.splinedeg
  num_ctrl   = args.ncp
  num_x      = args.nx
  num_y      = args.ny

  xknots     = default_knots(spline_deg, num_ctrl)
  yknots     = default_knots(spline_deg, num_ctrl)
  widths     = 5*np.ones(num_x)
  heights    = 5*np.ones(num_y)

  init_radii = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
  init_radii = np.array(init_radii)
  #init_radii = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
  init_ctrl  = generate_quad_lattice(widths, heights, init_radii)
  n_components, index_arr = index_array_from_ctrl(num_x, num_y, init_ctrl)
  left_side  = onp.array(init_ctrl[:,:,:,0] == 0.0)
  fixed_labels = index_arr[left_side]
  print(f'fixed labels: {fixed_labels.shape}')

  def flatten_add(unflat_ctrl, unflat_vel):
    almost_flat     = jax.ops.index_add(np.zeros((n_components, 2)), index_arr, unflat_ctrl)
    almost_flat_vel = jax.ops.index_add(np.zeros((n_components, 2)), index_arr, unflat_vel)
    return almost_flat.flatten(), almost_flat_vel.flatten()

  def flatten(unflat_ctrl, unflat_vel):
    almost_flat = jax.ops.index_update(np.zeros((n_components, 2)), index_arr, unflat_ctrl)
    almost_flat_vel = jax.ops.index_update(np.zeros((n_components, 2)), index_arr, unflat_vel)
    return almost_flat.flatten(), almost_flat_vel.flatten()

  fixed_locations = flatten(init_ctrl, np.zeros_like(init_ctrl))[0].reshape((n_components, 2))
  fixed_locations = np.take(fixed_locations, fixed_labels, axis=0)

  def unflatten(flat_ctrl, flat_vels, fixed_locs):
    fixed_locs = flatten(fixed_locs, np.zeros_like(fixed_locs))[0].reshape((n_components, 2))
    fixed_locs = np.take(fixed_locs, fixed_labels, axis=0)

    flat_ctrl  = flat_ctrl.reshape(n_components, 2)
    flat_vels  = flat_vels.reshape(n_components, 2)
    fixed      = jax.ops.index_update(flat_ctrl, fixed_labels, fixed_locs)
    fixed_vels = jax.ops.index_update(flat_vels, fixed_labels, np.zeros_like(fixed_locs))
    return np.take(fixed, index_arr, axis=0), np.take(fixed_vels, index_arr, axis=0)

  def unflatten_nofixed(flat_ctrl, flat_vels):
    flat_ctrl = flat_ctrl.reshape(n_components, 2)
    flat_vels = flat_vels.reshape(n_components, 2)
    return np.take(flat_ctrl, index_arr, axis=0), np.take(flat_vels, index_arr, axis=0)

  # Create the shape.
  shape = Shape2D(*[
    Patch2D(
      xknots,
      yknots,
      spline_deg,
      mat,
      quad_deg,
      None, #labels[ii,:,:],
      fixed_labels, # <-- Labels not locations
    )
    for  ii in range(len(init_ctrl))
  ])

  patch = Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    None, #labels[ii,:,:],
    fixed_labels, # <-- Labels not locations
  )

  friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

  def displacement(t):
    return np.sin(4 * np.pi * t) * np.ones_like(init_ctrl)

  p_lagrangian = generate_patch_lagrangian(patch)

  def full_lagrangian(q, qdot, ref_ctrl, displacement):
    def_ctrl, def_vels = unflatten(q, qdot, displacement)
    return np.sum(jax.vmap(p_lagrangian)(def_ctrl, def_vels, ref_ctrl))

  stepper = get_hamiltonian_stepper(full_lagrangian, friction_force,
                                    optimkind=args.optimizer)

  dt = np.float32(0.005)
  T  = 0.5

  def simulate(ref_ctrl):

    # Initially in the ref config with zero momentum.
    q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

    QQ = [ q ]
    PP = [ p ]
    TT = [ 0.0 ]

    while TT[-1] < T:

      t0 = time.time()
      #fixed_locs = displacement(TT[-1]) + ref_ctrl
      fixed_locs = ref_ctrl

      success = False
      this_dt = dt
      while True:
        new_q, new_p = stepper(QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)
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
    return generate_quad_lattice(widths, heights, radii)

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


  val, ctrl_seq = loss(init_radii)

  print('Saving result in video.')
  vid_path = os.path.join(args.exp_dir, 'sim.mp4')
  shape.create_movie(ctrl_seq, vid_path, labels=False)

  quit()

  valgrad_loss = jax.value_and_grad(loss, has_aux=True)

  radii = init_radii

  lr = 1.0
  for ii in range(5):
    (val, ctrl_seq), gradmo = valgrad_loss(radii)
    print()
    #print(radii)
    print(val)
    print(gradmo)

    shape.create_movie(ctrl_seq, 'cell5-%d.mp4' % (ii+1), labels=False)

    radii = np.clip(radii - lr * gradmo, 0.05, 0.95)
