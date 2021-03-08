import jax

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import numpy.random as npr

from varmint.patch2d      import Patch2D
from varmint.constitutive import NeoHookean2D, LinearElastic2D
from varmint.materials    import Material
from varmint.cell2d       import Cell2D, CellShape
from varmint.movie_utils  import create_static_image
from varmint.statics      import DenseStaticsSolver, SparseStaticsSolver

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

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--optimizer', choices=['newton', 'newtoncg-scipy', 'trustncg-scipy',
                                            'bfgs-scipy'], default='newton')


class WigglyMat(Material):
  _E = 0.003
  _nu = 0.48
  _density = 1.0


def simulate(ref_ctrl, cell, sparse=False, optimkind='newton'):
  flatten, unflatten = cell.get_statics_flatten_unflatten()

  q = flatten(ref_ctrl)
  if sparse:
    print(f'Using sparse solver.')
    solver = SparseStaticsSolver(cell)
  else:
    print(f'Using dense solver.')
    solver = DenseStaticsSolver(cell)
  solve  = solver.get_solver_fun(optimkind=optimkind)

  new_q = solve(q, ref_ctrl)

  return unflatten(new_q, ref_ctrl)


if __name__ == '__main__':
  args = parser.parse_args()
  eutils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.

  eutils.save_args(args)
  npr.seed(args.seed)

  WigglyMat._E = args.E
  mat = eval(args.mat_model)(WigglyMat)

  cell_shape = CellShape(
    num_x=args.nx,
    num_y=args.ny,
    num_cp=args.ncp,
    quad_degree=args.quaddeg,
    spline_degree=args.splinedeg,
  )

  cell = Cell2D(cell_shape=cell_shape, fixed_side='left', material=mat)
  init_radii = cell.generate_random_radii(args.seed)

  print('Starting statics simulation')
  ctrl_sol = simulate(cell.radii_to_ctrl(init_radii), cell,
                      sparse=args.sparse, optimkind=args.optimizer)
  print('Saving result in figure.')
  im_path = os.path.join(args.exp_dir, 'result.png')
  create_static_image(cell.patch, ctrl_sol, im_path)


# TODO(doktay): Incorporate adjoint optimization again.
#  free_energy = cell.get_free_energy_fun(patchwise=True)
#
#  def loss_and_adjoint_grad(loss_fn, init_radii):
#    # loss_fn should be a function of ctrl_seq
#    grad_loss = jax.jit(jax.grad(loss_fn))
#    ctrl_sol = simulate(cell.radii_to_ctrl(init_radii), cell)
#    dJdu = grad_loss(ctrl_sol)
#
#    def inner_loss(radii, def_ctrl):
#      ref_ctrl = cell.radii_to_ctrl(radii)
#
#      # So that fixed control points work out. This is hacky.
#      flat     = flatten(def_ctrl)
#      unflat   = unflatten(flat, fixed_locations)
#
#      all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
#      return np.sum(jax.vmap(lambda x: free_energy(x[..., 0], x[..., 1]))(all_args))
#
#    loss_val = loss_fn(ctrl_sol)
#    implicit_fn = jax.jit(jax.grad(inner_loss, argnums=1))
#    implicit_vjp = jax.jit(jax.vjp(implicit_fn, init_radii, ctrl_sol)[1])
#
#    def vjp_ctrl(v):
#      return implicit_vjp(v)[1]
#
#    def vjp_radii(v):
#      return implicit_vjp(v)[0]
#
#    flat_size = ctrl_sol.flatten().shape[0]
#    unflat_size = ctrl_sol.shape
#
#    def spmatvec(v):
#      v = v.reshape(ctrl_sol.shape)
#      vjp = implicit_vjp(v)[1]
#      return vjp.flatten()
#
#    A = scipy.sparse.linalg.LinearOperator((flat_size,flat_size), matvec=spmatvec)
#
#    # Precomputing full Jacobian might be better
#    #print('precomputing hessian')
#    #hess = jax.jacfwd(implicit_fn, argnums=1)(init_radii, ctrl_sol)
#    #print(f'computed hessian with shape {hess.shape}')
#
#    print('solving adjoint equation')
#
#    adjoint, info = scipy.sparse.linalg.minres(A, dJdu.flatten())
#    adjoint = adjoint.reshape(unflat_size)
#    grad = vjp_radii(adjoint)
#
#    return loss_val, -grad, ctrl_sol
#
#  def sample_loss_fn(ctrl):
#    return np.mean(ctrl[..., 0])
#
#  def close_to_center_loss_fn(ctrl):
#    return np.linalg.norm(ctrl[..., 0] - 10)
#
#
#
#  print('Starting training')
#  lr = 1.0
#  for ii in range(1):
#    loss_val, loss_grad, ctrl_sol = loss_and_adjoint_grad(sample_loss_fn, radii)
#    print()
#    print(loss_val)
#
#    shape.create_movie([ctrl_sol], 'long-hanging-cells-%d.mp4' % (ii+1), labels=False)
#
#    radii = np.clip(radii - lr * loss_grad, 0.05, 0.95)
