import jax

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import numpy.random as npr
import matplotlib.pyplot as plt

from varmint.cell2d import Cell2D, CellShape
from varmint.materials import Material
from varmint.constitutive import NeoHookean2D
from varmint.discretize import HamiltonianStepper

from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh

import matplotlib.pyplot as plt

import dataset
import surrogate_nns

import experiment_utils as eutils

import os
import time
import itertools
import argparse
import pickle


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(parser, exp_root=None)


# Surrogate parameters.
parser.add_argument('--dsroot', default='/n/fs/mm-iga/Varmint/slurm_experiments/')
parser.add_argument('--dsname', default='newdataset')
parser.add_argument('--stepsize', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--numiters', type=int, default=10000)

parser.add_argument('--skip_weight', type=float, default=1e-1)
parser.add_argument('--skip_connect', dest='skip_connect', action='store_true')

# If debug mode is on, will train and evaluate on a single trajectory.
parser.add_argument('--debug', dest='debug', action='store_true')

parser.add_argument('--loss_type', choices=['metric', 'mse'])

# NN parameters
parser.add_argument('--nn_whidden', type=int, default=2048)
parser.add_argument('--nn_nhidden', type=int, default=3)
parser.add_argument('--nn_activation', choices=['tanh', 'selu', 'relu'], default='tanh')


class WigglyMat(Material):
  _E = 0.003
  _nu = 0.48
  _density = 1.0


def hvp(f, x, v):
  return jax.grad(lambda x: np.vdot(jax.grad(f)(x), v))(x)


def get_residual_fun(args):
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
  friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -1e-7 * qdot

  flatten, unflatten = cell.get_dynamics_flatten_unflatten()
  full_lagrangian = cell.get_lagrangian_fun()

  stepper = HamiltonianStepper(full_lagrangian, friction_force)
  return stepper.residual_fun, simargs['dt'], cell


def main():
  args = parser.parse_args()
  args.dsdir = os.path.join(args.dsroot, args.dsname)
  args.exp_root = os.path.join(args.dsdir, 'trained_surrogates')
  print(f'Experiment directory created under dataset at {args.dsdir}')
  eutils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.

  npr.seed(args.seed)
  jrng = jax.random.PRNGKey(args.seed)

  train_oq, train_op, train_rd, train_nq, test_oq, test_op, test_rd, test_nq = \
      dataset.parse_tensors_with_cache(args.dsdir)

  args.ntrain = train_oq.shape[0]
  args.nfeat  = train_oq.shape[1]
  args.nrad   = train_rd.shape[1]
  args.ntest = test_oq.shape[0]
  eutils.save_args(args)

  print('Initializing.')
  step_size = args.stepsize
  batch_size = args.batchsize
  num_iterations = args.numiters
  momentum_mass = 0.9
  seed = args.seed

  def data_stream():
    rng = npr.RandomState(seed)
    while True:
      inds = rng.randint(0, args.ntrain, batch_size)
      yield train_oq[inds, :], train_op[inds, :], train_rd[inds, :], train_nq[inds, :]

  def test_data_stream():
    rng = npr.RandomState(seed)
    while True:
      inds = rng.randint(0, args.ntest, batch_size)
      yield test_oq[inds, :], test_op[inds, :], test_rd[inds, :], test_nq[inds, :]

  if args.debug:
    def full_train():
      return train_oq[:1000], train_op[:1000], train_rd[:1000], train_nq[:1000]

    def full_test():
      return train_oq[:1000], train_op[:1000], train_rd[:1000], train_nq[:1000]
  else:
    def full_train():
      return train_oq, train_op, train_rd, train_nq

    def full_test():
      return test_oq, test_op, test_rd, test_nq

  batches = data_stream()
  test_batches = test_data_stream()

  init_random_params, nn_fun = \
      surrogate_nns.get_mlp(args.nfeat, args.nn_whidden,
                            args.nn_nhidden, args.nn_activation)

  # Linear classifier. params is an d x o ndarray
  def predict(old_q, old_p, radii, params):
    inputs = np.concatenate((old_q, old_p), axis=1)
    return inputs @ params
    #return old_q

  def solve(old_q, old_p, radii, new_q):
    inputs = np.concatenate((old_q, old_p), axis=1)
    return np.linalg.solve(inputs.T @ inputs, inputs.T @ new_q)

  def mse_loss(old_q, old_p, radii, new_q):
    params = solve(old_q, old_p, radii, new_q)
    return np.mean(np.square(predict(old_q, old_p, radii, params) - new_q)), params

  print('solving')
  mse, params = mse_loss(*full_train())
  print(mse)
  plt.imshow(params)
  plt.colorbar()

  plt.savefig('matrixvis.pdf')


if __name__ == '__main__':
  main()
