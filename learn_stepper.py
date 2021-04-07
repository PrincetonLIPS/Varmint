import jax

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import numpy.random as npr

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
      return train_oq[:100], train_op[:100], train_rd[:100], train_nq[:100]

    def full_test():
      return train_oq[:100], train_op[:100], train_rd[:100], train_nq[:100]
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

  def predict(old_q, old_p, radii, params):
    inputs = np.concatenate((old_q, old_p, radii), axis=1)
    nn_out = nn_fun(params, inputs)

    if args.skip_connect:
      return old_q + args.skip_weight * nn_out  # Should match in size at the end
    else:
      return nn_out

  residual_fun, dt, cell = get_residual_fun(args)
  radii_shape = cell.generate_random_radii().shape
  vmap_res = jax.vmap(residual_fun, in_axes=(0, (0, 0, None, (0, 0))))
  vmap_rad_to_ctrl = jax.vmap(cell.radii_to_ctrl)

  def metric_loss(old_q, old_p, radii, new_q, params):
    radii_reshaped = radii.reshape((-1, *radii_shape))
    ref_ctrls = vmap_rad_to_ctrl(radii_reshaped)
    resfun_args = (old_q, old_p, dt, (ref_ctrls, ref_ctrls))
    vec = predict(old_q, old_p, radii, params)
    
    residuals = vmap_res(vec, resfun_args)

    return np.sum(np.abs(vmap_res(vec, resfun_args))) / batch_size

  def mse_loss(old_q, old_p, radii, new_q, params):
    return np.mean(np.square(predict(old_q, old_p, radii, params) - new_q))

  if args.loss_type == 'metric':
    loss = metric_loss
  elif args.loss_type == 'mse':
    loss = mse_loss
  else:
    raise ValueError(f'Invalid loss type {args.loss_type}.')

  #opt_init, opt_update, get_params = optimizers.momentum(step_size, momentum_mass)
  opt_init, opt_update, get_params = optimizers.sgd(step_size)

  @jax.jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, jax.grad(loss, argnums=4)(*batch, params), opt_state)

  # TODO TODO TODO
  @jax.jit
  def newton_update(i, opt_state, batch):
    params = get_params(opt_state)

    
    descent_dir = jax.scipy.sparse.linalg.cg(jax.hessian(loss, argnums=4)(*batch, params),
                                             jax.grad(loss, argnums=4)(*batch, params))
    return opt_update(i, descent_dir, opt_state)

  jrng = jax.random.PRNGKey(seed)
  _, init_params = init_random_params(jrng, (-1, args.nfeat * 2 + args.nrad))
  opt_state = opt_init(init_params)

  print("\nStarting training...")
  start_t = time.time()
  train_losses = []
  test_losses = []

  full_train_losses = []
  full_test_losses = []

  iters = []
  for it in range(num_iterations):
    if it % 10 == 0:
      params = get_params(opt_state)

      if args.debug:
        train_loss = loss(*full_train(), params)
        test_loss = loss(*full_test(), params)
      else:
        train_loss = loss(*next(batches), params)
        test_loss = loss(*next(test_batches), params)

      train_losses.append((it, train_loss))
      test_losses.append((it, test_loss))
      iters.append(it)

      print(f'At iteration {it}')
      print("10 iterations in {:0.2f} sec".format(time.time() - start_t))
      start_t = time.time()
      print("Training set loss {}".format(train_loss))
      print("Test set loss {}".format(test_loss))

    if it % 100 == 0:
      params = get_params(opt_state)
      train_loss = loss(*full_train(), params)
      test_loss  = loss(*full_test(), params)

      full_train_losses.append((it, train_loss))
      full_test_losses.append((it, test_loss))

      print("Full training set loss {}".format(train_loss))
      print("Full test set loss {}".format(test_loss))

    if args.debug:
      opt_state = update(it, opt_state, full_train())
    else:
      opt_state = update(it, opt_state, next(batches))

  params = get_params(opt_state)
  train_loss = loss(*full_train(), params)
  test_loss  = loss(*full_test(), params)

  full_train_losses.append((num_iterations, train_loss))
  full_test_losses.append((num_iterations, test_loss))

  print("Full training set loss {}".format(train_loss))
  print("Full test set loss {}".format(test_loss))

  metrics = {
      'm_final_train_loss': train_loss,
      'm_final_test_loss': test_loss,
      'm_num_iters': num_iterations,
  }

  print(f'Saving progress and model...')
  with open(os.path.join(args.exp_dir, 'train_losses.pkl'), 'wb') as f:
    pickle.dump(train_losses, f)

  with open(os.path.join(args.exp_dir, 'test_losses.pkl'), 'wb') as f:
    pickle.dump(test_losses, f)

  with open(os.path.join(args.exp_dir, 'full_train_losses.pkl'), 'wb') as f:
    pickle.dump(full_train_losses, f)

  with open(os.path.join(args.exp_dir, 'full_test_losses.pkl'), 'wb') as f:
    pickle.dump(full_test_losses, f)

  with open(os.path.join(args.exp_dir, 'iters.pkl'), 'wb') as f:
    pickle.dump(iters, f)

  with open(os.path.join(args.exp_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(params, f)

  with open(os.path.join(args.exp_dir, 'metrics.pkl'), 'wb') as f:
    pickle.dump(metrics, f)


if __name__ == '__main__':
  main()
