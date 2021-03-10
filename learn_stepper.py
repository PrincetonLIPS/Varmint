import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr

from varmint.cell2d import Cell2D
from varmint.materials import WigglyMat

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
parser.add_argument('--dsroot', default='/n/fs/mm-iga/Varmint/slurm_experiments/newdataset')
parser.add_argument('--stepsize', type=float, default=0.01)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--numiters', type=int, default=10000)


if __name__ == '__main__':
  args = parser.parse_args()
  args.exp_root = os.path.join(args.dsroot, 'trained_surrogates')
  print(f'Experiment directory created under dataset at {args.dsroot}')
  eutils.prepare_experiment_directories(args)
  # args.seed and args.exp_dir should be set.

  npr.seed(args.seed)
  jrng = jax.random.PRNGKey(args.seed)

  train_oq, train_op, train_rd, train_nq, test_oq, test_op, test_rd, test_nq = \
      dataset.parse_tensors_with_cache(args.dsroot)

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

  batches = data_stream()

  init_random_params, nn_fun = surrogate_nns.get_tanh_net(args.nfeat)

  def predict(old_q, old_p, radii, params):
    inputs = np.concatenate((old_q, old_p, radii), axis=1)
    nn_out = nn_fun(params, inputs)

    return nn_out  # Should match in size at the end

  def loss(old_q, old_p, radii, new_q, params):
    return np.mean(np.square(predict(old_q, old_p, radii, params) - new_q))

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jax.jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, jax.grad(loss, argnums=4)(*batch, params), opt_state)

  jrng = jax.random.PRNGKey(seed)
  _, init_params = init_random_params(jrng, (-1, args.nfeat * 2 + args.nrad))
  opt_state = opt_init(init_params)

  print("\nStarting training...")
  start_t = time.time()
  train_losses = []
  test_losses = []
  iters = []
  for it in range(num_iterations):
    opt_state = update(it, opt_state, next(batches))

    if it % 100 == 0:
      params = get_params(opt_state)
      train_loss = loss(train_oq, train_op, train_rd, train_nq, params)
      test_loss = loss(test_oq, test_op, test_rd, test_nq, params)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      iters.append(it)

      print(f'At iteration {it}')
      print("100 iterations in {:0.2f} sec".format(time.time() - start_t))
      start_t = time.time()
      print("Training set loss {}".format(train_loss))
      print("Test set loss {}".format(test_loss))

  print(f'Saving progress and model...')
  with open(os.path.join(args.exp_dir, 'train_losses.pkl'), 'wb') as f:
    pickle.dump(train_losses, f)

  with open(os.path.join(args.exp_dir, 'test_losses.pkl'), 'wb') as f:
    pickle.dump(test_losses, f)

  with open(os.path.join(args.exp_dir, 'iters.pkl'), 'wb') as f:
    pickle.dump(iters, f)

  with open(os.path.join(args.exp_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(params, f)
