import os
import shutil
import argparse
import random
import json
import pickle

import surrogate_nns
import jax.numpy as np

from collections import namedtuple


def prepare_experiment_args(parser, exp_root):
  # Experiment organization parameters.
  parser.add_argument('-n', '--exp_name')
  parser.add_argument('--exp_root', default=exp_root)
  parser.add_argument('--seed', type=int, default=-1)
  parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing experiment directory.')


def prepare_experiment_directories(args):
  if not os.path.exists(args.exp_root):
    print(f'Creating experiment root directory {args.exp_root}')
    os.mkdir(args.exp_root)
  if not args.exp_name:
    args.exp_name = 'exp{}'.format(random.randint(100000, 999999))
  print(f'Launching experiment {args.exp_name}')

  # TODO(doktay): Figure out best way to random seed.
  if args.seed == -1:
    args.seed = None

  args.exp_dir = os.path.join(args.exp_root, args.exp_name)
  if args.overwrite and os.path.exists(args.exp_dir):
    print('Overwriting existing directory.')
    shutil.rmtree(args.exp_dir)
  assert not os.path.exists(args.exp_dir)
  print(f'Creating experiment with name {args.exp_name} in {args.exp_dir}')
  os.mkdir(args.exp_dir)


def save_args(args):
  cmd_path = os.path.join(args.exp_dir, 'args.txt')
  with open(cmd_path, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def load_args(exp_dir):
  cmd_path = os.path.join(exp_dir, 'args.txt')
  with open(cmd_path, 'r') as f:
    args = json.load(f)

  return args


SurrogateExperiment = namedtuple('SurrogateExperiment', [
    'args', 'train_losses', 'test_losses', 'iters', 'predict_fun'
])


def read_surrogate_experiment(ds_root, expname):
  expdir = os.path.join(ds_root, 'trained_surrogates', expname)
  args = load_args(os.path.join(ds_root, 'trained_surrogates', expname))
  with open(os.path.join(expdir, 'train_losses.pkl'), 'rb') as f:
    train_losses = pickle.load(f)
  with open(os.path.join(expdir, 'test_losses.pkl'), 'rb') as f:
    test_losses = pickle.load(f)
  with open(os.path.join(expdir, 'iters.pkl'), 'rb') as f:
    iters = pickle.load(f)
  with open(os.path.join(expdir, 'model.pkl'), 'rb') as f:
    params = pickle.load(f)

  _, nn_fun = surrogate_nns.get_tanh_net(args['nfeat'])

  def predict_fun(old_q, old_p, radii):
    inputs = np.concatenate((old_q, old_p, radii), axis=1)
    nn_out = nn_fun(params, inputs)

    return nn_out  # Should match in size at the end

  return SurrogateExperiment(
    args=args,
    train_losses=train_losses,
    test_losses=test_losses,
    iters=iters,
    predict_fun=predict_fun,
  )
