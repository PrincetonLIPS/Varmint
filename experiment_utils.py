import os
import shutil
import argparse
import random
import json


def prepare_experiment_args(parser, exp_root):
  # Experiment organization parameters.
  parser.add_argument('-n', '--exp_name')
  parser.add_argument('--exp_root', default=exp_root) # default='/n/fs/mm-iga/flow_elasticity/experiments')
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
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  cmd_path = os.path.join(exp_dir, 'args.txt')
  with open(cmd_path, 'r') as f:
    args.__dict__ = json.load(f)

  return args
