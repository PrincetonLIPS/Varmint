from absl import flags
from absl import logging

import os
import re
import shutil
import random
import json
import pickle

import pandas as pd

import jax.numpy as np
import numpy.random as npr

from collections import namedtuple
from varmint.utils.mpi_utils import *

import varmint.utils.jaxboard as jaxboard

SLURM_ROOT = '/n/fs/mm-iga/Varmint/slurm_experiments/'


def initialize_experiment(verbose=False):
    # Weirdly the log level gets set to INFO with absl.
    import matplotlib.animation
    matplotlib.animation._log.setLevel('WARN')
    args = flags.FLAGS

    comm = MPI.COMM_WORLD
    if verbose:
        rprint(f'Initializing MPI with JAX.', comm=comm)
    local_rank = find_local_rank(comm)
    dev_id = local_rank % len(jax.devices())

    if verbose and local_rank == 0:
        print(f'Node {MPI.Get_processor_name()} reporting with {len(jax.devices())} devices: {jax.devices()}', flush=True)

    prepare_experiment_directories(args, comm)
    # args.seed and args.exp_dir should be set.

    config = args.config
    save_args(args, comm)

    if 'seed' in config:
        npr.seed(config.seed)

    if comm.rank == 0:
        logdir = args.exp_dir
        summary_writer = jaxboard.SummaryWriter(logdir)

    return args, dev_id, local_rank


def prepare_experiment_args(parser, exp_root, source_root):
    # Experiment organization parameters.
    flags.DEFINE_string('exp_name', None, 'Name of experiment', short_name='n')
    flags.DEFINE_string('exp_root', exp_root, 'Root Directory')
    flags.DEFINE_string('source_root', source_root, 'Source Directory')
    flags.DEFINE_string('exp_dir', '', 'Directory of experiment (set automatically)')
    flags.DEFINE_boolean('reload', False, 'If True, use existing directory without overwriting. Takes precedence over --overwrite.')
    flags.DEFINE_integer('load_iter', -1, 'Which iteration to reload from.')
    flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite an existing experiment')


def prepare_experiment_directories(args, comm):
    if comm.rank == 0:
        if not os.path.exists(args.exp_root):
            rprint(f'Creating experiment root directory {args.exp_root}', comm=comm)
            os.mkdir(args.exp_root)
        if not args.exp_name:
            args.exp_name = 'exp{}'.format(random.randint(100000, 999999))
    else:
        args.exp_name = None
    args.exp_name = comm.bcast(args.exp_name, root=0)
    rprint(f'Launching experiment {args.exp_name}', comm=comm)

    args.exp_dir = os.path.join(args.exp_root, args.exp_name)
    if comm.rank == 0:
        if args.reload and not os.path.exists(args.exp_dir):
            rprint(f'Trying to reload, but experiment at {args.exp_dir} does not exist.', comm=comm)
        elif args.overwrite and os.path.exists(args.exp_dir):
            rprint('Overwriting existing directory.', comm=comm)
            shutil.rmtree(args.exp_dir)
    comm.Barrier()
    if not os.path.exists(args.exp_dir):
        if comm.rank == 0:
            rprint(f'Creating experiment with name {args.exp_name} in {args.exp_dir}', comm=comm)
            os.mkdir(args.exp_dir)
    else:
        rprint(f'Reloading experiment with name {args.exp_name} in {args.exp_dir}', comm=comm)


def save_args(args, comm):
    if comm.rank == 0:
        if 'filename' in args.config:
            config_path = os.path.join(args.exp_dir, 'config.py')
            shutil.copyfile(args.config.filename, config_path)
        with open(os.path.join(args.exp_dir, 'flags.json'), 'w') as f:
            json.dump(args.config.to_json_best_effort(), f)


def load_args(exp_dir, comm):
    # TODO
    pass
    #cmd_path = os.path.join(exp_dir, 'args.txt')
    #with open(cmd_path, 'r') as f:
    #    args = json.load(f)

    #return args


def gather_experiment_df(exp_root, pattern, success_file=None):
    # Gather all experiments in exp_root matching pattern, which is a regex string.
    # If success file is not None, check the directory if the file exists.
    # If so, consider the experiment a success.

    p = re.compile(pattern)
    all_files = os.listdir(exp_root)
    matching_files = [f for f in all_files if p.match(f)]

    all_args = []
    for f in matching_files:
        exp_dir = os.path.join(exp_root, f)
        args = load_args(exp_dir)
        args['exp_name'] = f

        if success_file:
            if os.path.exists(os.path.join(exp_dir, success_file)):
                args['SUCCESS'] = True

                with open(os.path.join(exp_dir, 'metrics.pkl'), 'rb') as f:
                    metrics = pickle.load(f)
                    args.update(metrics)
            else:
                args['SUCCESS'] = False

        all_args.append(args)

    return pd.DataFrame(all_args)
