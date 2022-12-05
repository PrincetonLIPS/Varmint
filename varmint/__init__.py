from absl import app
from absl import flags

import os
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax.config import config
config.update("jax_enable_x64", True)

from ml_collections import config_flags
from ml_collections import config_dict
from mpi4py import MPI

from varmint.utils.experiment_utils import *
