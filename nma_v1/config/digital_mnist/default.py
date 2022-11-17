import os

import jax

import numpy as np
import jax.numpy as jnp

import haiku as hk

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.filename = os.path.abspath(__file__)

    config.seed = None

    config.ncp = 5
    config.radial_ncp = 5
    config.quad_deg = 5
    config.spline_deg = 2

    config.cell_size = 6.0
    config.border_size = 3.0

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D
    config.E = 0.005

    config.solver_parameters = {
        'max_iter': 1000,
        'step_size': 1.0,
        'tol': 1e-8,
        'ls_backtrack': 0.95,
        'update_every': 10,
    }

    config.n_disps = 6
    def _get_increment_dict(disps):
        return {
            '99': jnp.array([0.0, 0.0]),
            '98': jnp.array([0.0, 0.0]),
            '97': jnp.array([0.0, 0.0]),
            '96': jnp.array([0.0, 0.0]),
            '1': jnp.array([disps[0], 0.0]),
            '2': jnp.array([disps[1], 0.0]),
            '3': jnp.array([0.0, -disps[2]]),
            '4': jnp.array([-disps[3], 0.0]),
            '5': jnp.array([-disps[4], 0.0]),
            '6': jnp.array([0.0, disps[5]]),
        }
    config.get_increment_dict = _get_increment_dict

    def _get_nn_fn(max_disp, n_layers, n_activations, n_disps):
        def tanh_clip(x):
            return jnp.tanh(x) * max_disp
        def nn_fn(input):
            layers = []
            for _ in range(n_layers):
                layers.extend([hk.Linear(n_activations), jax.nn.relu])
            layers.extend([
                hk.Linear(n_disps, with_bias=False), tanh_clip,
            ])
            mlp = hk.Sequential(layers)

            return mlp(input)
        return nn_fn

    config.get_nn_fn = _get_nn_fn

    config.max_disp = 3.0
    config.n_layers = 2
    config.n_activations = 200

    config.softmax_temp = 10.0

    config.radii_range = [0.1, 0.9]
    config.radii_smoothness_penalty = 0.0

    # Adjoint optimization parameters
    config.lr = 0.01
    config.geometry_lr_multiplier = 1.0
    config.freeze_radii = False
    config.freeze_nn = False
    config.freeze_nn_val = 2.0

    config.rand_digits = False

    config.debug_single_shape = False
    config.num_digits = 10

    config.save_every = 50
    config.eval_every = 50
    config.num_eval = 5

    return config
