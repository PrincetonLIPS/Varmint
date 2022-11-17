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
    config.quad_deg = 5
    config.spline_deg = 2

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D
    config.E = 0.005

    config.solver_parameters = {
        'max_iter': 1000,
        'step_size': 1.0,
        'tol': 1e-8,
        'ls_backtrack': 0.95,
        'update_every': 10,
    }

    config.grid_str = "C0000 C0500 C0000 C0600 C0000 C0700 C0000 C0800 C0000\n"\
                      "C4000 C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0090\n"\
                      "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                      "C3000 C0000 C0000 00000 00000 00000 C0000 C0000 C00A0\n"\
                      "C0000 C0000 C0000 00000 00000 00000 C0000 C0000 C0000\n"\
                      "C2000 C0000 C0000 00000 00000 00000 C0000 C0000 C00B0\n"\
                      "C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0000 C0000 C00C0\n"\
                      "C0000 C000G C0000 C000F C0000 C000E C0000 C000D C0000\n"

    # Counterclockwise starting from bottom left.
    # TODO(doktay): Should probably automatically generate this.
    config.internal_corners = np.array([
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 5],
        [4, 5],
        [5, 5],
        [5, 4],
        [5, 3],
        [5, 2],
        [4, 2],
        [3, 2],
    ]) + 1

    config.cell_length = 5
    config.internal_radii = 6.0

    config.n_disps = 16
    def _get_increment_dict(disps):
        return {
            '99': jnp.array([0.0, 0.0]),
            '98': jnp.array([0.0, 0.0]),
            '97': jnp.array([0.0, 0.0]),
            '96': jnp.array([0.0, 0.0]),
            '1': jnp.array([-disps[0], 0.0]),
            '2': jnp.array([-disps[1], 0.0]),
            '3': jnp.array([-disps[2], 0.0]),
            '4': jnp.array([-disps[3], 0.0]),
            '5': jnp.array([0.0, -disps[4]]),
            '6': jnp.array([0.0, -disps[5]]),
            '7': jnp.array([0.0, -disps[6]]),
            '8': jnp.array([0.0, -disps[7]]),
            '9': jnp.array([-disps[8], 0.0]),
            'A': jnp.array([-disps[9], 0.0]),
            'B': jnp.array([-disps[10], 0.0]),
            'C': jnp.array([-disps[11], 0.0]),
            'D': jnp.array([0.0, -disps[12]]),
            'E': jnp.array([0.0, -disps[13]]),
            'F': jnp.array([0.0, -disps[14]]),
            'G': jnp.array([0.0, -disps[15]]),
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

    config.shape_family = 'rff'
    config.shape_parameters = {
        'lengthscale': 0.8,
        'num_feats': 100,
    }
    config.loss_type = 'mse_rotation'

    config.max_disp = 3.0
    config.n_layers = 2
    config.n_activations = 200

    config.radii_range = [0.2, 0.8]

    # Adjoint optimization parameters
    config.lr = 0.0001
    config.freeze_radii = False
    config.freeze_nn = False
    config.freeze_nn_val = 0.0

    config.debug_single_shape = False
    config.num_ds_samples = 10

    config.save_every = 5
    config.eval_every = 50
    config.num_eval = 5

    return config
