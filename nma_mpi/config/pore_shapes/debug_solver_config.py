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

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D
    config.E = 0.005

    config.solver_parameters = {
        'max_iter': 1000,
        'step_size': 1.0,
        'tol': 1e-8,
        'ls_backtrack': 0.95,
        'update_every': 10,
    }

    config.grid_str = "C0000 C0400 C0000 C0500 C0000 C0600 C0000\n"\
                      "C3000 C0000 C0000 C0000 C0000 C0000 C0070\n"\
                      "C0000 C0000 00000 00000 00000 C0000 C0000\n"\
                      "C2000 C0000 00000 00000 00000 C0000 C0080\n"\
                      "C0000 C0000 00000 00000 00000 C0000 C0000\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0090\n"\
                      "C0000 C000C C0000 C000B C0000 C000A C0000\n"

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
    ])

    config.cell_length = 5
    config.internal_radii = 0.8
    config.internal_radii_clip = [0.1, 0.9]
    config.normalized_init = False

    config.n_disps = 12
    def _get_increment_dict(disps):
        return {
            '99': jnp.array([0.0, 0.0]),
            #'98': jnp.array([0.0, 0.0]),
            #'97': jnp.array([0.0, 0.0]),
            '96': jnp.array([0.0, 0.0]),
            '1': jnp.array([-disps[0], 0.0]),
            '2': jnp.array([-disps[1], 0.0]),
            '3': jnp.array([-disps[2], 0.0]),
            '4': jnp.array([0.0, -disps[3]]),
            '5': jnp.array([0.0, -disps[4]]),
            '6': jnp.array([0.0, -disps[5]]),
            '7': jnp.array([-disps[6], 0.0]),
            '8': jnp.array([-disps[7], 0.0]),
            '9': jnp.array([-disps[8], 0.0]),
            'A': jnp.array([0.0, -disps[9]]),
            'B': jnp.array([0.0, -disps[10]]),
            'C': jnp.array([0.0, -disps[11]]),
        }
    config.get_increment_dict = _get_increment_dict

    def _get_nn_fn_old(max_disp, n_layers, n_activations, n_disps):
        def tanh_clip(x):
            return jnp.tanh(x) * max_disp
        def clip(x):
            return jnp.clip(x, -max_disp, max_disp)
        def nn_fn(input):
            mlp = hk.Sequential([
                hk.Linear(30), jax.nn.relu,
                hk.Linear(30), jax.nn.relu,
                hk.Linear(10), jax.nn.relu,
                hk.Linear(n_disps, with_bias=False),  tanh_clip,
            ])

            return mlp(input)
        return nn_fn

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
    config.loss_type = 'mse_reindex'
    config.loss_norm = 1

    config.max_disp = 3.0
    config.n_layers = 2
    config.n_activations = 200

    config.radii_range = [0.2, 0.8]
    config.radii_smoothness_penalty = 0.0
    config.perturb_mesh = True
    config.mesh_perturb_range = [-0.45, 0.45]

    # Adjoint optimization parameters
    config.lr = 0.0001
    config.geometry_lr_multiplier = 10.0
    config.freeze_radii = False
    config.freeze_nn = False
    config.freeze_nn_val = 0.0

    config.debug_single_shape = False
    config.num_ds_samples = 10

    config.save_every = 50
    config.eval_every = 50
    config.num_eval = 5

    return config
