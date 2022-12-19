import varmint

import os

import jax
import jax.numpy as jnp
import numpy as onp

import haiku as hk


def get_config() -> varmint.config_dict.ConfigDict:
    config = varmint.config_dict.ConfigDict()

    config.filename = os.path.abspath(__file__)

    config.seed = None
    config.jax_seed = 22
    config.dataset_seed = 10

    config.ncp = 5
    config.radial_ncp = 5
    config.quad_deg = 5
    config.spline_deg = 2

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D

    config.solver_parameters = {
        'tol': 1e-8,
    }

    config.cell_length = 5
    config.grid_str = "C1000 C0000 C0000 C0000 C0000 C0000 C0020\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0020\n"\
                      "C1000 C0000 00000 00000 00000 C0000 C0020\n"\
                      "C1000 C0000 00000 00000 00000 C0000 C0020\n"\
                      "C1000 C0000 00000 00000 00000 C0000 C0020\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0020\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0020\n"

    # Counterclockwise starting from bottom left.
    # TODO(doktay): Should probably automatically generate this.
    config.internal_corners = onp.array([
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

    config.internal_radii = 7.0
    config.internal_radii_clip = [0.1, 0.9]
    config.normalized_init = True

    config.n_disps = 2
    def _get_increment_dict(disps):
        return {
            '1': jnp.array([disps[0], 0.0]),
            '2': jnp.array([-disps[1], 0.0]),
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

    config.shape_family = 'heart'
    config.shape_parameters = {
    }
    config.loss_type = 'mse'
    config.loss_norm = 1

    config.max_disp = 3.0
    config.n_layers = 2
    config.n_activations = 200

    config.radii_range = [0.1, 0.9]
    config.radii_smoothness_penalty = 0.0
    config.perturb_mesh = False
    config.mesh_perturb_range = [-0.45, 0.45]

    # Adjoint optimization parameters
    config.max_iter = 10000
    config.lr = 0.01
    config.geometry_lr_multiplier = 5.0
    config.freeze_radii = False
    config.freeze_nn = True
    config.freeze_nn_val = 2.0

    config.debug_single_shape = False
    config.num_ds_samples = 2

    config.save_every = 50
    config.eval_every = 50
    config.ewa_weight = 0.95
    config.num_eval = 2

    return config
