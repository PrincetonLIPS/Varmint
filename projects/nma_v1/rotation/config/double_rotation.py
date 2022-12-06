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

    config.ncp = 5
    config.quad_deg = 5
    config.spline_deg = 2

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D

    config.solver_parameters = {
        'tol': 1e-8,
    }

    config.cell_length = 5
    config.grid_str = "C0000 C0000 C0200 C0200 C0200 C0000 C0000\n"\
                      "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0040\n"\
                      "C1000 C0000 C0000 S0000 C0000 C0000 C0040\n"\
                      "C1000 C0000 C0000 C0000 C0000 C0000 C0040\n"\
                      "C0000 C0000 C0000 C0000 C0000 C0000 C0000\n"\
                      "C0000 C0000 C0003 C0003 C0003 C0000 C0000\n"

    def _get_perturb_bounds(init_mesh_perturb):
        min_pert = -5.0*0.2*onp.ones_like(init_mesh_perturb)
        max_pert =  5.0*0.2*onp.ones_like(init_mesh_perturb)
        min_pert[2][2] = 0.
        min_pert[2][3] = 0.
        min_pert[3][2] = 0.
        min_pert[3][3] = 0.
        max_pert[2][2] = 0.
        max_pert[2][3] = 0.
        max_pert[3][2] = 0.
        max_pert[3][3] = 0.

        return min_pert, max_pert
    config.get_perturb_bounds = _get_perturb_bounds

    config.left_point   = jnp.array([15.0, 17.5])
    config.right_point  = jnp.array([20.0, 17.5])
    config.center_point = jnp.array([17.5, 17.5])
    config.angle_range = (-onp.pi / 6.0, onp.pi / 6.0)

    config.n_disps = 2
    def _get_increment_dict(disps):
        return {
            '99': jnp.array([0.0, 0.0]),
            '98': jnp.array([0.0, 0.0]),
            '97': jnp.array([0.0, 0.0]),
            '96': jnp.array([0.0, 0.0]),
            '1': jnp.array([-disps[0], 0.0]),
            '2': jnp.array([0.0, -disps[1]]),
            '3': jnp.array([0.0, disps[1]]),
            '4': jnp.array([disps[0], 0.0]),
        }
    config.get_increment_dict = _get_increment_dict

    def _get_nn_fn(max_disp, n_layers, n_activations, n_disps):
        # Clip max displacement using tanh.
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
    config.radii_range = [0.1, 0.9]
    config.perturb_mesh = True

    config.n_layers = 3
    config.n_activations = 30

    config.lr = 0.001
    config.max_iter = 10000

    config.save_every = 50
    config.eval_every = 50
    config.ewa_weight = 0.95

    return config
