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

    config.ncp = 5
    config.radial_ncp = 5
    config.quad_deg = 5
    config.spline_deg = 2

    config.cell_size = 6.0
    config.border_size = 3.0

    config.mat_model = 'NeoHookean2D'  # Choose between LinearElastic2D and NeoHookean2D

    config.solver_parameters = {
        'tol': 1e-8,
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

    def _get_nn_fn(max_disp, n_disps):
        def tanh_clip(x):
            return jnp.tanh(x) * max_disp
        def get_max(x):
            return (x > 0.5).astype(jnp.float64)
        def hk_print(x):
            print(x)
            return x

        def nn_fn(x):
            x = x.astype(jnp.float64) / 255.
            mlp = hk.Sequential([
              hk.Flatten(),
              hk.Linear(300), jax.nn.relu,
              hk.Linear(100), jax.nn.relu,
              hk.Linear(100), jax.nn.relu,
              hk.Linear(6), tanh_clip,
            ])
            return mlp(x)

        return nn_fn

    config.get_nn_fn = _get_nn_fn
    config.max_disp = 3.0
    config.freeze_pretrain = False
    config.freeze_colors = False

    config.nn_checkpoint = '/n/fs/mm-iga/Varmint_postnmav1/nma_mpi/notebooks/mnist_lenet_weights_2.pkl'
    config.material_checkpoint = '/n/fs/mm-iga/Varmint_postnmav1/nma_mpi/experiments/digital_mnist_experiments/digital_mnist_second_try_10digits_temp10_freezeradii/sim-digital_mnist_second_try_10digits_temp10_freezeradii-params-550.pkl'

    config.softmax_temp = 10.0

    config.radii_range = [0.3, 0.7]
    config.radii_smoothness_penalty = 0.0

    # Adjoint optimization parameters
    config.lr = 0.0001
    config.geometry_lr_multiplier = 1.0
    config.freeze_radii = True
    config.freeze_nn = False
    config.freeze_nn_val = 2.0
    config.init_from_ckpt = False

    config.rand_digits = False

    config.debug_single_shape = False
    config.num_digits = 10

    config.save_every = 50
    config.eval_every = 50
    config.num_trials = 5

    return config
