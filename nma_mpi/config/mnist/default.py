from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    # TODO(doktay): There should be a way to access this programmatically...
    config.filename = '/n/fs/mm-iga/Varmint/nma_mpi/config/mnist.py'

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

    config.grid_str = "C0000 C0300 C0300 C0300 C0000\n"\
                      "C2000 C0000 C0000 C0000 C0040\n"\
                      "C2000 C0000 C0000 C0000 C0040\n"\
                      "C2000 C0000 C0000 C0000 C0040\n"\
                      "C0000 C0005 C0005 C0005 C0000\n"

    config.n_disps = 4
    def _get_increment_dict(disps):
        return {
            '99': np.array([0.0, 0.0]),
            '98': np.array([0.0, 0.0]),
            '97': np.array([0.0, 0.0]),
            '96': np.array([0.0, 0.0]),
            '2': np.array([-disps[0], 0.0]),
            '3': np.array([0.0, -disps[1]]),
            '4': np.array([-disps[2], 0.0]),
            '5': np.array([0.0, -disps[3]]),
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

    config.max_disp = 4.0
    config.n_layers = 2
    config.n_activations = 30

    config.num_examples = 1

    config.radii_range = [0.2, 0.8]

    config.save_every = 5
    config.eval_every = 50
    config.num_eval = 5

    config.lr = 0.1

    return config
