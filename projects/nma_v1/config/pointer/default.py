from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    # TODO(doktay): There should be a way to access this programmatically...
    config.filename = '/n/fs/mm-iga/Varmint/nma_mpi/config/default.py'

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

    config.cell_length = 5
    config.grid_str = "C0000 C0000 C0000 C0000 C0000\n"\
                      "C1000 C0000 C0000 C0000 C0020\n"\
                      "C1000 C0000 S0000 C0000 C0020\n"\
                      "C1000 C0000 C0000 C0000 C0020\n"\
                      "C0000 C0000 C0000 C0000 C0000\n"

    config.n_disps = 2
    def _get_increment_dict(disps):
        return {
            '99': np.array([0.0, 0.0]),
            '98': np.array([0.0, 0.0]),
            '97': np.array([0.0, 0.0]),
            '96': np.array([0.0, 0.0]),
            '1': np.array([-disps[0], 0.0]),
            '2': np.array([-disps[1], 0.0]),
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

    config.pointer_loc = np.array([12.5, 12.5])

    config.max_disp = 4.0
    config.n_layers = 3
    config.n_activations = 30

    config.lr = 0.0001
    config.geometry_lr_multiplier = 1.0
    config.freeze_radii = False
    config.freeze_nn = False
    config.freeze_nn_val = 0.0

    config.save_every = 50
    config.eval_every = 50
    config.num_eval = 5

    return config
