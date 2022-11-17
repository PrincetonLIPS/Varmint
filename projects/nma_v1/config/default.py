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

    config.grid_str = "C0000 C0000 C0000 C0000 C0000\n"\
                      "C2000 C0000 C0000 C0000 C0040\n"\
                      "C2000 C0000 S0000 C0000 C0040\n"\
                      "C2000 C0000 C0000 C0000 C0040\n"\
                      "C0000 C0000 C0000 C0000 C0000\n"

    return config
