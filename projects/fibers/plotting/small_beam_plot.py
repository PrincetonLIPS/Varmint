import numpy as onp
import matplotlib.pyplot as plt

from varmint.utils.movie_utils import create_movie, create_static_image


def plot_small_beam(config, element, ref_ctrl, final_x_local, path):
    fig = plt.figure()
    ax = fig.gca()

    if ref_ctrl is not None:
        xlim, ylim = create_static_image(element, ref_ctrl, ax,
                                         boundary_path=6, alpha=0.5,
                                         facecolor='black', edgecolor=None)

    if final_x_local is not None:
        xlim, ylim = create_static_image(element, final_x_local, ax,
                                         boundary_path=6, alpha=1.0,
                                         facecolor='black', edgecolor=None)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    ax.set_xticks(onp.arange(int(xlim[0]), int(xlim[1] + 1.0), 1.0))
    ax.set_yticks(onp.arange(int(ylim[0]), int(ylim[1] + 1.0), 1.0))
    ax.set_xticks(onp.arange(int(xlim[0]), int(xlim[1] + 1.0), 0.5), minor=True)
    ax.set_yticks(onp.arange(int(ylim[0]), int(ylim[1] + 1.0), 0.5), minor=True)
    ax.grid(True, which='both', linewidth=1)
    ax.set_axisbelow(True)

    fig.savefig(path)
    plt.close(fig)
