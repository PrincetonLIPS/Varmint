import jax

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


def visualize_domain(config, step, domain, geometry_params, path, num_pts=1000):
    xx = onp.linspace(0, config.len_x, num_pts)
    yy = onp.linspace(0, config.len_y, num_pts)

    pts = onp.stack(onp.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
    vals = jax.jit(jax.vmap(domain, in_axes=(None, 0)))(geometry_params, pts)
    in_domain = vals < 0

    fig = plt.figure()
    ax = fig.gca()

    ax.scatter(pts[in_domain, 0], pts[in_domain, 1], c='black', s=1)
    ax.set_aspect('equal')

    fig.savefig(path)

    plt.figure(fig.number)
    config.summary_writer.plot('Implicit Domain', plt, step=step)

    plt.close(fig)


def visualize_pixel_domain(config, step, occupied_pixels, path):
    fig = plt.figure()
    ax = fig.gca()

    occupied_pixels = occupied_pixels.reshape(config.fidelity, config.fidelity).T
    occupied_pixels = occupied_pixels[::-1, :]
    ax.imshow(1-occupied_pixels, cmap='gray', extent=(0.0, config.len_x, 0.0, config.len_y))

    fig.savefig(path)

    plt.figure(fig.number)
    config.summary_writer.plot('Pixelized Domain', plt, step=step)

    plt.close(fig)
