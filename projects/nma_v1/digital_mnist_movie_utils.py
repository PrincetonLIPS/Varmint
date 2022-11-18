import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from operator import itemgetter
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from varmint.geometry.bsplines import bspline2d
import varmint.geometry.bsplines as bsplines

import time

from varmint.geometry.elements import Element
import digital_mnist_patches


def create_movie_mnist(
    config,
    element: Element,
    ctrl_seq,
    filename,
    color_params,
    color_eval_fn,
    fig_kwargs={},
    comet_exp=None,
):
    t0 = time.time()

    print('\tDetermining bounds.')
    # Get extrema of control points.
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for ctrl in ctrl_seq:
        min_x = float(np.minimum(np.min(ctrl[..., 0]), min_x))
        max_x = float(np.maximum(np.max(ctrl[..., 0]), max_x))
        min_y = float(np.minimum(np.min(ctrl[..., 1]), min_y))
        max_y = float(np.maximum(np.max(ctrl[..., 1]), max_y))

    # Pad each end by 10%.
    pad_x = 0.1 * (max_x - min_x)
    pad_y = 0.1 * (max_y - min_y)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    print('\tSetting up figure.')
    # Set up the figure and axes.
    fig = plt.figure(**fig_kwargs)
    ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
    fig.set_size_inches(10, 10)
    ax.set_aspect('equal')

    # Things we need to both initialize and update.
    objects = {}
    path = element.get_boundary_path()
    jit_map_fn = jax.jit(element.get_map_fn(path))

    color_params = jax.nn.sigmoid(color_params)

    def init():
        # Render the first time step.
        color_locs = color_eval_fn(ctrl_seq[0])
        colors = color_eval_fn(color_params)

        for i, patch_ctrl in enumerate(ctrl_seq[0]):
            locs = jit_map_fn(patch_ctrl)
            # line, = ax.plot(locs[:,0], locs[:,1], 'b-')
            poly, = ax.fill(locs[:, 0], locs[:, 1],
                            facecolor='lightsalmon',
                            edgecolor='orangered',
                            linewidth=1)
            objects[(i, 'p')] = poly
            objects[(i, 'p')].set_visible(False)

        scat = ax.scatter(color_locs[..., 0], color_locs[..., 1], c=colors, s=7)
        digital_mnist_patches.draw_mpl_patches(ax, config.cell_size, config.border_size, alpha=1.0)

        objects[(0, 'ip')] = scat
        objects[(0, 'ip')].set_visible(False)

        return objects.values()

    def update(tt):
        color_locs = color_eval_fn(ctrl_seq[tt])
        colors = color_eval_fn(color_params)

        objects[(0, 'ip')].set_offsets(color_locs.reshape((-1, 2)))
        objects[(0, 'ip')].set_visible(True)

        for i, patch_ctrl in enumerate(ctrl_seq[tt]):
            locs = jit_map_fn(patch_ctrl)
            if tt == 0:
                objects[(i, 'p')].set_visible(True)
            objects[(i, 'p')].set_xy(locs)

        return objects.values()

    print('\tAnimating..')
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(ctrl_seq),
        interval=100,
        blit=True,
    )
    anim.save(filename)

    if comet_exp is not None:
        comet_exp.log_figure(figure_name='movie')

    plt.close(fig)
    t1 = time.time()
    print(f'Generated movie with {len(ctrl_seq)} frames and '
          f'{len(ctrl_seq[0])} patches in {t1-t0} seconds.')
