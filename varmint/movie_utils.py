import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from operator import itemgetter
from matplotlib.animation import FuncAnimation

from .patch2d import Patch2D
from .bsplines import bspline2d

import time


def create_movie(
    patch,
    ctrl_seq,
    filename,
    fig_kwargs={},
):
  t0 = time.time()

  # Get extrema of control points.
  min_x = np.inf
  max_x = -np.inf
  min_y = np.inf
  max_y = -np.inf
  for ctrl in ctrl_seq:
    for patch_ctrl in ctrl:
      min_x = float(np.minimum(np.min(patch_ctrl[:,:,0]), min_x))
      max_x = float(np.maximum(np.max(patch_ctrl[:,:,0]), max_x))
      min_y = float(np.minimum(np.min(patch_ctrl[:,:,1]), min_y))
      max_y = float(np.maximum(np.max(patch_ctrl[:,:,1]), max_y))

  # Pad each end by 10%.
  pad_x = 0.1 * (max_x - min_x)
  pad_y = 0.1 * (max_y - min_y)
  min_x -= pad_x
  max_x += pad_x
  min_y -= pad_y
  max_y += pad_y

  # Set up the figure and axes.
  fig = plt.figure(**fig_kwargs)
  ax  = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
  ax.set_aspect('equal')

  # Things we need to both initialize and update.
  objects = {}
  N  = 100
  uu = np.linspace(1e-6, 1-1e-6, N)
  path = np.hstack([
    np.vstack([uu[0]*np.ones(N), uu]),
    np.vstack([uu, uu[-1]*np.ones(N)]),
    np.vstack([uu[-1]*np.ones(N), uu[::-1]]),
    np.vstack([uu[::-1], uu[0]*np.ones(N)]),
  ]).T

  print('Compiling bspline code for movie exporting.')
  jit_bspline2d = jax.jit(bspline2d, static_argnums=(4,))
  print('Done.')

  def init():
    # Render the first time step.
    for i, patch_ctrl in enumerate(ctrl_seq[0]):
      locs = jit_bspline2d(
        path,
        patch_ctrl,
        patch.xknots,
        patch.yknots,
        patch.spline_deg,
      )
      # line, = ax.plot(locs[:,0], locs[:,1], 'b-')
      poly, = ax.fill(locs[:,0], locs[:,1],
                      facecolor='lightsalmon',
                      edgecolor='orangered',
                      linewidth=1)
      objects[(i, 'p')] = poly
      objects[(i, 'p')].set_visible(False)

    return objects.values()

  def update(tt):
    for i, patch_ctrl in enumerate(ctrl_seq[tt]):
      locs = jit_bspline2d(
        path,
        patch_ctrl,
        patch.xknots,
        patch.yknots,
        patch.spline_deg,
      )
      if tt == 0:
        objects[(i, 'p')].set_visible(True)
      objects[(i, 'p')].set_xy(locs)

    return objects.values()

  anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(ctrl_seq),
    interval=100,
    blit=True,
  )
  anim.save(filename)

  plt.close(fig)
  t1 = time.time()
  print(f'Generated movie with {len(ctrl_seq)} frames and '
        f'{len(ctrl_seq[0])} patches in {t1-t0} seconds.')

def create_static_image(
    patch,
    ctrl_sol,
    filename,
    just_cp=False,
    fig_kwargs={},
):
  t0 = time.time()

  # Get extrema of control points.
  min_x = float(onp.min(ctrl_sol[..., 0]))
  max_x = float(onp.max(ctrl_sol[..., 0]))

  min_y = float(onp.min(ctrl_sol[..., 1]))
  max_y = float(onp.max(ctrl_sol[..., 1]))

  # Pad each end by 10%.
  pad_x = 0.1 * (max_x - min_x)
  pad_y = 0.1 * (max_y - min_y)
  min_x -= pad_x
  max_x += pad_x
  min_y -= pad_y
  max_y += pad_y

  # Set up the figure and axes.
  fig = plt.figure(**fig_kwargs)
  ax  = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
  ax.set_aspect('equal')

  if just_cp:
    flat_cp = ctrl_sol.reshape(-1, 2)
    ax.scatter(flat_cp[:, 0], flat_cp[:, 1], s=10)
  else:
    # Things we need to both initialize and update.
    objects = {}
    N  = 100
    uu = np.linspace(1e-6, 1-1e-6, N)
    path = np.hstack([
      np.vstack([uu[0]*np.ones(N), uu]),
      np.vstack([uu, uu[-1]*np.ones(N)]),
      np.vstack([uu[-1]*np.ones(N), uu[::-1]]),
      np.vstack([uu[::-1], uu[0]*np.ones(N)]),
    ]).T

    jit_bspline2d = jax.jit(bspline2d, static_argnums=(4,))

    # Render the first time step.
    for patch_ctrl in ctrl_sol:
      locs = jit_bspline2d(
        path,
        patch_ctrl,
        patch.xknots,
        patch.yknots,
        patch.spline_deg,
      )
      # line, = ax.plot(locs[:,0], locs[:,1], 'b-')
      poly, = ax.fill(locs[:,0], locs[:,1],
                      facecolor='lightsalmon',
                      edgecolor='orangered',
                      linewidth=1)

  plt.savefig(filename)
  plt.close(fig)
  t1 = time.time()
  print(f'Generated image with {len(ctrl_sol)} patches in {t1-t0} seconds.')
