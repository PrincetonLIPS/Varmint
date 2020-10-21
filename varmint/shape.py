import jax
import jax.numpy         as np
import matplotlib.pyplot as plt

from collections import namedtuple

import bsplines

Patch2D = namedtuple(
  'Patch2D',
  [
    'ctrl',
    'xknots',
    'yknots',
    'deg',
  ],
)

class Shape2D:
  ''' Class for managing collections of 2D patches

  '''

  def __init__(self, *patches):
    self.patches = patches

    # This is where you'd find all matching pairs of control points.

  def render(self, filename=None):
    fig = plt.figure()
    ax  = plt.axes()
    ax.set_aspect('equal')

    uu = np.linspace(0, 1, 1002)[1:-1]

    for patch in self.patches:
      for jj in range(patch.ctrl.shape[1]):
        xx = bsplines.bspline1d(
          uu,
          patch.ctrl[:,jj,:],
          patch.xknots,
          patch.deg,
        )
        ax.plot(xx[:,0], xx[:,1], 'k-')
      for ii in range(patch.ctrl.shape[0]):
        yy = bsplines.bspline1d(
          uu,
          patch.ctrl[ii,:,:],
          patch.yknots,
          patch.deg,
        )
        ax.plot(yy[:,0], yy[:,1], 'k-')
    if filename is None:
      plt.show()
    else:
      plt.savefig(filename)


def main():

  # Create a rectangle.
  r1_deg    = 4
  r1_ctrl   = bsplines.mesh(np.arange(10), np.arange(5))
  r1_xknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[0])
  r1_yknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[1])
  r1_patch  = Patch2D(r1_ctrl, r1_xknots, r1_yknots, r1_deg)
  print(r1_xknots, r1_yknots)

  # Create another rectangle.
  r2_deg    = 3
  r2_ctrl   = bsplines.mesh(np.array([3,4,5,6]), np.array([-4, -3, -2, -1, 0]))
  r2_xknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[0])
  r2_yknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[1])
  r2_patch  = Patch2D(r2_ctrl, r2_xknots, r2_yknots, r2_deg)

  # Bend a u-shaped thing around the top.
  u1_deg  = 2
  band    = np.array([-4.5, -3.5, -2.5])
  center  = np.array([4.5, 4])
  u1_ctrl = np.zeros((3,8,2))
  for ii, theta in enumerate(np.linspace(-np.pi, 0, 8)):
    u1_ctrl = jax.ops.index_update(u1_ctrl,
                                   jax.ops.index[:,ii,0],
                                   band * np.cos(theta))
    u1_ctrl = jax.ops.index_update(u1_ctrl,
                                   jax.ops.index[:,ii,1],
                                   band * np.sin(theta))
  u1_ctrl   = u1_ctrl + center
  u1_xknots = bsplines.default_knots(u1_deg, u1_ctrl.shape[0])
  u1_yknots = bsplines.default_knots(u1_deg, u1_ctrl.shape[1])
  u1_patch  = Patch2D(u1_ctrl, u1_xknots, u1_yknots, u1_deg)

  shape = Shape2D(r1_patch, r2_patch, u1_patch)
  shape.render()

if __name__ == '__main__':
  main()
