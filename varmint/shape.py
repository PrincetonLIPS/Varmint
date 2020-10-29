import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from exceptions import (
  DimensionError,
  LabelError,
  )

import bsplines

from patch import Patch2D

class Shape2D:
  ''' Class for managing collections of 2D patches.

  The objective of this class is to abstract out the parameterizations of the
  control points within the patches and resolve (simple) constraints between
  the patches.  Constraints are specified via having identical labels attached
  to control points.
  '''

  def __init__(self, *patches):
    ''' Constructor of a collection of two-dimensional patches.

    Parameters:
    -----------
     - *patches: A variable number of Patch2D instances.

    '''
    for patch in patches:
      assert isinstance(patch, Patch2D), \
        "Argument is a %r, but needs to be a Patch2D." % (type(patch))

    self.patches = patches

  def render(self, filename=None):
    fig = plt.figure()
    ax  = plt.axes()
    ax.set_aspect('equal')

    uu = np.linspace(0, 1, 1002)[1:-1]

    rendered_labels = set()

    for patch in self.patches:

      # Plot vertical lines.
      for jj in range(patch.ctrl.shape[1]):
        xx = bsplines.bspline1d(
          uu,
          patch.ctrl[:,jj,:],
          patch.xknots,
          patch.deg,
        )
        ax.plot(xx[:,0], xx[:,1], 'k-')

      # Plot horizontal lines.
      for ii in range(patch.ctrl.shape[0]):
        yy = bsplines.bspline1d(
          uu,
          patch.ctrl[ii,:,:],
          patch.yknots,
          patch.deg,
        )
        ax.plot(yy[:,0], yy[:,1], 'k-')

      # Plot the control points themselves.
      ax.plot(patch.ctrl[:,:,0].ravel(), patch.ctrl[:,:,1].ravel(), 'k.')

      # Plot labels.
      label_r, label_c = onp.where(patch.labels)
      for ii in range(len(label_r)):
        row = label_r[ii]
        col = label_c[ii]
        text = patch.labels[row,col]
        if text not in rendered_labels:
          rendered_labels.add(text)
        else:
          continue
        ax.annotate(text, patch.ctrl[row,col,:])


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
  r1_patch.labels[3:7,0]  = ['A', 'B', 'C', 'D']
  r1_patch.labels[:3,-1]  = ['E', 'F', 'G']
  r1_patch.labels[-3:,-1] = ['H', 'I', 'J']

  # Create another rectangle.
  r2_deg    = 3
  r2_ctrl   = bsplines.mesh(np.array([3,4,5,6]), np.array([-4, -3, -2, -1, 0]))
  r2_xknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[0])
  r2_yknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[1])
  r2_patch  = Patch2D(r2_ctrl, r2_xknots, r2_yknots, r2_deg)
  r2_patch.labels[:,-1] = ['A', 'B', 'C', 'D']

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
  u1_patch.labels[:,0]  = ['E', 'F', 'G']
  u1_patch.labels[:,-1] = ['H', 'I', 'J']

  shape = Shape2D(r1_patch, r2_patch, u1_patch)
  shape.render()

if __name__ == '__main__':
  main()
