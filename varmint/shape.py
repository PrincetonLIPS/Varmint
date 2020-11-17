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

  def patch_indices(self):
    start_idx = 0
    patch_indices = {}
    for patch in patches:
      shape   = patch.get_ctrl_shape()
      size    = onp.prod(shape)
      indices = start_idx + onp.arange(size)
      patch_indices[patch] = onp.reshape(indices, shape)
      start_idx += size
    return patch_indices

  def get_coincidences(self):
    ''' Compute the coincidence constraints. '''

    # Gather all coincidences, label-wise.
    all_coincidence = {}
    for patch in self.patches:

      # TODO: This needs to be guaranteed to be deterministic.
      for label, indices in patch.get_labels():
        if label not in all_coincidence:
          all_coincidence[label] = []
        all_coincidence[label].append((patch, indices))

    return all_coincidences

  def flatten_matrix(self):
    ''' The matrix that reparameterizes to remove linear constraints. '''
    num_unflat = 0
    for patch in self.patches:
      num_unflat += onp.prod(patch.get_ctrl_shape())

    # Start with a big identity matrix.
    matrix = onp.eye(num_unflat)

    # Accumulate a list of rows to remove.
    # Can't do it incrementally without breaking the indexing.
    rows_to_delete = []

    # Loop over coincidence constraints.
    for label, entries in self.get_coincidences().items():

      # Delete all the entries except the first one.
      for patch, indices in entries[1:]:
        rows_to_delete.append(patch_indices[patch][indices])

  def unflatten_ctrl(self, flat):
    ''' Turn the flattened generalized coordinate vector back into control
        points.
    '''
    # Can we resolve the constraints in one shot?
    pass

  def flatten_ctrl(self, ctrl):
    ''' Turn a list of control points into a flattened vector that accounts
        for constraints.
    '''
    unconstrained = np.hstack(map(np.ravel, ctrl))

    # Get all patch labels.
    # Get fixed values.
    # Turn these into a matrix and a vector reflecting equality constraints.
    # Find the parameterization via SVD.


  def render(self, ctrl, filename=None):
    fig = plt.figure()
    ax  = plt.axes()
    ax.set_aspect('equal')

    uu = np.linspace(0, 1, 1002)[1:-1]

    rendered_labels = set()

    for patch, patch_ctrl in zip(self.patches, ctrl):

      # Plot vertical lines.
      for jj in range(patch_ctrl.shape[1]):
        xx = bsplines.bspline1d(
          uu,
          patch_ctrl[:,jj,:],
          patch.xknots,
          patch.deg,
        )
        ax.plot(xx[:,0], xx[:,1], 'k-')

      # Plot horizontal lines.
      for ii in range(patch_ctrl.shape[0]):
        yy = bsplines.bspline1d(
          uu,
          patch_ctrl[ii,:,:],
          patch.yknots,
          patch.deg,
        )
        ax.plot(yy[:,0], yy[:,1], 'k-')

      # Plot the control points themselves.
      ax.plot(patch_ctrl[:,:,0].ravel(), patch_ctrl[:,:,1].ravel(), 'k.')

      # Plot labels.
      label_r, label_c = onp.where(patch.pretty_labels)
      for ii in range(len(label_r)):
        row = label_r[ii]
        col = label_c[ii]
        text = patch.pretty_labels[row,col]
        if text not in rendered_labels:
          rendered_labels.add(text)
        else:
          continue
        ax.annotate(text, patch_ctrl[row,col,:])

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
  r1_labels = onp.zeros((10,5), dtype='<U256')
  r1_labels[3:7,0]  = ['A', 'B', 'C', 'D']
  r1_labels[:3,-1]  = ['E', 'F', 'G']
  r1_labels[-3:,-1] = ['H', 'I', 'J']
  r1_patch  = Patch2D(r1_xknots, r1_yknots, r1_deg, r1_labels)

  # Create another rectangle.
  r2_deg    = 3
  r2_ctrl   = bsplines.mesh(np.array([3,4,5,6]), np.array([-4, -3, -2, -1, 0]))
  r2_xknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[0])
  r2_yknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[1])
  r2_labels = onp.zeros((4,5), dtype='<U256')
  r2_labels[:,-1] = ['A', 'B', 'C', 'D']
  r2_patch  = Patch2D(r2_xknots, r2_yknots, r2_deg, r2_labels)

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
  u1_labels = onp.zeros((3,8), dtype='<U256')
  u1_labels[:,0]  = ['E', 'F', 'G']
  u1_labels[:,-1] = ['H', 'I', 'J']
  u1_patch  = Patch2D(u1_xknots, u1_yknots, u1_deg, u1_labels)

  shape = Shape2D(r1_patch, r2_patch, u1_patch)

  print(shape.patch_indices())


  shape.render([r1_ctrl, r2_ctrl, u1_ctrl])

if __name__ == '__main__':
  main()
