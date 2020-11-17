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
    for patch in self.patches:
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
        all_coincidence[label].append((patch, tuple(indices)))

    return all_coincidence

  def get_fixed(self):
    ''' Get all the fixed constraints. '''

    all_fixed = {}
    for patch in self.patches:
      for label, value in patch.fixed:
        if label in all_fixed:
          if all_fixed[label] != value[ii]:
            raise LabelError("Inconsistent fixed constraint %s: %f vs %f" \
                             % (label, all_fixed[label], value[ii]))

          all_fixed[label] = value[ii]

    return all_fixed

  def flatten_matrix(self):
    ''' The matrix that reparameterizes to remove linear constraints. '''
    num_unflat = 0
    for patch in self.patches:
      num_unflat += onp.prod(patch.get_ctrl_shape())

    # Start with a big identity matrix.
    matrix = onp.eye(num_unflat)

    # Accumulate the set of rows to remove.
    # Can't do it incrementally without breaking the indexing.
    # Initialize with all the fixed constraints.
    rows_to_delete = set() # set(self.get_fixed().keys()) temporary

    patch_indices = self.patch_indices()

    # Loop over coincidence constraints.
    for label, entries in self.get_coincidences().items():

      # Delete all the entries except the first one.
      for patch, indices in entries[1:]:
        rows_to_delete.add(patch_indices[patch][indices])

    return onp.delete(matrix, list(rows_to_delete), axis=0)

  def unflatten_mat_vec(self):
    ''' The linear transformation that recovers all control points. '''

    # FIXME: This is a slow way to do it.
    flat_mat = self.flatten_matrix()

    matrix = onp.eye(flat_mat.shape[0])

    deleted_rows = set()

    patch_indices = self.patch_indices()

    # Loop over coincidence constraints.
    for label, entries in self.get_coincidences().items():

      parent_patch = entries[0][0]
      parent_index = patch_indices[parent_patch][entries[0][1]]

      # We deleted all but the first one.
      for patch, indices in entries[1:]:
        deleted_rows.add((parent_index, patch_indices[patch][indices]))

    # Get them in increasing order of destination.
    for parent, child in sorted(list(deleted_rows), key=lambda x: x[1]):
      print(child, parent)
      matrix = onp.insert(matrix, child, 0, axis=0)
      matrix[child,parent] = 1

    return matrix

  def flatten_ctrl(self, ctrl):
    ''' Turn a list of control points into a flattened vector that accounts
        for constraints.
    '''
    unconstrained = np.hstack(map(np.ravel, ctrl))
    flat_mat = self.flatten_matrix()
    flattened = self.flatten_matrix() @ unconstrained

    return flattened

  def unflatten_ctrl(self, flat):
    ''' Turn the flattened generalized coordinate vector back into control
        points.
    '''
    unflat_mat = self.unflatten_mat_vec()
    constrained = unflat_mat @ flat

    plt.imshow(unflat_mat)
    plt.show()


    # Now, to reshape for each patch.
    ctrl = []
    start_idx = 0
    for patch in self.patches:
      size = patch.get_ctrl_shape()
      end_idx = start_idx + onp.prod(size)
      ctrl.append(np.reshape(constrained[start_idx:end_idx], size))
      start_idx = end_idx

    return ctrl

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

  ctrl = [r1_ctrl, r2_ctrl, u1_ctrl]
  flat = shape.flatten_ctrl(ctrl)

  unflat = shape.unflatten_ctrl(flat)

  for ii in range(3):
    print((ctrl[ii]-unflat[ii]).ravel())

  #print(flat)
  #shape.render(ctrl)

if __name__ == '__main__':
  main()
