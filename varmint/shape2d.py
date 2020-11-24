import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from .exceptions import (
  DimensionError,
  LabelError,
  )
from .patch2d import Patch2D
from .bsplines import (
  bspline1d,
  mesh,
)

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

  def get_coincidence(self):
    ''' Collect the coincidence constraints. '''

    # Gather all coincidence constraints, label-wise.
    all_coincidence = {}

    for patch in self.patches:
      for label, indices in patch.get_labels():
        if label not in all_coincidence:
          all_coincidence[label] = []
        all_coincidence[label].append((patch, tuple(indices)))

    return all_coincidence

  def get_fixed(self):
    ''' Get all the fixed constraints. '''

    all_fixed = {}
    for patch in self.patches:
      for label, value in patch.get_fixed().items():
        if label in all_fixed:
          if all_fixed[label] != value:
            raise LabelError("Inconsistent fixed constraint %s: %f vs %f" \
                             % (label, all_fixed[label], value))

        all_fixed[label] = value

    return all_fixed

  def unflat_size(self):
    num_unflat = 0
    for patch in self.patches:
      num_unflat += onp.prod(patch.get_ctrl_shape())
    return num_unflat

  def flatten_mat(self):

    # Set of keys that are fixed to particular values.
    fixed = self.get_fixed().keys()

    patch_indices = self.patch_indices()

    # Indices to delete.
    to_delete = set()

    # Loop over all coincidence constraints.
    for label, entries in self.get_coincidence().items():

      if label in fixed:
        # If the label is in the fixed set, delete them all.
        # Note that all fixed labels that are used should appear here.
        for patch, indices in entries:
          to_delete.add( patch_indices[patch][indices] )

      else:

        # Connect the entries to their global indices and sort.
        index_list = sorted(map(
          lambda ent: patch_indices[ent[0]][ent[1]], entries
        ))

        # Keep the smallest one, delete the rest.
        to_delete.update(index_list[1:])

    num_unflat = self.unflat_size()

    # Return an identity matrix with those rows deleted.
    return onp.delete(onp.eye(num_unflat), list(to_delete), axis=0)

  def unflatten_mat_vec(self):
    ''' The linear transformation that recovers all control points. '''

    fixed = self.get_fixed()

    # Start with an empty vector and the transpose of the flattening matrix.
    unflat_mat = self.flatten_mat().T
    unflat_vec = onp.zeros(unflat_mat.shape[0])

    patch_indices = self.patch_indices()

    # Loop over coincidence constraints.
    for label, entries in self.get_coincidence().items():

      if label in fixed:
        # Leave zeros in the matrix and put these in the vector.
        for patch, indices in entries:
          unflat_vec[patch_indices[patch][indices]] = fixed[label]

      else:

        # Connect the entries to their global indices and sort.
        index_list = sorted(map(
          lambda ent: patch_indices[ent[0]][ent[1]], entries
        ))

        # Slightly tricky because we need to account for having deleted entries
        # that were before the one we're referencing.
        parent = onp.nonzero(unflat_mat[index_list[0],:])[0][0]

        for child in index_list[1:]:
          unflat_mat[child, parent] = 1

    return unflat_mat, unflat_vec

  def flatten(self, ctrl, vels):
    ravel_ctrl = np.hstack(map(np.ravel, ctrl))
    ravel_vels = np.hstack(map(np.ravel, vels))

    flatten_mat = self.flatten_mat()

    flat_ctrl = flatten_mat @ ravel_ctrl
    flat_vels = flatten_mat @ ravel_vels

    return flat_ctrl, flat_vels

  def unflatten(self, flat_ctrl, flat_vels):
    unflat_mat, unflat_vec = self.unflatten_mat_vec()

    # Fixed control points have zero velocity.
    ravel_ctrl = unflat_mat @ flat_ctrl + unflat_vec
    ravel_vels = unflat_mat @ flat_vels

    # Now, to reshape for each patch.
    ctrl = []
    vels = []
    start_idx = 0
    for patch in self.patches:
      size = patch.get_ctrl_shape()
      end_idx = start_idx + onp.prod(size)

      ctrl.append(np.reshape(ravel_ctrl[start_idx:end_idx], size))
      vels.append(np.reshape(ravel_vels[start_idx:end_idx], size))

      start_idx = end_idx

    return ctrl, vels

  def create_movie(
      self,
      ctrl_seq,
      filename,
      labels=False,
      fig_kwargs={},
  ):

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
    uu = np.linspace(1e-6, 1-1e-6, 100)

    def init():

      # Render the first time step.
      for patch, patch_ctrl in zip(self.patches, ctrl_seq[0]):

        # Plot vertical lines.
        for jj in range(patch_ctrl.shape[1]):
          xx = bspline1d(
            uu,
            patch_ctrl[:,jj,:],
            patch.xknots,
            patch.spline_deg,
          )
          xline, = ax.plot(xx[:,0], xx[:,1], 'b-')
          objects[(patch,'x',jj)] = xline

        # Plot horizontal lines.
        for ii in range(patch_ctrl.shape[0]):
          yy = bspline1d(
            uu,
            patch_ctrl[ii,:,:],
            patch.yknots,
            patch.spline_deg,
          )
          yline, = ax.plot(yy[:,0], yy[:,1], 'b-')
          objects[(patch,'y',ii)] = yline

        # Plot the control points themselves.
        cpts, = ax.plot(patch_ctrl[:,:,0].ravel(),
                        patch_ctrl[:,:,1].ravel(), 'b.')
        objects[(patch,'c')] = cpts

        if labels:
          # Plot labels.
          rendered_labels = set()

          label_r, label_c = onp.where(patch.pretty_labels)
          for ii in range(len(label_r)):
            row = label_r[ii]
            col = label_c[ii]
            text = patch.pretty_labels[row,col]
            if text not in rendered_labels:
              rendered_labels.add(text)
            else:
              continue
            ann = ax.annotate(text, patch_ctrl[row,col,:])
            objects[(patch,'a',ii)] = ann

      return objects.values()

    def update(tt):

      for patch, patch_ctrl in zip(self.patches, ctrl_seq[tt]):

        # Plot vertical lines.
        for jj in range(patch_ctrl.shape[1]):
          xx = bspline1d(
            uu,
            patch_ctrl[:,jj,:],
            patch.xknots,
            patch.spline_deg,
          )
          objects[(patch,'x',jj)].set_data(xx[:,0], xx[:,1])

        # Plot horizontal lines.
        for ii in range(patch_ctrl.shape[0]):
          yy = bspline1d(
            uu,
            patch_ctrl[ii,:,:],
            patch.yknots,
            patch.spline_deg,
          )
          objects[(patch,'y',ii)].set_data(yy[:,0], yy[:,1])

        objects[(patch,'c')].set_data(patch_ctrl[:,:,0].ravel(),
                                      patch_ctrl[:,:,1].ravel())

        if labels:
          rendered_labels = set()

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
            objects[(patch,'a',ii)].xy = patch_ctrl[row,col,:]

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


'''
def main():

  # Create a rectangle.
  r1_deg    = 2
  r1_ctrl   = bsplines.mesh(np.arange(10), np.arange(5))
  r1_xknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[0])
  r1_yknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[1])
  r1_labels = onp.zeros((10,5), dtype='<U256')
  r1_labels[3:7,0]  = ['A', 'B', 'C', 'D']
  r1_labels[:3,-1]  = ['E', 'F', 'G']
  r1_labels[-3:,-1] = ['H', 'I', 'J']
  r1_fixed = {
    'E': [-1,5],
  }
  r1_patch  = Patch2D(r1_xknots, r1_yknots, r1_deg, r1_labels, r1_fixed)

  # Create another rectangle.
  r2_deg    = 2
  r2_ctrl   = bsplines.mesh(np.array([3,4,5,6]), np.array([-4, -3, -2, -1, 0]))
  r2_xknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[0])
  r2_yknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[1])
  r2_labels = onp.zeros((4,5), dtype='<U256')
  r2_labels[:,-1] = ['A', 'B', 'C', 'D']
  r2_labels[:,0]  = ['K', 'L', 'M', 'N']
  r2_fixed  = {
    'K': [3,-4],
    'L': [4,-4],
    'M': [5,-4],
    'N': [6,-5],

    # Fix these for testing coincidence+fixed.
    'A': [2.5,-0.5],
    # 'B': [4,0],
  }
  r2_patch  = Patch2D(r2_xknots, r2_yknots, r2_deg, r2_labels, r2_fixed)

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
  u1_labels[:,-1]  = ['E', 'F', 'G']
  u1_labels[:,0] = ['J', 'I', 'H']
  u1_fixed = {
    'H': [6,4.5],
  }
  u1_patch  = Patch2D(u1_xknots, u1_yknots, u1_deg, u1_labels, u1_fixed)

  shape = Shape2D(r1_patch, r2_patch, u1_patch)

  ctrl = [r1_ctrl, r2_ctrl, u1_ctrl]

  flat = shape.flatten(ctrl)

  unflat = shape.unflatten(flat)

  for ii in range(len(ctrl)):
    print((ctrl[ii]-unflat[ii]).ravel())

  shape.render(unflat)

if __name__ == '__main__':
  main()
'''
