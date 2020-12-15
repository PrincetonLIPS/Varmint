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
  bspline2d,
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

  def get_flatten_fn(self):

    flatten_mat = self.flatten_mat()

    def flatten(ctrl, vels):
      ravel_ctrl = np.hstack(map(np.ravel, ctrl))
      ravel_vels = np.hstack(map(np.ravel, vels))

      flat_ctrl = flatten_mat @ ravel_ctrl
      flat_vels = flatten_mat @ ravel_vels

      return flat_ctrl, flat_vels

    return flatten

  def get_unflatten_fn(self):
    unflat_mat, unflat_vec = self.unflatten_mat_vec()

    sizes = [patch.get_ctrl_shape() for patch in self.patches]
    lens  = [onp.prod(size) for size in sizes]

    def unflatten(flat_ctrl, flat_vels):
      # Fixed control points have zero velocity.
      ravel_ctrl = unflat_mat @ flat_ctrl + unflat_vec
      ravel_vels = unflat_mat @ flat_vels

      # Now, to reshape for each patch.
      ctrl = []
      vels = []
      start_idx = 0
      for ii, patch in enumerate(self.patches):
        end_idx = start_idx + lens[ii]

        ctrl.append(np.reshape(ravel_ctrl[start_idx:end_idx], sizes[ii]))
        vels.append(np.reshape(ravel_vels[start_idx:end_idx], sizes[ii]))

        start_idx = end_idx

      return ctrl, vels

    return unflatten

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
    N  = 100
    uu = np.linspace(1e-6, 1-1e-6, N)
    path = np.hstack([
      np.vstack([uu[0]*np.ones(N), uu]),
      np.vstack([uu, uu[-1]*np.ones(N)]),
      np.vstack([uu[-1]*np.ones(N), uu[::-1]]),
      np.vstack([uu[::-1], uu[0]*np.ones(N)]),
    ]).T

    def init():

      # Render the first time step.
      for patch, patch_ctrl in zip(self.patches, ctrl_seq[0]):

        locs = bspline2d(
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
        objects[(patch, 'p')] = poly

        if labels:
          # Plot labels.
          rendered_labels = set()

          label_r, label_c = onp.where(patch.pretty_labels)
          for ii in range(len(label_r)):
            row = label_r[ii]
            col = label_c[ii]
            text = patch.pretty_labels[row,col]
            if True: #text not in rendered_labels:
              rendered_labels.add(text)
            else:
              continue
            ann = ax.annotate(text, patch_ctrl[row,col,:], size=6)
            objects[(patch,'a',ii)] = ann

      return objects.values()

    def update(tt):

      for patch, patch_ctrl in zip(self.patches, ctrl_seq[tt]):

        locs = bspline2d(
          path,
          patch_ctrl,
          patch.xknots,
          patch.yknots,
          patch.spline_deg,
        )
        objects[(patch, 'p')].set_xy(locs)

        if labels:
          # Plot labels.
          rendered_labels = set()

          label_r, label_c = onp.where(patch.pretty_labels)
          for ii in range(len(label_r)):
            row = label_r[ii]
            col = label_c[ii]
            text = patch.pretty_labels[row,col]
            if True: #text not in rendered_labels:
              rendered_labels.add(text)
            else:
              continue
            ann = ax.annotate(text, patch_ctrl[row,col,:], size=6)
            objects[(patch,'a',ii)] = ann


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
