import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from operator import itemgetter
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
    ''' Gather all fixed constraints. '''

    all_fixed = set()
    for patch in self.patches:
      for label in patch.get_fixed():
        all_fixed.add(label)

    return all_fixed

  def unflat_size(self):
    num_unflat = 0
    for patch in self.patches:
      num_unflat += onp.prod(patch.get_ctrl_shape())
    return num_unflat

  def flatten_mat(self):

    # Set of keys that are fixed to particular values.
    fixed = self.get_fixed()

    patch_indices = self.patch_indices()

    # Indices to delete.
    to_delete = []

    # Loop over all coincidence constraints.
    for label, entries in self.get_coincidence().items():
      # Entries are only row/col.

      if label in fixed:
        # If the label is in the fixed set, delete them all.
        # Note that all fixed labels that are used should appear here.
        for patch, indices in entries:
          to_delete.append( patch_indices[patch][indices] )

      else:

        # Connect the entries to their global indices and sort.
        index_list = sorted(
          sorted(
            map(
              lambda ent: patch_indices[ent[0]][ent[1]], entries
            ),
            key=itemgetter(1),
          ),
          key=itemgetter(0),
        )

        # Keep the smallest one from the sort, delete the rest.
        to_delete.extend(index_list[1:])

    num_unflat = self.unflat_size()

    # Expand the to_delete to include dimensions.
    to_delete = [item for pair in to_delete for item in pair]

    # Return an identity matrix with those rows deleted.
    return onp.delete(onp.eye(num_unflat), to_delete, axis=0)

  def unflatten_mat(self):
    ''' The linear transformation that recovers all control points. '''

    fixed = self.get_fixed()

    # Start with the transpose of the flattening matrix.
    unflat_mat = self.flatten_mat().T

    patch_indices = self.patch_indices()

    # Loop over coincidence constraints.
    for label, entries in self.get_coincidence().items():

      if label not in fixed:

        # Connect the entries to their global indices and sort.
        index_list = sorted(
          sorted(
            map(
              lambda ent: patch_indices[ent[0]][ent[1]], entries
            ),
            key=itemgetter(1),
          ),
          key=itemgetter(0),
        )

        # What is the index of the value being copied?
        x_parent = onp.nonzero(unflat_mat[index_list[0][0],:])[0][0]
        y_parent = onp.nonzero(unflat_mat[index_list[0][1],:])[0][0]

        for pair in index_list[1:]:
          unflat_mat[pair[0], x_parent] = 1
          unflat_mat[pair[1], y_parent] = 1

    return unflat_mat

  def unflatten_fixed_mat(self):

    # Build a matrix that takes raveled fixed locations and turns them into
    # raveled offsets.
    sz = self.unflat_size()

    # Start with empty matrix.
    unflat_fixed_mat = onp.zeros((sz,sz))

    fixed = self.get_fixed()

    patch_indices = self.patch_indices()

    for patch in self.patches:
      for label, indices in patch.get_labels():
        if label in fixed:
          for dim in [0,1]:
            index = patch_indices[patch][tuple(list(indices) + [dim])]
            unflat_fixed_mat[index, index] = 1

    return unflat_fixed_mat

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
    unflat_mat = self.unflatten_mat()
    unflat_fixed_mat = self.unflatten_fixed_mat()

    sizes = [patch.get_ctrl_shape() for patch in self.patches]
    lens  = [onp.prod(size) for size in sizes]

    def unflatten(flat_ctrl, flat_vels, fixed_locs):

      # Get into one big vector.
      fixed_ravel = np.hstack(map(np.ravel, fixed_locs))

      # FIXME: Do we need to have the velocities of moving but "fixed" points
      # be non-zero?

      ravel_ctrl = unflat_mat @ flat_ctrl + unflat_fixed_mat @ fixed_ravel
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

      return np.array(ctrl), np.array(vels)

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

          label_r, label_c = onp.where(patch.labels)
          for ii in range(len(label_r)):
            row = label_r[ii]
            col = label_c[ii]
            text = patch.labels[row,col]
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

          label_r, label_c = onp.where(patch.labels)
          for ii in range(len(label_r)):
            row = label_r[ii]
            col = label_c[ii]
            text = patch.labels[row,col]
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

    plt.close(fig)

  def create_static_image(
      self,
      ctrl_sol,
      filename,
      just_cp=False,
      fig_kwargs={},
  ):

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

      # Render the first time step.
      for patch, patch_ctrl in zip(self.patches, ctrl_sol):
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

    plt.savefig(filename)
    plt.close(fig)
