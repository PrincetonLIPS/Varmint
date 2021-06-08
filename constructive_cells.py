import jax
import jax.numpy as np
import numpy     as onp
import numpy.random as npr

from scipy.spatial          import Delaunay
from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import connected_components, dijkstra
from scipy.spatial.distance import pdist, squareform

def get_side_index_array(cell_indices, cell, side):
  # cell indices should have shape (n_cells, 4, cp, cp)
  # side can be top, bottom, left, right
  ind_array = onp.zeros_like(cell_indices)

  if side == 'top':
    ind_array[cell, 1, -1, :] = 1
  elif side == 'bottom':
    ind_array[cell, 3, -1, :] = 1
  elif side == 'left':
    ind_array[cell, 0, -1, :] = 1
  elif side == 'right':
    ind_array[cell, 2, -1, :] = 1
  else:
    raise ValueError(f'Invalid side {side}')

  return ind_array

def get_side_indices(cell_indices, side):
  # cell indices should have shape (4, cp, cp)
  # side can be top, bottom, left, right
  if side == 'top':
    return cell_indices[1, -1, :]
  elif side == 'bottom':
    return cell_indices[3, -1, :]
  elif side == 'left':
    return cell_indices[0, -1, :]
  elif side == 'right':
    return cell_indices[2, -1, :]
  else:
    raise ValueError(f'Invalid side {side}')

def get_connectivity_matrix(cell_array, arr2lin, ctrl):
  n_cp = ctrl.size // ctrl.shape[-1]
  unflat_indices = onp.arange(n_cp).reshape(ctrl.shape[:-1])

  row_inds = []
  col_inds = []

  # Handle the internal constraints for each cell.
  n_cells = ctrl.shape[0]
  for cell in range(n_cells):
    for i in range(4):
      ind1 = unflat_indices[cell, (i+1)%4, :, 0]
      ind2 = unflat_indices[cell, i,       :, 4]

      row_inds.extend([ind1, ind2])
      col_inds.extend([ind2, ind1])
  
  # Now handle the constraints between cells
  num_x = cell_array.shape[0]
  num_y = cell_array.shape[1]

  for x in range(num_x):
    for y in range(num_y):
      if (x, y) not in arr2lin:
        continue
      this_cell = unflat_indices[arr2lin[(x, y)]]
      if x > 0 and (x-1, y) in arr2lin:
        # Handle constraint with left
        that_cell = unflat_indices[arr2lin[(x-1, y)]]
        side1 = get_side_indices(this_cell, 'left')
        side2 = get_side_indices(that_cell, 'right')

        row_inds.append(side1)
        col_inds.append(np.flip(side2))
      if y > 0 and (x, y-1) in arr2lin:
        # Handle constraint with bottom
        that_cell = unflat_indices[arr2lin[(x, y-1)]]
        side1 = get_side_indices(this_cell, 'bottom')
        side2 = get_side_indices(that_cell, 'top')

        row_inds.append(side1)
        col_inds.append(np.flip(side2))

      if x < num_x - 1 and (x+1, y) in arr2lin:
        # Handle constraint with right
        that_cell = unflat_indices[arr2lin[(x+1, y)]]
        side1 = get_side_indices(this_cell, 'right')
        side2 = get_side_indices(that_cell, 'left')

        row_inds.append(side1)
        col_inds.append(np.flip(side2))

      if y < num_y - 1 and (x, y+1) in arr2lin:
        # Handle constraint with top
        that_cell = unflat_indices[arr2lin[(x, y+1)]]
        side1 = get_side_indices(this_cell, 'top')
        side2 = get_side_indices(that_cell, 'bottom')

        row_inds.append(side1)
        col_inds.append(np.flip(side2))

  all_rows = onp.concatenate(row_inds)
  all_cols = onp.concatenate(col_inds)

  spmat = csr_matrix((onp.ones_like(all_rows), (all_rows, all_cols)), shape=(n_cp, n_cp), dtype=onp.int8)
  return unflat_indices, spmat

def _gen_cell_edge(center, corner1, corner2, radii):
  num_ctrl = len(radii)

  right_perim = np.linspace(corner1, corner2, num_ctrl)
  left_perim = radii[:,np.newaxis] * (right_perim - center) + center

  ctrl = np.linspace(left_perim, right_perim, num_ctrl)

  return ctrl

def _gen_cell(corners, radii):
  sides    = corners.shape[0]
  num_ctrl = (len(radii) // sides) + 1
  centroid = np.mean(corners, axis=0)

  ctrl = []
  for ii in range(sides):
    corner1 = corners[ii,:]
    corner2 = corners[(ii+1) % sides]
    start   = (num_ctrl - 1) * ii
    end     = start + num_ctrl
    indices = np.arange(start, end)

    new_ctrl = _gen_cell_edge(
      centroid,
      corner1,
      corner2,
      np.take(radii, indices, mode='wrap'),
    )

    ctrl.append(new_ctrl)

  return np.stack(ctrl, axis=0)

def _gen_random_radii(ncp, seed=None):
  npr.seed(seed)
  init_radii = npr.rand((ncp - 1) * 4) * 0.9 + 0.05
  return np.array(init_radii)


class UnitCell2D(object):

  def __init__(self, corners, ncp, side_labels=None, radii=None):
    if radii is None:
      radii = _gen_random_radii(ncp)

    if side_labels is not None:
      # side_labels can handle boundary conditions by creating groups
      # upon which you can impose boundary conditions.
      # Generally sides that are shared should not have a label.
      self.side_labels = side_labels

    self.ctrl = _gen_cell(corners, radii)


class MaterialGrid(object):
  def __init__(self, infile):
    cell_length = 5  # TODO(doktay): This is arbitrary.
    ncp = 5  # TODO(doktay): Should be parameter.

    with open(infile, 'r') as f:
      lines = [list(l.strip()) for l in f.readlines()][::-1]

    # Let's transpose to be consistent with the way the code was written before
    cell_array = onp.array(lines).T

    widths  = cell_length * np.ones(cell_array.shape[0])
    heights = cell_length * np.ones(cell_array.shape[1])

    width_mesh  = np.concatenate([np.array([0.0]), np.cumsum(widths)])
    height_mesh = np.concatenate([np.array([0.0]), np.cumsum(heights)])

    # Map cell_array indices to a linear index. Create control point array.
    self.arr2lin = {}
    self.cells = []
    self.ctrls = []
    self.fixed = []

    for i in range(cell_array.shape[0]):
      for j in range(cell_array.shape[1]):
        if cell_array[i, j] != '0':
          corners = np.array([
            [width_mesh[i], height_mesh[j]],
            [width_mesh[i], height_mesh[j+1]],
            [width_mesh[i+1], height_mesh[j+1]],
            [width_mesh[i+1], height_mesh[j]]
          ])

          self.cells.append(UnitCell2D(corners, ncp))
          self.arr2lin[(i, j)] = len(self.cells) - 1

          self.ctrls.append(self.cells[-1].ctrl)

          cell_char = cell_array[i, j]
          if cell_char == 'l':
            self.fixed.append((len(self.cells)-1, 'left'))
          elif cell_char == 'r':
            self.fixed.append((len(self.cells)-1, 'right'))
          elif cell_char == 't':
            self.fixed.append((len(self.cells)-1, 'top'))
          elif cell_char == 'b':
            self.fixed.append((len(self.cells)-1, 'bottom'))

    self.ctrls = np.stack(self.ctrls, axis=0)

    # Now create index array from matching control points.
    unflat_indices, spmat = get_connectivity_matrix(cell_array, self.arr2lin, self.ctrls)
    n_components, labels = connected_components(
        csgraph=spmat,
        directed=False,
        return_labels=True
    )

    labels = onp.reshape(labels, self.ctrls.shape[:-1])
    self.n_components = n_components
    self.labels = labels

    # Figure out fixed boundary conditions.
    if self.fixed:
      ind_array = \
          sum(get_side_index_array(self.labels, i, side) for (i, side) in self.fixed)
      fixed_indices = unflat_indices[ind_array > 0]
      all_dists = dijkstra(spmat, directed=False, indices=fixed_indices,
                           unweighted=True, min_only=True)
      all_dists = onp.reshape(all_dists, self.ctrls.shape[:-1])
      self.fixed_labels = np.unique(self.labels[all_dists < np.inf])
      self.nonfixed_labels = np.unique(self.labels[all_dists == np.inf])
    else:
      self.fixed_labels = []
      self.nonfixed_labels = []
