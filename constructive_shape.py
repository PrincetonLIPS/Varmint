from collections import defaultdict
import numpy as np
import numpy.random as npr

from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import connected_components, dijkstra


class ShapeUnit2D(object):
  def get_ctrl_offset(self):
    return self.ctrl_offset

  def compute_internal_constraints(self):
    """Get internal constraints.
    
    Returns (row_inds, col_inds), where each element is an array of
    indices, shifted by ctrl_offset.
    """
    raise NotImplementedError()
  
  def get_side_indices(self, side):
    """Side can be left, top, right, bottom.
    
    Returns a 1-D array of indices, shifted by ctrl_offset.
    Order should be left->right or bottom->up.
    """
    raise NotImplementedError()

  def get_side_index_array(self, side, global_n_patches):
    """Returns a binary array that selects the indices of a side."""
    raise NotImplementedError()


class UnitCell2D(ShapeUnit2D):
  @staticmethod
  def _gen_random_radii(ncp, seed=None):
    npr.seed(seed)
    init_radii = npr.rand((ncp - 1) * 4) * 0.9 + 0.05
    return np.array(init_radii)

  @staticmethod
  def _gen_cell_edge(center, corner1, corner2, radii):
    num_ctrl = len(radii)

    right_perim = np.linspace(corner1, corner2, num_ctrl)
    left_perim = radii[:,np.newaxis] * (right_perim - center) + center

    ctrl = np.linspace(left_perim, right_perim, num_ctrl)

    return ctrl

  @staticmethod
  def _gen_cell(corners, radii):
    sides    = corners.shape[0]
    num_ctrl = (len(radii) // sides) + 1
    centroid = np.mean(corners, axis=0)


    # Computes: left, top, right, bottom
    ctrl = []
    for ii in range(sides):
      corner1 = corners[ii,:]
      corner2 = corners[(ii+1) % sides]
      start   = (num_ctrl - 1) * ii
      end     = start + num_ctrl
      indices = np.arange(start, end)

      new_ctrl = UnitCell2D._gen_cell_edge(
        centroid,
        corner1,
        corner2,
        np.take(radii, indices, mode='wrap'),
      )

      ctrl.append(new_ctrl)

    return np.stack(ctrl, axis=0)

  def __init__(self, corners, ncp, patch_offset, side_labels=None, radii=None):
    if radii is None:
      radii = UnitCell2D._gen_random_radii(ncp)

    if side_labels is not None:
      # side_labels can handle boundary conditions by creating groups
      # upon which you can impose boundary conditions.
      # Generally sides that are shared should not have a label.
      self.side_labels = side_labels

    self.ncp = ncp
    self.ctrl = UnitCell2D._gen_cell(corners, radii)
    n_all_ctrl = self.ctrl.size // self.ctrl.shape[-1]
    self.indices = np.arange(n_all_ctrl).reshape(self.ctrl.shape[:-1])
    self.ctrl_offset = patch_offset * ncp * ncp
    self.patch_offset = patch_offset

    self.n_patches = 4

  def compute_internal_constraints(self):
    row_inds = []
    col_inds = []

    for i in range(4):
      ind1 = self.indices[(i+1)%4, :, 0]
      ind2 = self.indices[i, :, -1]

      row_inds.extend([ind1, ind2])
      col_inds.extend([ind2, ind1])
    
    return (np.concatenate(row_inds) + self.ctrl_offset,
            np.concatenate(col_inds) + self.ctrl_offset)
  
  def get_side_indices(self, side):
    # bottom and right side indices must be flipped!
    # TODO: or is it the other way around? 
    if side == 'top':
      return self.indices[1, -1, :] + self.ctrl_offset
    elif side == 'bottom':
      return np.flip(self.indices[3, -1, :]) + self.ctrl_offset
    elif side == 'left':
      return self.indices[0, -1, :] + self.ctrl_offset
    elif side == 'right':
      return np.flip(self.indices[2, -1, :]) + self.ctrl_offset
    else:
      raise ValueError(f'Invalid side {side}')

  def get_side_index_array(self, side, global_n_patches):
    ind_array = np.zeros((global_n_patches, self.ncp, self.ncp))

    if side == 'top':
      ind_array[self.patch_offset + 1, -1, :] = 1
    elif side == 'bottom':
      ind_array[self.patch_offset + 3, -1, :] = 1
    elif side == 'left':
      ind_array[self.patch_offset + 0, -1, :] = 1
    elif side == 'right':
      ind_array[self.patch_offset + 2, -1, :] = 1
    else:
      raise ValueError(f'Invalid side {side}')

    return ind_array


class UnitSquare2D(ShapeUnit2D):
  def __init__(self, corners, ncp, patch_offset, side_labels=None):
    if side_labels is not None:
      self.side_labels = side_labels

    l1 = np.linspace(corners[0], corners[1], ncp)
    l2 = np.linspace(corners[3], corners[2], ncp)

    self.ncp = ncp
    self.ctrl = np.linspace(l1, l2, ncp)
    n_all_ctrl = self.ctrl.size // self.ctrl.shape[-1]
    self.indices = np.arange(n_all_ctrl).reshape(self.ctrl.shape[:-1])

    self.ctrl_offset = patch_offset * ncp * ncp
    self.patch_offset = patch_offset

    self.n_patches = 1

  def compute_internal_constraints(self):
    # Unit squares do not have internal constraints.
    return (np.array([]), np.array([]))

  def get_side_indices(self, side):
    if side == 'top':
      return self.indices[:, -1] + self.ctrl_offset
    elif side == 'bottom':
      return self.indices[:, 0] + self.ctrl_offset
    elif side == 'left':
      return self.indices[0, :] + self.ctrl_offset
    elif side == 'right':
      return self.indices[-1, :] + self.ctrl_offset
    else:
      raise ValueError(f'Invalid side {side}')

  def get_side_index_array(self, side, global_n_patches):
    ind_array = np.zeros((global_n_patches, self.ncp, self.ncp))

    if side == 'top':
      ind_array[self.patch_offset, :, -1] = 1
    elif side == 'bottom':
      ind_array[self.patch_offset, :, 0] = 1
    elif side == 'left':
      ind_array[self.patch_offset, 0, :] = 1
    elif side == 'right':
      ind_array[self.patch_offset, -1, :] = 1
    else:
      raise ValueError(f'Invalid side {side}')

    return ind_array


def get_connectivity_matrix(num_x, num_y, arr2lin, units, global_ctrl):
  n_cp = global_ctrl.size // global_ctrl.shape[-1]
  unflat_indices = np.arange(n_cp).reshape(global_ctrl.shape[:-1])

  row_inds = []
  col_inds = []

  # Handle internal constraints for each unit
  for u in units:
    new_row, new_col = u.compute_internal_constraints()
    row_inds.append(new_row)
    col_inds.append(new_col)

  # Handle the constraints between cells
  for x in range(num_x):
    for y in range(num_y):
      if (x, y) not in arr2lin:
        continue

      this_unit = units[arr2lin[(x, y)]]
      if x > 0 and (x-1, y) in arr2lin:
        # Handle constraint with left
        that_unit = units[arr2lin[(x-1, y)]]
        side1 = this_unit.get_side_indices('left')
        side2 = that_unit.get_side_indices('right')
        
        row_inds.append(side1)
        col_inds.append(side2)

      if y > 0 and (x, y-1) in arr2lin:
        # Handle constraint with bottom
        that_unit = units[arr2lin[(x, y-1)]]
        side1 = this_unit.get_side_indices('bottom')
        side2 = that_unit.get_side_indices('top')
        
        row_inds.append(side1)
        col_inds.append(side2)
        
      if x < num_x - 1 and (x+1, y) in arr2lin:
        # Handle constraint with right
        that_unit = units[arr2lin[(x+1, y)]]
        side1 = this_unit.get_side_indices('right')
        side2 = that_unit.get_side_indices('left')
        
        row_inds.append(side1)
        col_inds.append(side2)

      if y < num_y - 1 and (x, y+1) in arr2lin:
        # Handle constraint with top
        that_unit = units[arr2lin[(x, y+1)]]
        side1 = this_unit.get_side_indices('top')
        side2 = that_unit.get_side_indices('bottom')
        
        row_inds.append(side1)
        col_inds.append(side2)

  all_rows = np.concatenate(row_inds)
  all_cols = np.concatenate(col_inds)

  spmat = csr_matrix((np.ones_like(all_rows), (all_rows, all_cols)), shape=(n_cp, n_cp), dtype=np.int8)
  return unflat_indices, spmat


class MaterialGrid(object):
  def __init__(self, infile, ncp):
    cell_length = 5  # TODO(doktay): This is arbitrary.

    with open(infile, 'r') as f:
      lines = [l.strip().split(' ') for l in f.readlines()][::-1]
    
    # Grid will be specified as cells or solid squares (maybe add other shapes in the future)
    # Each location has two characters: First is shape type, second is boundary condition class.
    # S - square, C - cell, 0 - empty
    # 0 - no BC, 1...n - BC class
    # 00 00 00 00 00 00
    # S2 C0 C0 S0 S1 00
    # 00 S1 S1 S1 00 00

    # Let's transpose to be consistent with the way the code was written before
    cell_array = np.array(lines).T

    widths  = cell_length * np.ones(cell_array.shape[0])
    heights = cell_length * np.ones(cell_array.shape[1])

    width_mesh  = np.concatenate([np.array([0.0]), np.cumsum(widths)])
    height_mesh = np.concatenate([np.array([0.0]), np.cumsum(heights)])

    # Map cell_array indices to a linear index. Create control point array.
    self.arr2lin  = {}
    self.units    = []
    self.ctrls    = []
    self.fixed    = []
    self.fixed_groups = defaultdict(list)
    self.traction = []
    self.compress = []

    npatches = 0
    num_x = cell_array.shape[0]
    num_y = cell_array.shape[1]

    for i in range(num_x):
      for j in range(num_y):
        if cell_array[i, j][0] != '0':
          corners = np.array([
            [width_mesh[i], height_mesh[j]],  # bottom left
            [width_mesh[i], height_mesh[j+1]],  # top left
            [width_mesh[i+1], height_mesh[j+1]],  # top right
            [width_mesh[i+1], height_mesh[j]]  # bottom right
          ])

          if cell_array[i, j][0] == 'S':
            # Construct a square and add to cells and ctrls.
            unit = UnitSquare2D(corners, ncp, npatches)
          elif cell_array[i, j][0] == 'C':
            # Construct a cell.
            unit = UnitCell2D(corners, ncp, npatches)
          else:
            raise ValueError("Invalid shape.")
          
          npatches += unit.n_patches
          self.units.append(unit)
          self.ctrls.append(unit.ctrl.reshape(-1, ncp, ncp, 2))
          self.arr2lin[(i, j)] = len(self.units) - 1

          if cell_array[i, j][1] != '0':
            group = cell_array[i, j][1]
            self.fixed.append((len(self.units)-1, 'left'))
            self.fixed_groups[group].append((len(self.units)-1, 'left'))

          if cell_array[i, j][2] != '0':
            group = cell_array[i, j][2]
            self.fixed.append((len(self.units)-1, 'top'))
            self.fixed_groups[group].append((len(self.units)-1, 'top'))

          if cell_array[i, j][3] != '0':
            group = cell_array[i, j][3]
            self.fixed.append((len(self.units)-1, 'right'))
            self.fixed_groups[group].append((len(self.units)-1, 'right'))

          if cell_array[i, j][4] != '0':
            group = cell_array[i, j][4]
            self.fixed.append((len(self.units)-1, 'bottom'))
            self.fixed_groups[group].append((len(self.units)-1, 'bottom'))

    self.ctrls = np.concatenate(self.ctrls, axis=0)

    # Now create index array from matching control points.
    unflat_indices, spmat = get_connectivity_matrix(num_x, num_y, self.arr2lin, self.units, self.ctrls)
    n_components, labels = connected_components(
        csgraph=spmat,
        directed=False,
        return_labels=True
    )

    labels = np.reshape(labels, self.ctrls.shape[:-1])
    self.n_components = n_components
    self.labels = labels

    # Figure out fixed boundary conditions.
    if self.fixed:
      ind_array = \
          sum(self.units[i].get_side_index_array(side, npatches) for (i, side) in self.fixed)
      fixed_indices = unflat_indices[ind_array > 0]
      all_dists = dijkstra(spmat, directed=False, indices=fixed_indices,
                           unweighted=True, min_only=True)
      all_dists = np.reshape(all_dists, self.ctrls.shape[:-1])
      self.fixed_labels = np.unique(self.labels[all_dists < np.inf])
      self.nonfixed_labels = np.unique(self.labels[all_dists == np.inf])
      self.group_labels = {}

      for group in self.fixed_groups:
        sides = self.fixed_groups[group]
        group_array = \
            sum(self.units[i].get_side_index_array(side, npatches) for (i, side) in sides)
        group_indices = unflat_indices[group_array > 0]      
        group_all_dists = dijkstra(spmat, directed=False, indices=group_indices,
                                  unweighted=True, min_only=True)
        group_all_dists = np.reshape(group_all_dists, self.ctrls.shape[:-1])
        self.group_labels[group] = group_all_dists < np.inf
    else:
      self.fixed_labels = []
      self.nonfixed_labels = []