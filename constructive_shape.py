from collections import defaultdict
from functools import partial
from jax._src.api import vmap
import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp

from scipy.sparse           import csr_matrix, csc_matrix, kron, save_npz
from scipy.sparse.csgraph   import connected_components, dijkstra

from varmint.sparsity import pattern_to_reconstruction


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

  def get_side_orientation(self, side, global_n_patches):
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

    right_perim = jnp.linspace(corner1, corner2, num_ctrl)
    left_perim = radii[:,jnp.newaxis] * (right_perim - center) + center

    ctrl = jnp.linspace(left_perim, right_perim, num_ctrl)

    return ctrl

  @staticmethod
  def _gen_cell(corners, radii):
    sides    = corners.shape[0]
    num_ctrl = (len(radii) // sides) + 1
    centroid = jnp.mean(corners, axis=0)


    # Computes: left, top, right, bottom
    ctrl = []
    for ii in range(sides):
      corner1 = corners[ii,:]
      corner2 = corners[(ii+1) % sides]
      start   = (num_ctrl - 1) * ii
      end     = start + num_ctrl
      indices = jnp.arange(start, end)

      new_ctrl = UnitCell2D._gen_cell_edge(
        centroid,
        corner1,
        corner2,
        jnp.take(radii, indices, mode='wrap'),
      )

      ctrl.append(new_ctrl)

    return jnp.stack(ctrl, axis=0)

  def __init__(self, corners, ncp, patch_offset, side_labels=None, radii=None):
    if radii is None:
      radii = UnitCell2D._gen_random_radii(ncp)

    if side_labels is not None:
      # side_labels can handle boundary conditions by creating groups
      # upon which you can impose boundary conditions.
      # Generally sides that are shared should not have a label.
      self.side_labels = side_labels

    self.ncp = ncp
    self.corners = corners
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
  
  def get_side_orientation(self, side, global_n_patches):
    ind_array = np.zeros((global_n_patches, 4))

    # Always on the "right" side of patch
    if side == 'top':
      ind_array[self.patch_offset + 1, 2] = 1
    elif side == 'bottom':
      ind_array[self.patch_offset + 3, 2] = 1
    elif side == 'left':
      ind_array[self.patch_offset + 0, 2] = 1
    elif side == 'right':
      ind_array[self.patch_offset + 2, 2] = 1
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
    self.corners = corners
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
  
  def get_side_orientation(self, side, global_n_patches):
    ind_array = np.zeros((global_n_patches, 4))

    if side == 'top':
      ind_array[self.patch_offset, 1] = 1
    elif side == 'bottom':
      ind_array[self.patch_offset, 3] = 1
    elif side == 'left':
      ind_array[self.patch_offset, 0] = 1
    elif side == 'right':
      ind_array[self.patch_offset, 2] = 1
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
  def __init__(self, instr, ncp):
    cell_length = 5  # TODO(doktay): This is arbitrary.
    self.ncp = ncp

    #with open(infile, 'r') as f:
    #  lines = [l.strip().split(' ') for l in f.readlines()][::-1]
    
    lines = [l.split(' ') for l in instr.strip().split('\n')[::-1]]
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
    self.traction_groups = defaultdict(list)

    npatches = 0
    self.n_cells = 0
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
            self.n_cells += 1
          else:
            raise ValueError("Invalid shape.")

          npatches += unit.n_patches
          self.units.append(unit)
          self.ctrls.append(unit.ctrl.reshape(-1, ncp, ncp, 2))
          self.arr2lin[(i, j)] = len(self.units) - 1

          if cell_array[i, j][1] != '0':
            group = cell_array[i, j][1]
            if group.isdigit():
              self.fixed.append((len(self.units)-1, 'left'))
              self.fixed_groups[group].append((len(self.units)-1, 'left'))
            else:
              self.traction_groups[group].append((len(self.units)-1, 'left'))

          if cell_array[i, j][2] != '0':
            group = cell_array[i, j][2]
            if group.isdigit():
              self.fixed.append((len(self.units)-1, 'top'))
              self.fixed_groups[group].append((len(self.units)-1, 'top'))
            else:
              self.traction_groups[group].append((len(self.units)-1, 'top'))

          if cell_array[i, j][3] != '0':
            group = cell_array[i, j][3]
            if group.isdigit():
              self.fixed.append((len(self.units)-1, 'right'))
              self.fixed_groups[group].append((len(self.units)-1, 'right'))
            else:
              self.traction_groups[group].append((len(self.units)-1, 'right'))

          if cell_array[i, j][4] != '0':
            group = cell_array[i, j][4]
            if group.isdigit():
              self.fixed.append((len(self.units)-1, 'bottom'))
              self.fixed_groups[group].append((len(self.units)-1, 'bottom'))
            else:
              self.traction_groups[group].append((len(self.units)-1, 'bottom'))


    self.ctrls = np.concatenate(self.ctrls, axis=0)

    # Now create index array from matching control points.
    unflat_indices, spmat = get_connectivity_matrix(num_x, num_y, self.arr2lin, self.units, self.ctrls)
    n_components, labels = connected_components(
        csgraph=spmat,
        directed=False,
        return_labels=True
    )

    print(f'Number of degrees of freedom: {2 * n_components}.')  # Each component has 2, one for x and one for y.

    all_mgrid_indices = []
    for patch_indices in unflat_indices:
      mgrid_indices = np.stack(np.meshgrid(patch_indices.flatten(), patch_indices.flatten()), axis=-1)
      all_mgrid_indices.append(mgrid_indices)
    all_mgrid_indices = np.stack(all_mgrid_indices, axis=0)

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

    print('constructing sparsity overestimate')
    self.jac_spy_edges = labels.flatten()[all_mgrid_indices].reshape((-1, 2))
    
    # Create an overestimate of the Jacobian sparsity pattern.
    jac_sparsity_graph = \
      csc_matrix((np.ones_like(self.jac_spy_edges[:, 0]), (self.jac_spy_edges[:, 0], self.jac_spy_edges[:, 1])),
                 (n_components, n_components), dtype=np.int8)
    jac_sparsity_graph = jac_sparsity_graph[:, self.nonfixed_labels]
    jac_sparsity_graph = jac_sparsity_graph[self.nonfixed_labels, :]
    self.jac_sparsity_graph = kron(jac_sparsity_graph, np.ones((2,2)), format='csc')
#    save_npz('VERYHUGEsparsemat.npz', self.jac_sparsity_graph)
    print('constructed sparsity overestimate')

    print('constructing sparsity ops (currently slow)')
    self.jvpmat, self.reconstruct = pattern_to_reconstruction(self.jac_sparsity_graph)
    print('constructed sparsity ops')

    self.traction_group_labels = {}
    for group in self.traction_groups:
      sides = self.traction_groups[group]
      self.traction_group_labels[group] = \
        sum(self.units[i].get_side_orientation(side, npatches) for (i, side) in sides)
    
    self.all_orientations = np.zeros((npatches, 4)) + \
        sum(self.traction_group_labels[g] for g in self.traction_group_labels)
  
  def get_radii_to_ctrl_fn(self):
    # Returns a JAX-able function that maps radii to control points of this shape.

    all_corners = []
    all_indices = []
    for u in self.units:
      if isinstance(u, UnitCell2D):
        all_corners.append(u.corners)
        all_indices.extend(list(range(u.patch_offset, u.patch_offset + 4)))

    # Case when we have no cells.
    if len(all_corners) == 0:
      return lambda _: self.ctrls

    all_corners = np.stack(all_corners, axis=0)
    all_indices = np.array(all_indices)

    vmap_gencell = jax.vmap(UnitCell2D._gen_cell)
    vmap_gencell = partial(vmap_gencell, all_corners)

    def radii_to_ctrl(radii):
      cell_ctrls = vmap_gencell(radii).reshape((-1, self.ncp, self.ncp, 2))
      return jax.ops.index_update(self.ctrls, all_indices, cell_ctrls,
                                  indices_are_sorted=True)
    return radii_to_ctrl