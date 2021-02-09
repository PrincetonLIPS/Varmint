import jax
import jax.numpy as np
import numpy     as onp

from scipy.spatial          import Delaunay
from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import connected_components
from scipy.spatial.distance import pdist, squareform

from .bsplines import mesh

def get_side_indices(cell_indices, side):
  # cell indices should have shape (4, cp, cp)
  # side can be top, bottom, left, right
  if side == 'top':
    return cell_indices[1, 4, :]
  elif side == 'bottom':
    return cell_indices[3, 4, :]
  elif side == 'left':
    return cell_indices[0, 4, :]
  elif side == 'right':
    return cell_indices[2, 4, :]
  else:
    raise ValueError(f'Invalid side {side}')

def index_array_from_ctrl(num_x, num_y, ctrl):
  n_cp = ctrl.size // ctrl.shape[-1]
  unflat_indices = onp.arange(n_cp).reshape(ctrl.shape[:-1])

  row_inds = []
  col_inds = []

  # Handle the internal constraints for each cell.
  n_cells = ctrl.shape[0] // 4
  for cell in range(n_cells):
    si = cell * 4
    for i in range(4):
      ind1 = unflat_indices[si + ((i+1)%4), :, 0]
      ind2 = unflat_indices[si + i,       :, 4]

      row_inds.extend([ind1, ind2])
      col_inds.extend([ind2, ind1])
  
  # Now handle the constraints between cells
  unflat_indices = unflat_indices.reshape(num_x, num_y, 4, ctrl.shape[-3], ctrl.shape[-2])
  for x in range(num_x):
    for y in range(num_y):
      this_cell = unflat_indices[x, y]
      if x > 0:
        # Handle constraint with left
        that_cell = unflat_indices[x-1, y]
        side1 = get_side_indices(this_cell, 'left')
        side2 = get_side_indices(that_cell, 'right')
        
        row_inds.append(side1)
        col_inds.append(np.flip(side2))
      if y > 0:
        # Handle constraint with bottom
        that_cell = unflat_indices[x, y-1]
        side1 = get_side_indices(this_cell, 'bottom')
        side2 = get_side_indices(that_cell, 'top')
        
        row_inds.append(side1)
        col_inds.append(np.flip(side2))

      if x < num_x - 1:
        # Handle constraint with right
        that_cell = unflat_indices[x+1, y]
        side1 = get_side_indices(this_cell, 'right')
        side2 = get_side_indices(that_cell, 'left')
        
        row_inds.append(side1)
        col_inds.append(np.flip(side2))

      if y < num_y - 1:
        # Handle constraint with top
        that_cell = unflat_indices[x, y+1]
        side1 = get_side_indices(this_cell, 'top')
        side2 = get_side_indices(that_cell, 'bottom')
        
        row_inds.append(side1)
        col_inds.append(np.flip(side2))

  all_rows = onp.concatenate(row_inds)
  all_cols = onp.concatenate(col_inds)

  spmat = csr_matrix((onp.ones_like(all_rows), (all_rows, all_cols)), shape=(n_cp, n_cp), dtype=onp.int8)

  n_components, labels = connected_components(
      csgraph=spmat,
      directed=False,
      return_labels=True
  )

  labels= onp.reshape(labels, ctrl.shape[:-1])
  
  return n_components, labels

def match_labels(ctrl, keep_singletons=True, epsilon=1e-6):

  # Keep the last dimension the same.
  unrolled = np.reshape(ctrl, (-1, ctrl.shape[-1]))

  # Compute inter-control point distances.
  dists = squareform(pdist(unrolled))

  # Turn this into an adjacency matrix.
  adjacency = csr_matrix(dists < epsilon)

  # Find the connected components.
  n_components, labels = connected_components(
    csgraph=adjacency,
    directed=False,
    return_labels=True
  )

  if not keep_singletons:

    # See how many times each label occurred.
    counts = np.bincount(labels)

    # Identify labels with only one occurrence.
    singletons = np.nonzero(counts==1)[0]

    # Get the control points that are not connected.
    loners = np.sum(labels[:,np.newaxis] == singletons[np.newaxis,:], axis=1)

    # Relabel those control points to be -1.
    labels[loners==1] = -1

  # Turn into strings as appropriate.
  str_labels = onp.array(list(map(
    lambda ii: '' if ii < 0 else 'group%d' % (ii),
    labels)), dtype='<U256')

  # Reshape to reflect the control point shape, i.e., excluding the last dim.
  str_labels = onp.reshape(str_labels, ctrl.shape[:-1])

  return str_labels

def _gen_cell_edge(center, corner1, corner2, radii):
  num_ctrl = len(radii)

  right_perim = np.linspace(corner1, corner2, num_ctrl)

  theta_start = np.arctan2(corner1[1]-center[1],
                           corner1[0]-center[0])
  theta_end = np.arctan2(corner2[1]-center[1],
                         corner2[0]-center[0])
  theta = np.linspace(theta_start, theta_end, num_ctrl)

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

def generate_quad_lattice(widths, heights, radii):
  width_mesh  = np.concatenate([np.array([0.0]), np.cumsum(widths)])
  height_mesh = np.concatenate([np.array([0.0]), np.cumsum(heights)])

  inds = np.stack(np.meshgrid(np.arange(len(widths)), np.arange(len(heights)), indexing='ij'), axis=-1).reshape(-1, 2)
  
  def handle_cell(ind):
    ii, jj = ind[0], ind[1]
    corners = np.array([
        [width_mesh[ii], height_mesh[jj]],
        [width_mesh[ii], height_mesh[jj+1]],
        [width_mesh[ii+1], height_mesh[jj+1]],
        [width_mesh[ii+1], height_mesh[jj]]
    ])

    cell_ctrl = _gen_cell(corners, radii[ii,jj,:])
    return cell_ctrl

  ctrls = jax.lax.map(handle_cell, inds)

  return ctrls.reshape(ctrls.shape[0] * ctrls.shape[1], ctrls.shape[2], ctrls.shape[3], ctrls.shape[4])

'''
def generate_tri_lattice(num_x, num_y, width):

  hexmesh = mesh(np.arange(num_x), np.sqrt(3)*np.arange(num_y//2))
  hexmesh = np.concatenate([
    hexmesh,
    hexmesh + np.array([[[0.5,np.sqrt(3)/2]]])
  ])
  points = width*np.reshape(hexmesh, (-1,2))

  tri = Delaunay(points)
  print(tri.simplices.shape)

  ctrl = []
  for simplex in tri.simplices:
    corners = points[simplex,:]

    # TODO: think through the radii thing...
    #cell_ctrl = _gen_cell(corners, radii[ii,jj,:])
    #  ctrl.extend(cell_ctrl)

  return np.array(ctrl)
'''
