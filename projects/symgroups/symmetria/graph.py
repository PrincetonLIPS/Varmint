from typing import Tuple

import jax
import time
import tqdm
import logging
import numpy        as np
import jax.numpy    as jnp
import scipy.sparse as sprs

from scipy.spatial import KDTree
from scipy.sparse  import csc_matrix, csr_matrix

from .utils  import sqeuclidean
from .groups import SymmetryGroup

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class OrbitGraph:
  def __init__(
      self,
      sym_group: SymmetryGroup,
      num_verts: int,
      method:    str, # FIXME
  ) -> None:
    ''' Construct and manipulate the orbit graph.

    Args:
      sym_group (SymmetryGroup): The plane or space group object.

      num_verts (int): The approximate number of vertices for the graph.

      method (str): 'mesh', 'sobol', or 'random' for placing the vertices
        in the fundamental region.
    '''
    self.sg        = sym_group
    self.num_verts = num_verts
    self.max_trans = 2

    # Generate primal vertices.
    self.primal_verts = self.gen_primal_vertices(method)

  def gen_primal_vertices(
      self,
      method:      str  = 'mesh',
      include_ext: bool = True,
  ) -> jnp.DeviceArray:
    if method == 'random':
      interior_verts = self.sg.fund.get_points_random(self.num_verts)

    elif method == 'sobol':
      interior_verts = self.sg.fund.get_points_sobol(self.num_verts)

    elif method == 'mesh':
      include_ext = False
      interior_verts = self.sg.fund.get_points_mesh(self.num_verts)

    else:
      raise NotImplementedError("Unknown point generation method '%s'" \
                                % (method))

    verts = jnp.row_stack([
      self.sg.fund.vertices,
      interior_verts,
    ]) if include_ext else interior_verts
    log.debug("Using %d vertices." % (interior_verts.shape[0]))

    verts = jnp.unique(verts, axis=0)

    return verts

  def orbit_primal_vertices(self) -> jnp.DeviceArray:
    return self.sg.orbit(self.primal_verts, self.max_trans)

  def _get_basis_orbits(self) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    num_verts = self.primal_verts.shape[0]

    flat_orbits = self.orbit_primal_vertices().reshape(-1, self.sg.dims)

    # Transform the orbits into the appropriate basis.
    basis_orbits = flat_orbits @ self.sg.basic_basis.T
    basis_orbits = basis_orbits.reshape(-1, num_verts, self.sg.dims)

    # Transform the primals into the correct basis.
    basis_primal = self.primal_verts @ self.sg.basic_basis.T
    log.debug("Computed %d distinct copies of the fundamental region." \
              % (basis_orbits.shape[0]))

    return basis_primal, basis_orbits

  def distances_dense(self) -> jnp.DeviceArray:
    basis_primal, basis_orbits = self._get_basis_orbits()
    num_verts = basis_primal.shape[0]

    t0 = time.time()
    # Doing a scan instead of vmap helps this scale wrt memory.
    @jax.jit
    def scanfunc(minima, group_copy):
      minima = jnp.minimum(
        sqeuclidean(basis_primal, group_copy),
        minima,
      )
      return minima, None
    sqdists, _ = jax.lax.scan(
      scanfunc,
      jnp.inf * jnp.ones((num_verts, num_verts)),
      basis_orbits,
    )
    dists = jnp.sqrt(sqdists)
    t1 = time.time()
    log.debug("Computed dense distance matrix in %0.2f secs." % (t1-t0))

    #dists = jnp.sqrt(jnp.min(
    #  jax.vmap(sqeuclidean, in_axes=(None, 0))(
    #    basis_primal,
    #    basis_orbits,
    #  ),
    #  axis=0,
    #))

    return dists

  def distances_sparse(self, threshold: float) -> csr_matrix:
    basis_primal, basis_orbits = self._get_basis_orbits()

    t0 = time.time()
    kdt_primal = KDTree(basis_primal)
    kdt_orbits = KDTree(basis_orbits.reshape(-1, self.sg.dims))
    t1 = time.time()
    log.debug("Created KD trees in %0.2f secs." % (t1-t0))

    dists = kdt_primal.sparse_distance_matrix(kdt_orbits, threshold).tocsr()
    t2 = time.time()
    log.debug("Computed sparse distance matrix in %0.2f secs." % (t2-t0))

    return dists

  def laplacian_dense(self, epsilon: float) -> jnp.DeviceArray:
    dists = self.distances_dense()

    dists = dists + jnp.diag(jnp.inf*jnp.ones(dists.shape[0]))

    # Compute the Laplacian with kernel-based edge weights.
    W       = jnp.exp(-0.5 * dists**2 / epsilon**2)
    degrees = jnp.sum(W, axis=1)
    inv_D   = jnp.diag(1/degrees)
    L       = jnp.eye(W.shape[0]) - inv_D @ W

    return L

  def laplacian_sparse(self, epsilon: float, threshold: float) -> csc_matrix:
    dists = self.distances_sparse(threshold*epsilon)
    num_verts = dists.shape[0]

    csc_rows = []
    csc_cols = []
    csc_data = []
    for ii in tqdm.tqdm(range(num_verts)):

      M = dists[ii,:].reshape(-1, num_verts).T
      M.data = np.exp(-0.5 * M.data**2 / epsilon**2)

      closest = M.max(axis=1)

      cols = closest.row
      rows = ii*np.ones(len(cols)-1, dtype=np.int32)
      data = closest.data
      #csc_entries.extend(zip(data, zip(rows, cols)))

      match = cols == ii
      cols  = np.delete(cols, match)
      data  = np.delete(data, match)

      csc_rows.extend(rows)
      csc_cols.extend(cols)
      csc_data.extend(data)

    W = csc_matrix(
      (csc_data, (csc_rows, csc_cols)),
      shape=(num_verts, num_verts),
    )

    degrees = np.sum(W, axis=1)
    degrees = np.squeeze(np.asarray(degrees))
    inv_D = sprs.diags(
      1./degrees,
      shape=W.shape,
      format='csc',
    )

    L = sprs.eye(num_verts) - inv_D @ W

    return L
