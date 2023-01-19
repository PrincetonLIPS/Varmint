import jax.numpy  as jnp
import jax.random as jrnd
import logging

from functools       import cached_property
from scipy.spatial   import ConvexHull
from scipy.stats.qmc import Sobol
from meshpy          import triangle, tet

from ..utils import boost2jax, boostlist2jax

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class FundamentalRegion:
  def __init__(self, dims: int, **kwargs) -> None:
    super().__init__(**kwargs)
    self.dims = dims

    #self.is_inside_cuts = jax.jit(self.is_inside_cuts)
    #self.is_inside      = jax.jit(self.is_inside)

  @cached_property
  def vertices(self):
    return jnp.unique(
      boostlist2jax(self._asu.shape_vertices())[:,:self.dims],
      axis=0,
    )

  @cached_property
  def hull(self):
    return ConvexHull(self.vertices)

  @cached_property
  def volume(self):
    return self.hull.volume

  @cached_property
  def bounding_box(self):
    bmin = boostlist2jax(self._asu.box_min())[:self.dims]
    bmax = boostlist2jax(self._asu.box_max())[:self.dims]
    return bmin, bmax

  @cached_property
  def cut_normals(self):
    ''' Get the matrix of cut plane normals. '''

    normals = []
    for cut in self._asu.cuts:
      normal = jnp.column_stack([
        # Truncate the dimension of the plane case.
        boost2jax(x) for x in cut.n[:self.dims]
      ])
      normals.append(normal)

    return jnp.row_stack(normals).T

  @cached_property
  def cut_distances(self):
    ''' Get the column vector of cut plane origin distances. '''

    distances = [boost2jax(cut.c) for cut in self._asu.cuts]

    return jnp.row_stack(distances).T

  def is_inside_cuts(self, points):
    return points @ self.cut_normals >= -self.cut_distances

  def is_inside(self, points):
    return jnp.all(self.is_inside_cuts(points), axis=1)

  def get_points_random(
      self,
      num_points : int,
      seed : int=1,
  ) -> jnp.DeviceArray:
    '''Select a random set of points in the fundamental region.

    Parameters
    ----------
    num_points : int
      Number of points to generate.

    seed : int, optional
      Random seed to use.

    Returns
    -------
    points : jax.numpy.DeviceArray
      A JAX array of floats that is num_points x D where D is 2 for plane groups
      and 3 for space groups.
    '''

    rng_key = jrnd.PRNGKey(seed)

    bmin, bmax = self.bounding_box

    # Will be areas when a plane group.
    fund_volume = self.volume
    bbox_volume = jnp.prod(bmax-bmin)

    assert fund_volume > 0
    assert bbox_volume > 0

    vol_frac = fund_volume / bbox_volume
    if jnp.isclose(vol_frac, 1.0):
      total_points = num_points
    else:
      overshoot_factor = 1.5
      total_points = int(overshoot_factor * num_points / vol_frac)

    # Loop until we get the right number, although this should almost never
    # actually loop since we're overshooting.
    while True:
      loop_key, rng_key = jrnd.split(rng_key, 2)

      points = jrnd.uniform(
        loop_key,
        shape=(total_points, self.dims),
        minval=bmin,
        maxval=bmax,
      )

      inside = self.is_inside(points)
      if jnp.sum(inside) >= num_points:
        points = points[inside,:][:num_points,:]
        break
      else:
        log.debug("Did not generate enough random points. Trying again.")

    return points

  def get_points_sobol(
      self,
      num_points : int,
      seed : int=1,
  ) -> jnp.DeviceArray:
    '''Create a low-discrepancy set of points in the fundamental region using a
       Sobol sequence. See https://en.wikipedia.org/wiki/Sobol_sequence

    Parameters
    ----------
    num_points : int
      Minimum number of points to generate.

    seed : int, optional
      Random seed to use.

    Returns
    -------
    points : jax.numpy.DeviceArray
      A JAX array of floats that is N x D where D is 2 for plane groups and 3
      for space groups. N will be at least num_points.
    '''

    sampler = Sobol(d=self.dims, scramble=True, seed=seed)

    bmin, bmax = self.bounding_box

    # Will be areas when a plane group.
    fund_volume = self.volume
    bbox_volume = jnp.prod(bmax-bmin)

    assert fund_volume > 0
    assert bbox_volume > 0

    vol_frac = fund_volume / bbox_volume
    if jnp.isclose(vol_frac, 1.0):
      total_points = num_points
    else:
      overshoot_factor = 1.1
      total_points = int(overshoot_factor * num_points / vol_frac)

    # Loop until we get the right number, although this should almost never
    # actually loop since we're overshooting.
    while True:
      log2_total = int(jnp.ceil(jnp.log2(total_points)))

      points = jnp.array(sampler.random_base2(m=log2_total))
      points = points * (bmax-bmin) + bmin

      inside = self.is_inside(points)
      if jnp.sum(inside) >= num_points:
        points = points[inside,:]
        break
      else:
        total_points *= 1.1
        log.debug("Sobol did not generate enough points. Trying %d total." \
                  % (total_points))

    return points

  def get_points_mesh(
      self,
      num_points : int,
  ) -> jnp.DeviceArray:
    '''Create a set of points based on the vertices of a Delaunay mesh.

    Parameters
    ----------
    num_points : int
      Minimum number of points to generate.

    Returns
    -------
    points : jax.numpy.DeviceArray
      A JAX array of floats that is N x D where D is 2 for plane groups and 3
      for space groups. N will be at least num_points.
    '''

    mesher = triangle if self.dims == 2 else tet

    max_vol = self.hull.volume / num_points

    # Loop until we get the right number, although this should almost never
    # actually loop since we're overshooting.
    while True:
      info = mesher.MeshInfo()
      info.set_points(self.vertices)
      info.set_facets(self.hull.simplices)

      mesh = mesher.build(
        info,
        max_volume = max_vol,
        verbose = False,
      )

      points = jnp.array(mesh.points)

      if points.shape[0] < num_points:
        max_vol *= 0.9
        log.debug("Mesh did not generate enough points. Trying max_vol=%f" \
                  % (max_vol))
      else:
        break

    return points
