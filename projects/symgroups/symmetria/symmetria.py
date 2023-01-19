from __future__ import annotations
from typing     import Union, Callable

import jax
import jax.numpy as jnp
import logging

jax.config.update('jax_enable_x64', True)

from .groups import SymmetryGroup, SpaceGroup, PlaneGroup
from .orbifold import Orbifold
from .harmonics import HarmonicsGLDense, HarmonicsGLSparse, HarmonicsRR

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class Symmetria:
  ''' Tool for constructing and manipulating space group invariant functions.

  Examples:
    Creating a Symmetria plane group by number::

      sym = Symmetria.plane_group(3)

    Creating a Symmetria plane group by name::

      sym = Symmetria.plane_group('pm')

    Creating a Symmetria space group by number::

      sym = Symmetria.space_group(25)

    Creating a Symmetria space group by name::

      sym = Symmetria.space_group('Pmm2')


  '''

  def __init__(self, sym_group: SymmetryGroup, dims: int) -> None:
    ''' Don't call this directly. '''
    self.sg   = sym_group
    self.dims = dims

  @classmethod
  def space_group(cls, sg_id: Union[int, str]) -> Symmetria:
    ''' Class method to get Symmetria object for space group.

    Args:
      sg_id: Identifier for space group. Integer or string.

    Returns:
      Symmetria object for the space group.

    '''
    sg = SpaceGroup(sg_id)
    return cls(sg, 3)

  @classmethod
  def plane_group(cls, pg_id: Union[int, str]) -> Symmetria:
    ''' Class method to get Symmetria object for plane group.

    Args:
      pg_id: Identifier for plane group. Integer or string.

    Returns:
      Symmetria object for the plane group.
    '''
    pg = PlaneGroup(pg_id)
    return cls(pg, 2)

  def get_harmonics(
      self,
      num_verts:     int,
      method:        str,
      epsilon:       float,
      dense:         bool,
      quad_deg:      int,
      num_basis:     int,
      mesh_size:     int,
      ica:           bool,
      ica_thresh:    float,
      graph_method:  str,
      num_harmonics: int,
      use_cache:    bool = True,
      cache_dir:     str = '.symmetria.cache',
  ) -> Callable[jnp.DeviceArray, [jnp.DeviceArray, jnp.DeviceArray]]:
    ''' Get the harmonics for the plane/space group.

    This computes eigenfunctions and eigenvalues of the Laplace-Beltrami
    operator for the plane or space group.  This can be done one of two ways:
    1) by decomposing a graph Laplacian, or 2) by building an orbifold map,
    using that map to construct basis functions, and then approximating the
    eigenfunctions within the span of that basis.

    Args:
      num_verts (int): Number of vertices in orbit graph.  If using RR or GL
        with dense computations you probably don't want this to be more than
        about 5000.  If you do GL with a small epsilon and use sparsity, it can
        be more like 100000.

      method (str): Either 'gl' (graph Laplacian) or 'rr (Rayleigh-Ritz).

      epsilon (float): Kernel length scale. (GL only) This should be pretty
        small (~0.01) if you want to use sparse computations with a large
        number of vertices.

      dense (bool): Use dense linear algebra (GL only) otherwise sparse. RR is
        always dense.

      quad_deg (int): Quadrature degree (RR only).  This and mesh_size
        determine how the integrals are computed in Rayleigh-Ritz.  Degree of
        1 or 2 generally seems fine.

      num_basis (int): Number of basis functions (RR only). A good number here
        is something like 1000.

      mesh_size (int): Approximate number of mesh elements (RR only). The other
        aspect of quadrature in Rayleigh-Ritz.  Something like 10000 works.

      ica (bool): Use ICA disentangling. You probably want this.

      ica_thresh (float): Threshold for ICA eigenvalue multiplicity.
        Eigenvalues are considered to be the same if they are within this
        relative size.

      graph_method (str): 'mesh', 'random', or 'sobol'. Determines how points
        are placed in the fundamental region for the graph vertices.

      num_harmonics (int): Number of harmonics to return.  This matters mostly
        for the sparse computations and caching.

      use_cache (bool): Whether to load/save computations to cache. Default is
        True.

      cache_dir (str): Directory to use for cache. Default is
        '.symmetria.cache'.

    Returns:
      interpolator (Callable): Function from R^2 or R^3 to R^k where k is the
        number of harmonics.
      evals (np.array): Numpy array of eigenvalues.

    '''

    if method == 'gl':
      if dense:
        H = HarmonicsGLDense(
          group         = self.sg,
          num_verts     = num_verts,
          epsilon       = epsilon,
          graph_method  = graph_method,
          num_harmonics = num_harmonics,
        )
      else:
        H = HarmonicsGLSparse(
          group         = self.sg,
          num_verts     = num_verts,
          epsilon       = epsilon,
          graph_method  = graph_method,
          num_harmonics = num_harmonics,
        )
    else:
      H = HarmonicsRR(
        group         = self.sg,
        num_verts     = num_verts,
        graph_method  = graph_method,
        num_basis     = num_basis,
        num_harmonics = num_harmonics,
        mesh_size     = mesh_size,
        quad_deg      = quad_deg,
      )

    H.fit()
    if ica:
      interpolator = H.disentangle(eval_thresh=ica_thresh)
    else:
      interpolator = H.interpolate()

    return interpolator, H.evals

  def get_orbifold_map(
      self,
      num_verts:     int,
      graph_method:  str,
      embed_dims:    int = 0,
      interp_method: str = 'polyharm2',
      use_cache:    bool = True,
      cache_dir:     str = '.symmetria.cache',
  ) -> Callable[jnp.DeviceArray, [jnp.DeviceArray, jnp.DeviceArray]]:
    ''' Construct an orbifold map for the plane/space group.

    This returns a function that computes embeddings for the plane/space group.

    Args:
      num_verts (int): Number of vertices in the orbit graph.  Shouldn't be
        much more than 5000 as we need to do an eigendecomposition.

      graph_method (str): 'mesh', 'random', or 'sobol'.  Determines of the
        vertices of the orbit graph are laid out in the fundamental region.

      embed_dims (int): Number of dimensions to embed into.  If zero, it will
        attempt to select a reasonable number. Defaults to zero.

      interp_method (str): String to specify what method to use for
        interpolation. Defaults to polyharm2. (TODO options)

      use_cache (bool): Whether to load/save computations to cache. Default is
        True.

      cache_dir (str): Directory to use for cache. Default is
        '.symmetria.cache'.

    Return:
      interpolator (Callable): Function from R^2 or R^3 to R^k where k is the
        number of embedding dimensions.
    '''


    O = Orbifold(
      group = self.sg,
      num_verts = num_verts,
      graph_method = graph_method,
      use_cache = use_cache,
      cache_dir = cache_dir,
    )

    O.fit()

    return O.interpolate()

  def get_null_embedder(
      self,
      embed_dims,
  ) -> Callable[jnp.DeviceArray, jnp.DeviceArray]:

    def embedder(x):

      # Get the quotient.
      qx, idents = self.sg.quotient(x)

      if embed_dims == 0:
        ret_x = qx

      elif embed_dims < qx.shape[0]:
        # Truncate
        ret_x = qx[:,:embed_dims]

      else:
        # Pad with zeros.
        ret_x =jnp.column_stack([
          qx,
          jnp.zeros((qx.shape[0], embed_dims-qx.shape[1])),
        ])

      return ret_x, idents

    return embedder
