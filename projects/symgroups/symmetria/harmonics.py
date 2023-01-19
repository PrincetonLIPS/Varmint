import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import logging
import networkx as nx
import numpy as np
import os
import scipy.linalg as spla
import scipy.sparse.linalg as sprsla
import sklearn.decomposition as skd
import time

from .groups import SymmetryGroup
from .graph  import OrbitGraph
from .interpolate import InterpolatorFromName
from .orbifold import Orbifold
from .utils import sqeuclidean

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class Harmonics:
  def __init__(
      self,
      group:         SymmetryGroup,
      num_verts:     int,
      graph_method:  str,
      num_harmonics: int,
      use_cache:     bool = True,
      cache_dir:     str = '.symmetria.cache',
      **kwargs,
  ):
    self.group         = group
    self.num_verts     = num_verts
    self.graph_method  = graph_method
    self.num_harmonics = num_harmonics
    self.use_cache     = use_cache
    self.cache_dir     = cache_dir

  def disentangle(self, eval_thresh=0.01, num_pts=10000):

    if self.evals is None:
      raise Exception("Must call fit() before we can disentangle.")

    # Find (approximate) multiplicity groups.
    # Use relative difference between eigenvalues and then find connected
    # components.
    sqdists  = sqeuclidean(self.evals[:,jnp.newaxis])
    reldists = jnp.sqrt(sqdists) / self.evals[:,jnp.newaxis]
    similar  = jnp.abs(reldists) < eval_thresh
    graph    = nx.Graph(np.array(similar))
    groups   = list(map(lambda c: jnp.array(list(c)),
                        nx.connected_components(graph)))
    log.debug("Found %d multiplicity groups." % (len(groups)))

    # Get a bunch of points in the fundamnental region.
    pts = self.group.fund.get_points_random(num_pts)

    # Evaluate the harmonics.
    harmonics = self.interpolate()(pts)[0]

    self.evecs = jnp.array(self.evecs)

    # Loop over the multiplicity groups.
    for group in groups:
      if len(group) == 1:
        continue
      log.debug("Multiplicity group: %s" % (group))

      ica = skd.FastICA(n_components = len(group), whiten='unit-variance')
      ica.fit(harmonics[:,group])
      T = ica.components_

      # Polar decomposition.
      U, S, Vh = spla.svd(T)
      Q = U @ Vh

      self.evecs = self.evecs.at[:,group].set(self.evecs[:,group] @ Q.T)

      # TODO: recompute eigenvalues.

    return self.interpolate()

class HarmonicsGL(Harmonics):
  def __init__(
      self,
      epsilon: float,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.epsilon = epsilon
    self.evals   = None
    self.evecs   = None

  def interpolate(
      self,
      max_points:    int = 10000,
      method:        str = 'polyharm5',
      method_kwargs: dict = {},
  ):
    ''' Construct an interpolation function for the Laplacian eigenfunctions.

    Parameters
    ----------
    max_points : int
      Maximum number of points to be used in constructing the interpolator.
      Default is 10000.

    method : str
      The method to be used for interpolation. TODO: expand options.
      Default is 'polyharm5'.

    method_kwargs : dict
      Any additional arguments to be passed to the interpolator method.
      Default is {}.

    Returns
    -------
    interpolator : Callable[jnp.DeviceArray, [jnp.DeviceArray, jnp.DeviceArray]]
      Returns a function that maps points to their eigenfunctions and also
      returns an integer indicating which translation coset the point belongs
      to.  That is, which region of the unit cell the point lives in.
    '''
    if self.evals is None or self.evecs is None:
      raise Exception("Must call fit() before we can interpolate.")

    interp = InterpolatorFromName(method, **method_kwargs)

    interp.fit(self.verts, self.evecs)

    def _interpolator(x):
      quots, idents = self.group.quotient(x)
      return interp.interpolate(quots), idents

    return jax.jit(_interpolator)

  def _store(self):
    log.info("Caching results to %s." % (self.filename))
    np.savez_compressed(
      self.filename,
      verts = np.array(self.verts),
      evals = np.array(self.evals),
      evecs = np.array(self.evecs),
    )
    log.debug("Saved.")

  def _load(self):
    log.info("Attempting to load cached results from %s." % (self.filename))
    if os.path.exists(self.filename):
      cached = np.load(self.filename)
      self.verts = cached['verts']
      self.evals = cached['evals']
      self.evecs = cached['evecs']
      log.info("Successfully loaded from cache.")
    else:
      log.info("No cached data found.")

class HarmonicsGLSparse(HarmonicsGL):
  '''
  A class for computing space group harmonics using sparse graph Laplacians.

  ...

  Attributes
  ----------
  evals : jnp.DeviceArray
    The computed eigenvalues.

  evecs : jnp.DeviceArray
    The computed eigenvectors.

  verts : jnp.DeviceArray
    The locations of the vertices.

  filename : str
    The location of the cache file.

  Methods
  -------

  fit()
    Fit the eigenvalues and eigenvectors or load from cache as appropriate.

  interpolate(max_points=10000, method='polyharm2', method_kwargs={})
    Get an interpolation function for the eigenfunctions.

  '''

  def __init__(
      self,
      threshold: float = 5.0,
      **kwargs,
  ):
    '''
    Parameters
    ----------
    group : SymmetryGroup
      A SymmetryGroup object representing the plane or space group.

    num_verts : int
      The approximate number of vertices to use in the graph

    graph_method : str
      The method to use for placing vertices. 'mesh', 'sobol', or 'random'.
      Default is 'mesh'.

    epsilon : float
      The length scale of the edge weight kernel.

    num_harmonics : int
      The number of harmonics to compute.

    threshold : float
      The multiplier of epsilon for graph connectivity.
      Default is 5.0.

    use_cache : bool
      Whether to store and load eigenvalues and eigenvectors to file.
      Default is True.

    cache_dir : str
      Directory to store cache files.
      Default is .symmetria.cache.
    '''
    super().__init__(**kwargs)
    self.threshold = threshold

    if self.use_cache:
      self._load()

  @property
  def filename(self):
    fname = 'harmonics_%s_gl_sparse_%s_g%03d_v%d_e%f_n%d_t%f.npz' % (
      'plane' if self.group.dims == 2 else 'space',
      self.graph_method,
      self.group.number,
      self.num_verts,
      self.epsilon,
      self.num_harmonics,
      self.threshold,
    )
    return os.path.join(self.cache_dir, fname)

  def fit(self):
    ''' Finds the eigenvalues and eigenvectors of the graph Laplacian. '''

    if self.evals is None or self.evecs is None:

      self.graph = OrbitGraph(self.group, self.num_verts, self.graph_method)
      self.verts = self.graph.primal_verts

      t0 = time.time()
      L = self.graph.laplacian_sparse(self.epsilon, self.threshold)
      t1 = time.time()
      log.debug("Sparse Laplacian construction took %0.2f secs." % (t1-t0))
      log.info("Laplacian is %0.2f%% sparse." % (100*(1-L.nnz/L.shape[0]**2)))

      evals, evecs = sprsla.eigs(L, k=self.num_harmonics, which='SR')
      t2 = time.time()
      log.debug("Sparse eigensolve took %0.2f secs." % (t2-t1))

      evals = jnp.real(evals) * 2 / self.epsilon**2
      evecs = jnp.real(evecs) * 2 / self.epsilon**2

      # Sparse eigensolver does not return them sorted.
      order      = jnp.argsort(evals)
      self.evals = evals[order][:self.num_harmonics]
      self.evecs = evecs[:,order][:,:self.num_harmonics]

      if self.use_cache:
        self._store()

    else:
      log.debug("Eigenvalues and eigenvectors already computed. Skipping fit.")

    return self.evecs, self.evals

class HarmonicsGLDense(HarmonicsGL):
  '''
  A class for computing space group harmonics using ehse graph Laplacians.

  ...

  Attributes
  ----------
  evals : jnp.DeviceArray
    The computed eigenvalues.

  evecs : jnp.DeviceArray
    The computed eigenvectors.

  verts : jnp.DeviceArray
    The locations of the vertices.

  filename : str
    The location of the cache file.

  Methods
  -------

  fit()
    Fit the eigenvalues and eigenvectors or load from cache as appropriate.

  interpolate(max_points=10000, method='polyharm2', method_kwargs={})
    Get an interpolation function for the eigenfunctions.

  '''

  def __init__(
      self,
      **kwargs,
  ):
    '''
    Parameters
    ----------
    group : SymmetryGroup
      A SymmetryGroup object representing the plane or space group.

    num_verts : int
      The approximate number of vertices to use in the graph

    graph_method : str
      The method to use for placing vertices. 'mesh', 'sobol', or 'random'.
      Default is 'mesh'.

    epsilon : float
      The length scale of the edge weight kernel.

    use_cache : bool
      Whether to store and load eigenvalues and eigenvectors to file.
      Default is True.

    cache_dir : str
      Directory to store cache files.
      Default is .symmetria.cache.
    '''
    super().__init__(**kwargs)

    if self.use_cache:
      self._load()

  @property
  def filename(self):
    fname = 'harmonics_%s_gl_dense_%s_g%03d_v%d_e%f_n%d.npz' % (
      'plane' if self.group.dims == 2 else 'space',
      self.graph_method,
      self.group.number,
      self.num_verts,
      self.epsilon,
      self.num_harmonics,
    )
    return os.path.join(self.cache_dir, fname)

  def fit(self):
    ''' Finds the eigenvalues and eigenvectors of the graph Laplacian. '''

    if self.evals is None or self.evecs is None:

      self.graph = OrbitGraph(self.group, self.num_verts, self.graph_method)
      self.verts = self.graph.primal_verts

      t0 = time.time()
      L = self.graph.laplacian_dense(self.epsilon)
      t1 = time.time()
      log.debug("Dense Laplacian construction took %0.2f secs." % (t1-t0))

      evals, evecs = jnpla.eig(L)
      t2 = time.time()
      log.debug("Dense eigensolve took %0.2f secs." % (t2-t1))

      # Take real parts and rescale.
      self.evals = jnp.real(evals) * 2 / self.epsilon**2
      self.evecs = jnp.real(evecs) * 2 / self.epsilon**2

      self.evals = self.evals[:self.num_harmonics]
      self.evecs = self.evecs[:,:self.num_harmonics]

      if self.use_cache:
        self._store()

    else:
      log.debug("Eigenvalues and eigenvectors already computed. Skipping fit.")

    return self.evecs, self.evals

class HarmonicsRR(Harmonics):
  def __init__(
      self,
      num_basis     : int,
      mesh_size     : int,
      quad_deg      : int,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.num_basis     = num_basis
    self.mesh_size     = mesh_size
    self.quad_deg      = quad_deg
    self.evals         = None
    self.evecs         = None

    if self.use_cache:
      self._load()

  @property
  def filename(self):
    fname = 'harmonics_%s_rr_%s_g%03d_v%d_b%d_m%d_q%d_n%d.npz' % (
      'plane' if self.group.dims == 2 else 'space',
      self.graph_method,
      self.group.number,
      self.num_verts,
      self.num_basis,
      self.mesh_size,
      self.quad_deg,
      self.num_harmonics,
    )
    return os.path.join(self.cache_dir, fname)

  def _store(self):
    log.info("Caching results to %s." % (self.filename))
    np.savez_compressed(
      self.filename,
      evals   = np.array(self.evals),
      evecs   = np.array(self.evecs),
    )

  def _load(self):
    log.info("Attempting to load cached results from %s." % (self.filename))
    if os.path.exists(self.filename):
      cached = np.load(self.filename)
      self.evals = cached['evals']
      self.evecs = cached['evecs']
      log.info("Successfully loaded from cache.")
    else:
      log.info("No cached data found.")

  def fit(self):

    # First, get an orbifold map.
    O = Orbifold(
      group        = self.group,
      num_verts    = self.num_verts,
      graph_method = self.graph_method,
      embed_dims   = 0,
      use_cache    = self.use_cache,
      cache_dir    = self.cache_dir,
    )
    O.fit()
    embedder = O.interpolate(method='polyharm5')
    log.debug("Orbifold map constructed.")

    # Get quadrature points and weights.
    quad_pts, quad_wts = self.group.fund.get_quad_scheme(
      self.quad_deg, self.mesh_size)
    log.debug("Using %d quadrature points." % (quad_pts.shape[0]))

    # TODO: more interesting configuration of basis functions.

    # Get centers in fundamental region.
    centers = self.group.fund.get_points_sobol(self.num_basis)
    basis_centers = centers @ self.group.basic_basis.T
    log.debug("Using %d basis functions." % (centers.shape[0]))

    # Push through to embedding space.
    self.emb_centers, _ = embedder(basis_centers)
    num_basis = self.emb_centers.shape[0]

    # Build the basis functions and their gradients.
    def emb_basis_func(z, center):
      r2 = jnp.sum((z - center)**2)
      return 0.5 * r2 * jnp.where(r2>0, jnp.log(r2), 0)

    self.basis_func = jax.jit(
      lambda x, c: emb_basis_func(embedder(x)[0], c)
    )

    if self.evals is None or self.evecs is None:

      grad_basis = jax.jit(jax.grad(self.basis_func))

      grad_centers = jax.jit(jax.vmap(grad_basis, in_axes=(None,0)))

      # We need to compute a matrix that is num_basis x num_basis that arises
      # from integrating the inner product between each basis function over the
      # domain.  We're doing quadrature, so this corresponds to a big contraction.
      # However, that uses a ton of memory so we're going to do it with a scan.
      # It would probably be much smarter to do a scan over basis functions than
      # a scan over quadrature points.
      @jax.jit
      def scan_func(carry, xs):
        quad_pt, quad_wt = xs
        gradients = grad_centers(quad_pt, self.emb_centers)
        return (quad_wt * gradients) @ gradients.T + carry, None

      t0 = time.time()
      A, _ = jax.lax.scan(
        scan_func,
        jnp.zeros((num_basis, num_basis)),
        (quad_pts, quad_wts),
      )
      t1 = time.time()
      log.debug("Computed A matrix in %0.2f secs." % (t1-t0))

      basis_matrix = jax.vmap(
        jax.vmap(
          self.basis_func,
          in_axes=(0,None),
        ),
        in_axes=(None,0),
      )(quad_pts, self.emb_centers)

      B = (basis_matrix * quad_wts[jnp.newaxis,:]) @ basis_matrix.T
      t2 = time.time()
      log.debug("Computed B matrix in %0.2f secs." % (t2-t1))

      evals, evecs = spla.eigh(A, B)
      t3 = time.time()
      log.debug("Performed eigendecomposition in %0.2f secs." % (t3-t2))

      self.evals = evals[:self.num_harmonics]
      self.evecs = evecs[:,:self.num_harmonics]

      if self.use_cache:
        self._store()

  def interpolate(self):
    if self.evals is None or self.evecs is None:
      raise Exception("Must call fit() before we can interpolate.")

    def _interpolator(x):
      x = jnp.atleast_2d(x)
      basis_matrix = jax.vmap(
        jax.vmap(
          self.basis_func,
          in_axes=(0,None),
        ),
        in_axes=(None,0),
      )(x, self.emb_centers)

      # FIXME identities
      return basis_matrix.T @ self.evecs, None

    return _interpolator
