from __future__ import annotations

import logging
import jax
import numpy as np
import os
import time

from .groups import SymmetryGroup
from .graph  import OrbitGraph
from .mds    import MDS

from .interpolate import InterpolatorFromName

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class Orbifold:
  def __init__(
      self,
      group:        SymmetryGroup,
      num_verts:    int,
      graph_method: str,
      embed_dims:   int = 0,
      use_cache:    bool = True,
      cache_dir:    str = '.symmetria.cache',
  ):
    ''' A class for building and using the orbifold map for the plane and space
      groups.

    Args:
      group (SymmetryGroup): A SymmetryGroup object representing the plane or
        space group.

      num_verts (int): The approximate number of vertices to use in the graph.

      graph_method (str): The method to use for placing vertices. 'mesh',
        'sobol', or 'random'. Default is 'mesh'.

      embed_dims (int): The number of dimensions of the embedding space. 0 means
        figure it out based on minimizing the error in the distance. Default is 0.

      use_cache (bool): Whether to store and load eigenvalues and eigenvectors to
        file. Default is True.

      cache_dir (str): Directory to store cache files. Default is
        .symmetria.cache.

    '''
    self.group        = group
    self.num_verts    = num_verts
    self.graph_method = graph_method
    self.embed_dims   = embed_dims
    self.use_cache    = use_cache
    self.cache_dir    = cache_dir
    self.embeddings   = None

    if self.use_cache:
      self._load()

  @property
  def _filename(self):
    ''' Get a good filename for caching. '''
    fname = 'orbifold_%s_%s_g%03d_v%d_e%d.npz' % (
      'plane' if self.group.dims == 2 else 'space',
      self.graph_method,
      self.group.number,
      self.num_verts,
      self.embed_dims,
    )
    return os.path.join(self.cache_dir, fname)

  def _store(self):
    ''' Store embeddings to file. '''
    log.info("Caching results to %s." % (self._filename))
    np.savez_compressed(
      self._filename,
      verts      = np.array(self.verts),
      embeddings = np.array(self.embeddings)
    )
    log.debug("Saved.")

  def _load(self):
    ''' Load embeddings from file. '''
    log.info("Attempting to load cached results from %s." % (self._filename))
    if os.path.exists(self._filename):
      cached = np.load(self._filename)
      self.verts      = cached['verts']
      self.embeddings = cached['embeddings']
      log.info("Successfully loaded from cache.")
    else:
      log.info("No cached data found.")

  def fit(self):
    ''' Compute the orbifold embeddings.

    Returns:
      verts (jnp.DeviceArray): Array of vertex locations in the fundamental
        region.

      embeddings (jnp.DeviceArray): Embeddings for each vertex.

    '''

    if self.embeddings is None:
      self.graph = OrbitGraph(self.group, self.num_verts, self.graph_method)
      self.verts = self.graph.primal_verts

      t0 = time.time()
      dists = self.graph.distances_dense()
      t1 = time.time()
      log.debug("Computed distances in %0.2f secs." % (t1-t0))

      self.embeddings = MDS(dists, self.embed_dims)
      t2 = time.time()
      log.debug("Performed MDS in %0.2f secs." % (t2-t1))

      if self.use_cache:
        self._store()

    else:
      log.debug("Embeddings already computed. Skipping fit.")

    return self.verts, self.embeddings

  def interpolate(
      self,
      max_points:    int = 10000,
      method:        str = 'polyharm2',
      method_kwargs: dict = {},
  ):
    ''' Construct an interpolation function for the orbifold embedding.

    Args:
      max_points (int): Maximum number of points to be used in constructing the
        interpolator.  Default is 10000.

      method (str): The method to be used for interpolation. TODO: expand
        options. Default is 'polyharm2'.

      method_kwargs (dict): Any additional arguments to be passed to the
        interpolator method.  Default is {}.

    Returns:
      interpolator (Callable):
        A function that maps points to their location on the orbifold and also
        returns an integer indicating which translation coset the point belongs
        to.  That is, which region of the unit cell the point lives in.
    '''
    if self.embeddings is None:
      raise Exception("Must call fit() before we can interpolate.")

    interp = InterpolatorFromName(method, **method_kwargs)

    interp.fit(self.verts, self.embeddings)

    def _interpolator(x):
      quots, idents = self.group.quotient(x)
      return interp.interpolate(quots), idents

    return jax.jit(_interpolator)
