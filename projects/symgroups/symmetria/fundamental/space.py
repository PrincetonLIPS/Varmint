from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
  from ..groups import SpaceGroup

import jax
import jax.numpy as jnp
import numpy as np
import quadpy
import meshpy.tet as mpt

from scipy.spatial import ConvexHull
from cctbx.sgtbx.direct_space_asu import (
  reference_table as sg_asu_table,
)

from ..utils import tet_volume
from .region import FundamentalRegion

class SpaceFundamentalRegion(FundamentalRegion):
  def __init__(self, space_group: SpaceGroup, **kwargs) -> None:
    super().__init__(dims=3, **kwargs)
    self.sg   = space_group
    self._asu = sg_asu_table.get_asu(space_group.number)

  def get_quad_scheme(self, degree, num_regions):

    # Need to do this in the basis space.
    basis_verts = self.vertices @ self.sg.basic_basis.T

    hull = ConvexHull(np.array(basis_verts))

    info = mpt.MeshInfo()
    info.set_points(hull.points)
    info.set_facets(hull.simplices)

    max_volume = hull.volume / num_regions
    mesh       = mpt.build(info, max_volume=max_volume)

    mesh_points = jnp.array(mesh.points)
    mesh_tets   = jnp.array(mesh.elements)
    volumes = jax.vmap(
      lambda mt: tet_volume(mesh_points[mt,:]),
    )(mesh_tets)

    scheme = quadpy.t3.get_good_scheme(degree)
    scheme_pts = jnp.array(scheme.points)
    scheme_wts = jnp.array(scheme.weights)

    quad_pts = jax.vmap(
      lambda mt: scheme_pts.T @ mesh_points[mt,:],
      in_axes=0,
    )(mesh_tets)

    quad_wts = scheme_wts[:,jnp.newaxis] * volumes[jnp.newaxis,:]

    return quad_pts.reshape(-1,3), quad_wts.ravel()
