from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
  from ..groups import PlaneGroup

import quadpy
import meshpy.triangle as mpt
import jax
import jax.numpy as jnp
import numpy as np

from scipy.spatial import ConvexHull
from cctbx.sgtbx.direct_space_asu import (
  plane_group_reference_table as pg_asu_table,
)

from ..utils import triangle_area
from .region import FundamentalRegion

class PlaneFundamentalRegion(FundamentalRegion):
  def __init__(self, plane_group: PlaneGroup, **kwargs):
    super().__init__(dims=2, **kwargs)
    self.sg = plane_group
    self._asu = pg_asu_table.get_asu(plane_group.number)

  def get_quad_scheme(self, degree, num_regions):

    # Need to do this in the basis space.
    basis_verts = self.vertices @ self.sg.basic_basis.T

    hull = ConvexHull(np.array(basis_verts))

    info = mpt.MeshInfo()
    info.set_points(hull.points)
    info.set_facets(hull.simplices)

    max_area = hull.volume / num_regions
    mesh = mpt.build(info, max_volume=max_area, min_angle=25)

    mesh_points = jnp.array(mesh.points)
    mesh_tris   = jnp.array(mesh.elements)

    areas = jax.vmap(
      lambda mt: triangle_area(mesh_points[mt,:]),
      in_axes=0,
    )(mesh_tris)

    scheme = quadpy.t2.get_good_scheme(degree)
    scheme_pts = jnp.array(scheme.points)
    scheme_wts = jnp.array(scheme.weights)

    quad_pts = jax.vmap(
      lambda mt: scheme_pts.T @ mesh_points[mt,:],
      in_axes=0,
    )(mesh_tris)

    quad_wts = scheme_wts[:,jnp.newaxis] * areas[jnp.newaxis,:]

    return quad_pts.reshape(-1,2), quad_wts.ravel()
