from typing import Union

import re
import jax.numpy as jnp

from functools   import cached_property
from cctbx.uctbx import unit_cell
from cctbx.sgtbx import space_group_info

from .group        import SymmetryGroup, basic_unit_cells
from ..fundamental import SpaceFundamentalRegion
from ..utils       import scitbx2jax
from ..projective  import up, dn


class SpaceGroup(SymmetryGroup):

  def __init__(self, group_id: Union[int, str], **kwargs) -> None:

    # Basic info about the space group.
    self._sg = space_group_info(group_id).group()
    self._cs = self._sg.crystal_system()
    self._uc = unit_cell(basic_unit_cells[self._cs])

    # Get the asymmetric (fundamental) unit.
    self.fund = SpaceFundamentalRegion(self)

    super().__init__(dims=3, **kwargs)

  @cached_property
  def number(self) -> int:
    return self._sg.info().type().number()

  @cached_property
  def operations(self) -> jnp.DeviceArray:
    op_matrices = []
    for op in self._sg.all_ops(mod=self.op_mod):
      mat = scitbx2jax(op.as_4x4_rational())

      op_matrices.append(mat)

    return jnp.stack(op_matrices, axis=0)

  @cached_property
  def basic_basis(self) -> jnp.DeviceArray:
    return jnp.array(self._uc.orthogonalization_matrix()).reshape(3,3)

  @cached_property
  def name(self):
    return re.sub(
      r'\(.*\)',
      '',
      self._sg.info().symbol_and_number(),
    ).replace(' ', '')

  def is_reflection(self, op):
    op_locs = dn(up(self.fund.vertices) @ op.T)
    close_verts = jnp.sum(
      jnp.all(jnp.isclose(self.fund.vertices, op_locs), axis=1)
    )
    # More than two (so a face) but not all (so not identity).
    return close_verts > 2 and close_verts < self.fund.vertices.shape[0]
