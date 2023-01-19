import jax.numpy as jnp

from typing        import Union
from cctbx.sgtbx   import plane_groups, space_group
from cctbx.uctbx   import unit_cell
from functools     import cached_property

from .group        import SymmetryGroup, basic_unit_cells
from ..fundamental import PlaneFundamentalRegion
from ..utils       import scitbx2jax
from ..projective  import up, dn

class PlaneGroup(SymmetryGroup):
  def __init__(self, group_id: Union[int, str], **kwargs) -> None:
    try:
      if group_id in self.hmh2index:
        self.hmh = group_id
      else:
        self.hmh = self.hmh_table[group_id]
    except:
      print(self.hmh_table)
      raise Exception('Group not found')

    # Basic information about the group.
    self._sg = space_group(self.hmh)
    self._cs = self._sg.crystal_system()
    self._uc = unit_cell(basic_unit_cells[self._cs])

    self.fund = PlaneFundamentalRegion(self)

    super().__init__(dims=2, **kwargs)

  @cached_property
  def number(self) -> int:
    return self.hmh2index[self.hmh]

  @cached_property
  def hmh_table(self):
    by_string = list(map(
      lambda x: (x[0].replace('_', ''), x[1]),
      plane_groups.hermann_mauguin_hall_table,
    ))
    by_index_str = list(zip(
      map(str, range(1,18)),
      map(
        lambda x: x[1],
        plane_groups.hermann_mauguin_hall_table,
      )
    ))
    by_index_int = list(zip(
      range(1,18),
      map(
        lambda x: x[1],
        plane_groups.hermann_mauguin_hall_table,
      )
    ))
    return dict(by_string + by_index_str + by_index_int)

  @cached_property
  def hmh2index(self):
    return dict(zip(
      map(
        lambda x: x[1],
        plane_groups.hermann_mauguin_hall_table,
      ),
      range(1,18),
    ))

  @cached_property
  def index(self):
    return self.hmh2index[self.hmh]

  @cached_property
  def name(self):
    return plane_groups.hermann_mauguin_hall_table[self.index-1][0] \
                       .replace('_', '')

  @cached_property
  def operations(self) -> jnp.DeviceArray:
    op_matrices = []
    for op in self._sg.all_ops(mod=self.op_mod):
      mat = scitbx2jax(op.as_4x4_rational())

      # Remove the z dimension.
      mat = jnp.delete(jnp.delete(mat, 2, axis=0), 2, axis=1)

      op_matrices.append(mat)

    return jnp.stack(op_matrices, axis=0)

  @cached_property
  def basic_basis(self) -> jnp.DeviceArray:
    return jnp.array(self._uc.orthogonalization_matrix()).reshape(3,3)[:2,:2]

  def is_reflection(self, op):
    op_locs = dn(up(self.fund.vertices) @ op.T)
    close_verts = jnp.sum(
      jnp.all(jnp.isclose(self.fund.vertices, op_locs), axis=1)
    )
    # More than one (so an edge) but not all (so not identity).
    return close_verts > 1 and close_verts < self.fund.vertices.shape[0]
