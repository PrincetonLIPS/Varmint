import logging
import networkx         as nx
import jax
import jax.numpy        as jnp
import jax.numpy.linalg as jnpla

from scipy.stats.qmc import Sobol

from ..utils      import scitbx2jax
from ..projective import up, dn

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

basic_unit_cells = {
  'Triclinic':    (1, 1, 1, 90, 90, 90),
  'Monoclinic':   (1, 1, 1, 90, 90, 90),
  'Orthorhombic': (1, 1, 1, 90, 90, 90),
  'Tetragonal':   (1, 1, 1, 90, 90, 90),
  'Trigonal':     (1, 1, 1, 90, 90, 120),
  'Hexagonal':    (1, 1, 1, 90, 90, 120),
  'Cubic':        (1, 1, 1, 90, 90, 90),
}

@jax.jit
def _apply_ops(ops, points):
  up_points = up(points)
  return dn(jnp.stack([up_points @ op.T for op in ops]))

@jax.jit
def mult_mod(op1, op2):
  prod = op1 @ op2
  return prod.at[:-1,-1].set(prod[:-1,-1] % 1.0)

@jax.jit
def mult_pairs(op1, op2):
  return jax.vmap(
    jax.vmap(
      mult_mod,
      in_axes=(0, None),
    ),
    in_axes=(None,0),
  )(op1,op2)

class SymmetryGroup:
  def __init__(self, dims: int, **kwargs) -> None:
    super().__init__(**kwargs)

    self.dims   = dims
    self.op_mod = 1

    self.qr = self._build_quotient_resolver()

    log.info("Group %s (%d)" % (self.name, self.number))

  def cell_orbit(self, points):
    ''' Apply all non-translation group actions. '''
    return _apply_ops(self.operations, points)

  def orbit(self, points, trans):

    # Get the within-cell orbits.
    local_orbit = self.cell_orbit(points)

    # Get all the translations.
    grid    = jnp.concatenate([jnp.arange(trans+1), jnp.arange(-trans,0)])
    offsets = jnp.stack(jnp.meshgrid(*[grid]*self.dims), axis=-1)

    # Expand local orbits to include all translations.
    return jnp.expand_dims(local_orbit, tuple(range(self.dims))) \
      + jnp.expand_dims(offsets, (-2, -3))

  def _build_quotient_resolver(self):

    # The goal here is to construct a set of operations such that we have
    # completely covered the unit cell.  The challenge is that the unit cell
    # might have a weird shape.  So we expand our fundamental regions into
    # neighboring cells.  We can do this with DeviceArrays.
    grid    = jnp.array([-1, 0, 1])
    offsets = jnp.stack(
      jnp.meshgrid(*[grid]*self.dims),
      axis=-1,
    ).reshape(-1,self.dims)
    t_mats  = [
      jnp.eye(self.dims+1).at[:-1,-1].set(offset) \
      for offset in offsets
    ]

    # Get all possible forward ops and then invert them.
    # Could definitely do this in a cleverer way.
    rev_ops = []
    op_ids  = []
    for t_mat in t_mats:
      for ii, op_mat in enumerate(self.operations):
        rev_ops.append(jnpla.inv(t_mat @ op_mat))
        op_ids.append(ii)
    rev_ops = jnp.array(rev_ops)
    op_ids  = jnp.array(op_ids)

    # Build a test to see if the point maps back and where.
    cut_normals   = self.fund.cut_normals
    cut_distances = self.fund.cut_distances
    @jax.jit
    def _test_op(op, X):
      rev_X = X @ op.T
      inside = jnp.all(
        rev_X[:,:self.dims] @ cut_normals >= -cut_distances,
        axis=1,
      )
      return inside, rev_X

    test_ops = jax.jit(jax.vmap(_test_op, in_axes=(0,None)))

    # Get a zillion random samples.
    # We don't use a grid here becasuse it would be confusing if points
    # landed on faces shared bewteen adjacent fundamental regions.
    # Put them into homogenous coordinates.
    sampler = Sobol(d=self.dims, scramble=True, seed=1)
    samp_X  = jnp.column_stack([
      jnp.array(sampler.random_base2(m=13)),
      jnp.ones((2**13,1)),
    ])

    # The game is, apply each reverse operation to the unit cell.
    # Test if _any_ of the resulting points land in the fundamental region.
    # If any of them do, then we have a useful transformation to keep around.
    useful = jnp.any(test_ops(rev_ops, samp_X)[0], axis=1)
    good_ops = rev_ops[useful,...]
    good_ids = op_ids[useful]

    # TODO: probably much faster to apply the forward operations to the
    # vertices of the fundamental region and see if any of them are in the
    # unit cell.  Would create some boundary annoyance, but seems more elegant.

    inv_basis = jnpla.inv(self.basic_basis)

    @jax.jit
    def _loop_bodyfun(ii, value):
      indices, fund_X, X = value

      is_inside, loc_X = _test_op(good_ops[ii,:,:], X)

      # Update indices to catch inside cases.
      indices = jnp.where(is_inside, ii, indices)

      # Update locations.
      fund_X = jnp.where(is_inside[:,jnp.newaxis], loc_X, fund_X)

      return (indices, fund_X, X)

    # vmap will run out of memory, so we loop
    def _map_to_fundamental_loop(x):

      x = jnp.atleast_2d(x)

      # Homogenous coordinates.
      X = jnp.column_stack([
        jnp.atleast_2d((x @ inv_basis.T) % 1.0),
        jnp.ones((x.shape[0],1)),
      ])

      # Loop over possible quotient locations.
      # The jitter will unroll this...
      # Make this a scan?
      indices = -1 * jnp.ones(x.shape[0], dtype=jnp.int32)
      fund_X  = jnp.zeros_like(X)
      indices, fund_X, _ = jax.lax.fori_loop(
        0,
        good_ops.shape[0],
        _loop_bodyfun,
        (indices, fund_X, X),
      )

      return fund_X[:,:self.dims], good_ids[indices]

    return jax.jit(_map_to_fundamental_loop)

  def quotient(self, points: jnp.DeviceArray) -> jnp.DeviceArray:
    return self.qr(points)

  def reflection_generators(self, mod=False):
    grid    = jnp.array([0, -1, 1, -2, 2])
    offsets = jnp.stack(
      jnp.meshgrid(*[grid]*self.dims),
      axis=-1,
    ).reshape(-1,self.dims)
    t_mats  = [
      jnp.eye(self.dims+1).at[:-1,-1].set(offset) \
      for offset in offsets
    ]

    # Look at all possible ops.
    # TODO: vmap this slow-ass thing.
    reflect_ops = []
    for t_mat in t_mats:
      for ii, op_mat in enumerate(self.operations):
        new_op = t_mat @ op_mat
        if self.is_reflection(new_op):
          reflect_ops.append((ii, new_op, op_mat, t_mat))

    # Collect based on their ids.
    aggregated = {}
    for ii, new_op, op_mat, t_mat in reflect_ops:
      if ii not in aggregated:
        aggregated[ii] = (ii, new_op, op_mat, t_mat)
      elif jnp.sum(jnp.abs(t_mat)) < jnp.sum(jnp.abs(aggregated[ii][3])):
        # Minimum translation.
        aggregated[ii] = (ii, new_op, op_mat, t_mat)

    # Filter out just the ops.
    if mod:
      ref_ops = [self.mod_op(agg[1]) for agg in aggregated.values()]
    else:
      ref_ops = [agg[1] for agg in aggregated.values()]

    if len(ref_ops) == 0:
      return jnp.eye(self.dims+1)[jnp.newaxis,:,:]

    return jnp.stack(ref_ops, axis=0)

  def get_generators(self, primitive=True):
    gen_set = self._sg.info().any_generator_set()
    if primitive:
      gens = gen_set.primitive_generators
    else:
      gens = gen_set.non_primitive_generators

    return [scitbx2jax(g.as_4x4_rational()) for g in gens]

  def mod_op(self, op):
    return op.at[:self.dims,-1].set(op[:self.dims,-1] % 1.0)

  def generate_group(self, gen_ops):
    gen_group = gen_ops
    num_new = gen_group.shape[0]
    while num_new > 0:
      new_ops = jnp.unique(
        mult_pairs(gen_group, gen_group).reshape(-1,self.dims+1,self.dims+1),
        axis=0,
      )
      is_new  = jnp.all(jnp.logical_not(jnp.all(
        jnp.isclose(
          new_ops[:,jnp.newaxis,:,:],
          gen_group[jnp.newaxis,:,:,:],
        ),
        axis=(2,3))), axis=1)
      num_new = jnp.sum(is_new)
      gen_group = jnp.concatenate([gen_group, new_ops[is_new,:,:]], axis=0)
    return gen_group

  def all_reflections(self):

    # Generating set.
    gen_ref = self.reflection_generators(mod=True)

    # Get the entire reflective subgroup.
    subgroup_ref = self.generate_group(gen_ref)

    return subgroup_ref

  def factor_reflections(self):
    ref_subgroup = self.all_reflections()

    # Find equivalence classes.
    inv_ops = jax.vmap(jnpla.inv, in_axes=0)(self.operations)
    gig_pairs = mult_pairs(self.operations, inv_ops)
    equiv = jnp.any(
      jnp.all(
        jnp.isclose(
          gig_pairs[jnp.newaxis,:,:,:,:],
          ref_subgroup[:,jnp.newaxis,jnp.newaxis,:,:],
        ),
        axis=(-1,-2),
      ),
      axis=0,
    )

    G = nx.from_numpy_matrix(equiv - jnp.eye(equiv.shape[0]))
    cc = list(map(set.pop, nx.connected_components(G)))
    factor = jnp.stack([self.operations[comp] for comp in cc], axis=0)

    return ref_subgroup, factor
