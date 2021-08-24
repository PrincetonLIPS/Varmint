import jax
import jax.numpy        as np
import jax.numpy.linalg as npla

vmap_dot = jax.jit(
  jax.vmap(
    np.dot,
    in_axes=(0, 0),
  ),
)

vmap_lsolve = jax.jit(
  jax.vmap(
    npla.solve,
    in_axes=(0,0),
  ),
)

vmap_rsolve = jax.jit(
  jax.vmap(
    lambda A, B: npla.solve(B.T, A.T).T,
    in_axes=(0,0),
  ),
)

vmap_tsolve = jax.jit(
  jax.vmap(
    lambda A, B: npla.solve(A.T, B),
    in_axes=(0,0),
  ),
)

vmap_tensordot = jax.jit(
  jax.vmap(
    np.tensordot,
    in_axes=(0, 0, None),
  ),
  static_argnums=(2,)
)

vmap_inv = jax.jit(
  jax.vmap(
    npla.inv,
    in_axes=(0,),
  ),
)

vmap_det = jax.jit(
  jax.vmap(
    npla.det,
    in_axes=(0,),
  ),
)

vmap_diag = jax.jit(
  jax.vmap(
    np.diag,
    in_axes=(0,),
  ),
)