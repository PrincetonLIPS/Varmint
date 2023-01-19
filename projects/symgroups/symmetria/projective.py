import jax
import jax.numpy as jnp

@jax.jit
def up(points: jnp.DeviceArray) -> jnp.DeviceArray:
  ''' Put points into homogenous coordinates.

  This just adds a one to the last dimension.

  Args:
    points (jnp.DeviceArray): (shape)xD array where D is 2 or 3.

  Returns:
    homog_points (jnp.DeviceArray): (shape)x(D+1) with ones appended.
  '''
  return jnp.concatenate(
    [
      points,
      jnp.ones(points.shape[:-1]+(1,)),
    ],
    axis=-1,
  )

@jax.jit
def dn(homog_points: jnp.DeviceArray) -> jnp.DeviceArray:
  ''' Take points out of homogenous coordinates.

  This just removes the last dimension.

  Args:
    homog_points (jnp.DeviceArray): (shape)x(D+1) array where D is 2 or 3.

  Returns:
    points (jnp.DeviceArray): (shape)xD with ones removed.
  '''
  return homog_points[...,:-1]
