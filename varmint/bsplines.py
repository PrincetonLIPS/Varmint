import jax
import jax.numpy as np

from functools import partial

@jax.jit
def divide00(num, denom):
  ''' Divide such that 0/0 = 0. '''
  force_zero = np.logical_and(num==0, denom==0)
  return np.where(force_zero, 0, num) / np.where(force_zero, 1, denom)

@partial(jax.jit, static_argnums=(2,))
def bspline_basis(u, knots, degree):
  '''Computes b-spline basis functions.

     u: The locations at which to evaluate the basis functions.
        Can be a numpy array of any size.

     knots: The knot vector. Should be non-decreasing and consistent with the
            specified

     degree: The degree of the piecewise polynomials.
  '''
  u1d = np.atleast_1d(u)

  # Determine the target size of the returned object.
  num_basis_funcs = knots.shape[0]-degree-1
  ret_shape = u1d.shape + (num_basis_funcs,)

  # Set things up for broadcasting.
  # Append a singleton dimension onto the u points.
  # Prepend the correct number of singleton dimensions onto knots.
  # The vars are named 2d but they could be bigger.
  u2d = np.expand_dims(u1d, -1)
  k2d = np.expand_dims(knots, tuple(range(len(u1d.shape))))

  if degree == 0:

    # The degree zero case is just the indicator function on the
    # half-open interval specified by the knots.
    return (k2d[...,:-1] <= u2d) * (u2d < k2d[...,1:])

  else:

    # Take advantage of the recursive definition.
    B = bspline_basis(u, knots, degree-1)

    # There are two halves.  The indexing is a little tricky.
    # Also we're using the np.divide 'where' argument to deal
    # with the fact that we want 0/0 to equal zero.
    # This all computes much more than we strictly need, because
    # so much of this is just multiplying zero by things.
    # However, I think the vectorized implementation is worth
    # it for using things like JAX and GPUs.
    v0_num   = B[...,:-1] * (u2d - k2d[...,:-degree-1])
    v0_denom = k2d[...,degree:-1] - k2d[...,:-degree-1]
    v0       = divide00(v0_num, v0_denom)

    v1_num   = B[...,1:] * (k2d[...,degree+1:] - u2d)
    v1_denom = k2d[...,degree+1:] - k2d[...,1:-degree]
    v1       = divide00(v1_num, v1_denom)

    return v0 + v1
