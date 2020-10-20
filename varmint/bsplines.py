import jax
import jax.numpy as np

from functools import partial

@jax.jit
def divide00(num, denom):
  ''' Divide such that 0/0 = 0. '''
  force_zero = np.logical_and(num==0, denom==0)
  return np.where(force_zero, 0, num) / np.where(force_zero, 1, denom)

@partial(jax.jit, static_argnums=(2,))
def bspline1d_basis(u, knots, degree):
  ''' Computes b-spline basis functions in one dimension.

  Parameters:
  -----------

   - u: The locations at which to evaluate the basis functions, generally
        assumed to be in the interval [0,1) although they really just need to
        be consistent with the knot ranges. Can be an ndarray of any size.

   - knots: The knot vector. Should be non-decreasing and consistent with the
            specified degree.  A one-dimensional ndarray.

   - degree: The degree of the piecewise polynomials. Integer.

  Returns:
  --------
   Returns an ndarray whose first dimensions are the same as u, but with an
   additional dimension determined by the number of basis functions, i.e.,
   one less than the number of knots minus the degree.

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
    B = bspline1d_basis(u, knots, degree-1)

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

@partial(jax.jit, static_argnums=(3,))
def bspline2d_basis(u, xknots, yknots, degree):
  ''' Compute b-spline basis functions in two dimensions as tensor products.

  Parameters:
  -----------

   - u: The locations at which to evaluate the basis functions, generally
        assumed to be in the interval [0,1) although they really just need to
        be consistent with the knot ranges.  Must be a two-dimensional ndarray
        where the second dimension has length two.  The first dimension (N)
        corresponds to the number of points at which the basis functions should
        be evaluated.

   - xknots: The knot vector for the first dimension. Should be non-decreasing
             and consistent with the specified degree.  A one-dimensional
             ndarray of length J.

   - yknots: The knot vector for the second dimension. Should be non-decreasing
             and consistent with the specified degree.  A one-dimensional
             ndarray of length K.

   - degree: The degree of the piecewise polynomials. Integer.

  Returns:
  --------
   Returns an ndarray of dimension N x J x K, where ret[n,:,:] is the tensor
   product of basis functions for x and y, evaluated at u[n,:].
  '''

  basis_x  = np.expand_dims(bspline1d_basis(u[:,0], xknots, degree), -1)
  basis_y  = np.expand_dims(bspline1d_basis(u[:,1], yknots, degree), -2)
  basis_xy = basis_x * basis_y

  return basis_xy

@partial(jax.jit, static_argnums=(4,))
def bspline3d_basis(u, xknots, yknots, zknots, degree):
  ''' Compute b-spline basis functions in three dimensions as tensor products.

  Parameters:
  -----------

   - u: The locations at which to evaluate the basis functions, generally
        assumed to be in the interval [0,1) although they really just need to
        be consistent with the knot ranges.  Must be a two-dimensional ndarray
        where the second dimension has length three.  The first dimension (N)
        corresponds to the number of points at which the basis functions should
        be evaluated.

   - xknots: The knot vector for the first dimension. Should be non-decreasing
             and consistent with the specified degree.  A one-dimensional
             ndarray of length J.

   - yknots: The knot vector for the second dimension. Should be non-decreasing
             and consistent with the specified degree.  A one-dimensional
             ndarray of length K.

   - zknots: The knot vector for the third dimension. Should be non-decreasing
             and consistent with the specified degree.  A one-dimensional
             ndarray of length L.

   - degree: The degree of the piecewise polynomials. Integer.

  Returns:
  --------
   Returns an ndarray of dimension N x J x K x L, where ret[n,:,:,:] is the
   tensor product of basis functions for x, y, and z, evaluated at u[n,:].
  '''

  basis_x  = bspline1d_basis(u[:,0], xknots, degree)[:,:,np.newaxis,np.newaxis]
  basis_y  = bspline1d_basis(u[:,1], yknots, degree)[:,np.newaxis,:,np.newaxis]
  basis_z  = bspline1d_basis(u[:,2], zknots, degree)[:,np.newaxis,np.newaxis,:]

  basis_xyz = basis_x * basis_y * basis_z

  return basis_xyz
