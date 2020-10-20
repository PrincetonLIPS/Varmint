import jax
import jax.numpy    as np
import numpy.random as npr
import timeit

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
  u2d = np.atleast_2d(u)

  basis_x  = bspline1d_basis(u2d[:,0], xknots, degree)[:,:,np.newaxis]
  basis_y  = bspline1d_basis(u2d[:,1], yknots, degree)[:,np.newaxis,:]
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

@partial(jax.jit, static_argnums=(3,))
def bspline1d(u, control, knots, degree):
  ''' Evaluate a one-dimensional bspline function. '''
  return bspline1d_basis(u, knots, degree) @ control

@partial(jax.jit, static_argnums=(4,))
def bspline2d(u, control, xknots, yknots, degree):
  ''' Evaluate a two-dimensional bspline function. '''

  basis_xy = bspline2d_basis(u, xknots, yknots, degree)

  # This is 3x slower for reasons I don't understand.
  return np.tensordot(basis_xy, control, ((1,2), (0,1)))

@partial(jax.jit, static_argnums=(5,))
def bspline3d(u, control, xknots, yknots, zknots, degree):
  ''' Evaluate a three-dimensional bspline function. '''

  basis_xyz = bspline3d_basis(u, xknots, yknots, zknots, degree)

  # This is 3x slower for reasons I don't understand.
  return np.tensordot(basis_xyz, control, ((1,2,3), (0,1,2)))

@partial(jax.jit, static_argnums=(2,))
def bspline1d_basis_derivs(u, knots, degree):
  ''' Derivatives of basis functions '''
  d_basis = bspline1d_basis(u, knots, degree-1)
  v0 = divide00(d_basis[:,:-1] * degree, knots[degree:-1] - knots[:-degree-1])
  v1 = divide00(d_basis[:,1:] * degree, knots[degree+1:] - knots[1:-degree])
  return v0 - v1

def bspline1d_derivs(u, control, knots, degree):
  ''' Evaluate the derivative of a one-dimensional bspline function. '''
  return bspline1d_basis_derivs(u, knots, degree) @ control

partial(jax.jit, static_argnums=(2,))
def bspline2d_basis_derivs(u, xknots, yknots, degree):
  ''' Derivatives of 2d basis functions '''
  u2d = np.atleast_2d(u)

  basis_x     = bspline1d_basis(u2d[:,0], xknots, degree)
  basis_y     = bspline1d_basis(u2d[:,1], yknots, degree)
  basis_x_du1 = bspline1d_basis_derivs(u2d[:,0], xknots, degree)
  basis_y_du2 = bspline1d_basis_derivs(u2d[:,1], yknots, degree)

  return np.stack([ basis_y[:,np.newaxis,:] * basis_x_du1[:,:,np.newaxis],
                    basis_x[:,:,np.newaxis] * basis_y_du2[:,np.newaxis,:] ],
                  axis=3)

def compare_1d_deriv_times():
  npr.seed(1)
  u           = npr.rand(100)
  num_control = 10
  degree      = 3
  num_knots   = num_control + degree + 1
  knots = np.hstack([np.zeros(degree),
                     np.linspace(0, 1, num_knots - 2*degree),
                     np.ones(degree)])


  df_basis_fn = jax.jit(jax.vmap(
    jax.jacfwd(bspline1d_basis, argnums=0),
    in_axes=(0, None, None),
    ), static_argnums=(2,))

  version1 = lambda : bspline1d_basis_derivs(u, knots, degree)
  version2 = lambda : np.squeeze(df_basis_fn(u, knots, degree))

  version1_t = timeit.timeit(version1, number=1000)
  version2_t = timeit.timeit(version2, number=1000)

  print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                    version2_t/version1_t))

def compare_2d_deriv_times():
  npr.seed(1)

  df_basis_fn = jax.jit(jax.vmap(
    jax.jacfwd(bspline2d_basis, argnums=0),
    in_axes=(0, None, None, None),
  ), static_argnums=(3,))

  npr.seed(1)

  u            = npr.rand(100,2)
  num_xcontrol = 10
  num_ycontrol = 11
  degree       = 3
  num_xknots   = num_xcontrol + degree + 1
  num_yknots   = num_ycontrol + degree + 1
  xknots = np.hstack([np.zeros(degree),
                      np.linspace(0, 1, num_xknots - 2*degree),
                      np.ones(degree)])
  yknots = np.hstack([np.zeros(degree),
                      np.linspace(0, 1, num_yknots - 2*degree),
                      np.ones(degree)])

  version1 = lambda : bspline2d_basis_derivs(u, xknots, yknots, degree)
  version2 = lambda : np.squeeze(df_basis_fn(u, xknots, yknots, degree))

  version1_t = timeit.timeit(version1, number=1000)
  version2_t = timeit.timeit(version2, number=1000)

  print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                    version2_t/version1_t))
