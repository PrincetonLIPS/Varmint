import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import timeit

from functools import partial

from varmint.utils.typing import Array1D, Array2D, Array3D, Array4D, ArrayND


def mesh(*control):
    """ Generate a control point mesh.

    Parameters:
    -----------

    - *control: A variable number of 1d array objects.  These are turned into a
                control mesh of that many dimensions.  So to create a 2D mesh,
                you could give it a sequence of length J and a sequence of length
                K; it will return an ndarray that is J x K x 2. If you want to
                create a 3D mesh, you could give it three sequences of lengths
                J, K, and M, respectively, and you'd get back an ndarray of size
                J x K x M x 3.  The last dimension will always correspond to the
                number of sequences provided.

    Returns:
    --------
     Returns an ndarray object with a mesh of control points.

    Examples:
    ---------
    >> mesh(np.arange(3), np.arange(4))
        DeviceArray([[[0, 0],
                      [0, 1],
                      [0, 2],
                      [0, 3]],

                     [[1, 0],
                      [1, 1],
                      [1, 2],
                      [1, 3]],

                     [[2, 0],
                      [2, 1],
                      [2, 2],
                      [2, 3]]], dtype=int32)

    >> mesh(np.arange(3), np.arange(4), np.arange(5)).shape
        (3, 4, 5, 3)
    """
    return np.stack(np.meshgrid(*control, indexing='ij'), axis=-1)


def divide00(num, denom):
    """ Divide such that 0/0 = 0.

    The trick here is to do this in such a way that reverse-mode and forward-mode
    automatic differentation via JAX still work reasonably.
    """

    force_zero = np.logical_and(num == 0, denom == 0)
    return np.where(force_zero, np.float32(0.0), num) \
        / np.where(force_zero, np.float32(1.0), denom)


def default_knots(degree, num_ctrl):
    return np.hstack([np.zeros(degree),
                      np.linspace(0, 1, num_ctrl - degree + 1),
                      np.ones(degree)])


################################################################################
# Standard BSpline library functions.
################################################################################


def bspline1d_basis(u: Array1D, knots: Array1D, degree: int) -> Array2D:
    """ Computes b-spline basis functions in one dimension.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
        assumed to be in the interval [0,1) although they really just need to
        be consistent with the knot ranges. Can be an ndarray of any size.

    - knots: The knot vector. Should be non-decreasing and consistent with the
            specified degree.  A one-dimensional ndarray of size J.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------
    Returns an ndarray of shape (N, J-D-1).

    """
    u1d = np.atleast_1d(u)

    # Set things up for broadcasting.
    # Append a singleton dimension onto the u points.
    # Prepend the correct number of singleton dimensions onto knots.
    # The vars are named 2d but they could be bigger.
    u2d = np.expand_dims(u1d, -1)
    k2d = np.expand_dims(knots, tuple(range(len(u1d.shape))))

    # Handle degree=0 case first.
    # Modify knots so that when u=1.0 we get 1.0 rather than 0.0.
    k2d = np.where(k2d == knots[-1], knots[-1]+np.finfo(u2d.dtype).eps, k2d)

    # The degree zero case is just the indicator function on the
    # half-open interval specified by the knots.
    B = (k2d[..., :-1] <= u2d) * (u2d < k2d[..., 1:]) + 0.0

    # We expect degree to be small, so unrolling is tolerable.
    for deg in range(1, degree+1):

        # There are two halves.  The indexing is a little tricky.
        # Also we're using the np.divide 'where' argument to deal
        # with the fact that we want 0/0 to equal zero.
        # This all computes much more than we strictly need, because
        # so much of this is just multiplying zero by things.
        # However, I think the vectorized implementation is worth
        # it for using things like JAX and GPUs.
        # FIXME: Pretty sure I could do the denominator with one subtract.
        v0_num = B[..., :-1] * (u2d - k2d[..., :-deg-1])
        v0_denom = k2d[..., deg:-1] - k2d[..., :-deg-1]
        v0 = divide00(v0_num, v0_denom)

        v1_num = B[..., 1:] * (k2d[..., deg+1:] - u2d)
        v1_denom = k2d[..., deg+1:] - k2d[..., 1:-deg]
        v1 = divide00(v1_num, v1_denom)

        B = v0 + v1

    return B


def bspline2d_basis(u: Array2D, xknots: Array1D, yknots: Array1D, degree: int) -> Array3D:
    """ Compute b-spline basis functions in two dimensions as tensor products.

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

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Returns an ndarray of shape (N, J-D-1, K-D-1),
    where ret[n,:,:] is the tensor product of basis functions for x and y,
    evaluated at u[n,:].

    """

    u2d = np.atleast_2d(u)

    basis_x = bspline1d_basis(u2d[:, 0], xknots, degree)[:, :, np.newaxis]
    basis_y = bspline1d_basis(u2d[:, 1], yknots, degree)[:, np.newaxis, :]
    basis_xy = basis_x * basis_y

    return basis_xy


def bspline3d_basis(u: Array3D, xknots: Array1D, yknots: Array1D, zknots: Array1D, degree: int) -> Array4D:
    """ Compute b-spline basis functions in three dimensions as tensor products.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Must be a two-dimensional ndarray
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

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Returns an ndarray of shape (N, J-D-1, K-D-1, L-D-1),
    where ret[n,:,:,:] is the tensor product of basis functions
    for x, y, and z, evaluated at u[n,:].

    """

    u2d = np.atleast_2d(u)

    basis_x = bspline1d_basis(u2d[:, 0], xknots, degree)[
        :, :, np.newaxis, np.newaxis]
    basis_y = bspline1d_basis(u2d[:, 1], yknots, degree)[
        :, np.newaxis, :, np.newaxis]
    basis_z = bspline1d_basis(u2d[:, 2], zknots, degree)[
        :, np.newaxis, np.newaxis, :]

    basis_xyz = basis_x * basis_y * basis_z

    return basis_xyz


def bspline1d(u: Array1D, control: Array2D,
              knots: Array1D, degree: int) -> Array2D:
    """ Evaluate a one-dimensional bspline function.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Can be a 1d ndarray of any size.

    - control: The 2d array of control points of shape (J-D-1, n_d).

    - knots: The knot vector. Should be non-decreasing and consistent with the
             specified degree. A one-dimensional ndarray of length J.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Returns an ndarray of shape (N, n_d).

    """

    return bspline1d_basis(u, knots, degree) @ control


def bspline2d(u: Array2D, control: Array3D, xknots: Array1D,
              yknots: Array1D, degree: int) -> Array2D:
    """ Evaluate a two-dimensional bspline function.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Of shape (N, 2).
         The first dimension (N) corresponds to the number of points
         at which the basis functions should be evaluated.

    - control: The 3d array of control points of shape (J-D-1, K-D-1, n_d).

    - xknots: The x knot vector. Should be non-decreasing and consistent with the
              specified degree. A one-dimensional ndarray of length J.

    - yknots: The y knot vector. Should be non-decreasing and consistent with the
              specified degree. A one-dimensional ndarray of length K.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Returns an ndarray of shape (N, n_d).

    """

    basis_xy = bspline2d_basis(u, xknots, yknots, degree)

    return np.tensordot(basis_xy, control, ((1, 2), (0, 1)))


def bspline3d(u: Array2D, control: Array4D, xknots: Array1D,
              yknots: Array1D, zknots: Array1D, degree: int) -> Array2D:
    """ Evaluate a three-dimensional bspline function.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Of shape (N, 3).
         The first dimension (N) corresponds to the number of points
         at which the basis functions should be evaluated.

    - control: The 4d array of control points of shape (J-D-1, K-D-1, L-D-1, n_d).

    - xknots: The knot vector for the first dimension. Should be non-decreasing
              and consistent with the specified degree. A one-dimensional
              ndarray of length J.

    - yknots: The knot vector for the second dimension. Should be non-decreasing
              and consistent with the specified degree. A one-dimensional
              ndarray of length K.

    - zknots: The knot vector for the third dimension. Should be non-decreasing
              and consistent with the specified degree. A one-dimensional
              ndarray of length L.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Returns a 2d ndarray of shape (N, n_d).

    """

    basis_xyz = bspline3d_basis(u, xknots, yknots, zknots, degree)

    return np.tensordot(basis_xyz, control, ((1, 2, 3), (0, 1, 2)))


################################################################################
# Derivatives with respect to inputs (parent configuration).
################################################################################


def _bspline1d_basis_derivs_hand(u, knots, degree):
    """ Derivatives of basis functions.
    
    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
        assumed to be in the interval [0,1) although they really just need to
        be consistent with the knot ranges. Can be an ndarray of any size.

    - knots: The knot vector. Should be non-decreasing and consistent with the
            specified degree. A one-dimensional ndarray of length J.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Derivatives of each basis function with respect to the inputs u at each
    specified u. Has shape (N, J-D-1).

    """
    d_basis = bspline1d_basis(u, knots, degree-1)
    v0 = divide00(d_basis[:, :-1] * degree,
                  knots[degree:-1] - knots[:-degree-1])
    v1 = divide00(d_basis[:, 1:] * degree, knots[degree+1:] - knots[1:-degree])
    return v0 - v1


_bspline1d_basis_derivs_jax = \
    jax.vmap(
        lambda u, knots, degree:
        np.squeeze(jax.jacfwd(bspline1d_basis, argnums=0)(
            u, knots, degree
        )),
        (0, None, None),
    )


def _bspline1d_derivs_hand(u, control, knots, degree):
    """ Evaluate the derivative of a one-dimensional bspline function.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Can be a 1d ndarray of any size.

    - control: The 2d array of control points.  The first dimension should
               have size J-D-1.

    - knots: The knot vector. Should be non-decreasing and consistent with the
             specified degree. A one-dimensional ndarray of length J.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Derivatives of the bspline function with respect to the inputs u at each
    specified u. Has shape (N, n_d).
    
    """
    return bspline1d_basis_derivs(u, knots, degree) @ control


_bspline1d_derivs_jax = \
    jax.vmap(
        lambda u, control, knots, degree:
        np.squeeze(jax.jacfwd(bspline1d, argnums=0)
                   (u, control, knots, degree)),
        (0, None, None, None),
    )


def _bspline2d_basis_derivs_hand(u, xknots, yknots, degree):
    """ Derivatives of 2d basis functions.
    
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

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Jacobians of the bspline bases with respect to the inputs u at each
    specified u. Has shape (N, J-D-1, K-D-1, 2).

    """
    u2d = np.atleast_2d(u)

    basis_x = bspline1d_basis(u2d[:, 0], xknots, degree)
    basis_y = bspline1d_basis(u2d[:, 1], yknots, degree)
    basis_x_du1 = bspline1d_basis_derivs(u2d[:, 0], xknots, degree)
    basis_y_du2 = bspline1d_basis_derivs(u2d[:, 1], yknots, degree)

    return np.stack([basis_y[:, np.newaxis, :] * basis_x_du1[:, :, np.newaxis],
                     basis_x[:, :, np.newaxis] * basis_y_du2[:, np.newaxis, :]],
                    axis=3)


_bspline2d_basis_derivs_jax = \
    jax.vmap(
        lambda u, xknots, yknots, degree:
        np.squeeze(jax.jacfwd(bspline2d_basis, argnums=0)(
            u, xknots, yknots, degree
        )),
        (0, None, None, None),
    )


def _bspline2d_derivs_hand(u, control, xknots, yknots, degree):
    """ Derivatives of 2d spline functions.
    
    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Of shape (N, 2).
         The first dimension (N) corresponds to the number of points
         at which the basis functions should be evaluated.

    - control: The 3d array of control points of shape (J-D-1, K-D-1, n_d).

    - xknots: The x knot vector. Should be non-decreasing and consistent with the
              specified degree. A one-dimensional ndarray of length J.

    - yknots: The y knot vector. Should be non-decreasing and consistent with the
              specified degree. A one-dimensional ndarray of length K.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Jacobians of the bspline function with respect to the inputs u at each
    specified u. Has shape (N, n_d, 2).

    """

    # This is slightly annoying because it gives transposed Jacobians.
    return np.swapaxes(
        np.tensordot(
            bspline2d_basis_derivs(u, xknots, yknots, degree),
            control,
            ((1, 2), (0, 1)),
        ),
        1, 2,  # exchange the last two axes
    )


_bspline2d_derivs_jax = \
    jax.vmap(
        lambda u, control, xknots, yknots, degree:
        np.squeeze(jax.jacfwd(bspline2d, argnums=0)(
            u, control, xknots, yknots, degree
        )),
        (0, None, None, None, None),
    )


_bspline3d_basis_derivs_jax = \
    jax.vmap(
        lambda u, xknots, yknots, zknots, degree:
        np.squeeze(jax.jacfwd(bspline3d_basis, argnums=0)(
            u, xknots, yknots, zknots, degree
        )),
        (0, None, None, None, None),
    )


_bspline3d_derivs_jax = \
    jax.vmap(
        lambda u, control, xknots, yknots, zknots, degree:
        np.squeeze(jax.jacfwd(bspline3d, argnums=0)(
            u, control, xknots, yknots, zknots, degree
        )),
        (0, None, None, None, None, None),
    )


# Hand-coded appears slightly faster, but don't use it unless we need the speed.
bspline1d_derivs = _bspline1d_derivs_hand
bspline1d_basis_derivs = _bspline1d_basis_derivs_hand

bspline2d_derivs = _bspline2d_derivs_hand
bspline2d_basis_derivs = _bspline2d_basis_derivs_hand

# Don't bother coding the 3d derivatives by hand.
bspline3d_derivs = _bspline3d_derivs_jax
bspline3d_basis_derivs = _bspline3d_basis_derivs_jax


################################################################################
# Derivatives with respect to control points.
################################################################################


def _bspline1d_derivs_ctrl_hand(u, control, knots, degree):
    """ Evaluate the Jacobian with respect to control points of a
    one-dimensional bspline function.

    Parameters:
    -----------

    - u: The locations at which to evaluate the basis functions, generally
         assumed to be in the interval [0,1) although they really just need to
         be consistent with the knot ranges. Can be a 1d ndarray of any size.

    - control: The 1d or 2d array of control points.  The first dimension should
               have size J-D-1.

    - knots: The knot vector. Should be non-decreasing and consistent with the
             specified degree. A one-dimensional ndarray of length J.

    - degree: The degree of the piecewise polynomials, D. Integer.

    Returns:
    --------

    Derivatives of the bspline function with respect to the inputs u at each
    specified u. Has shape (N, n_d).
    
    """

    if len(control.shape) > 1:
        raise ValueError("Hand computation not supported for control point shape."
                         "Use JAX version, or fix me.")

    return bspline1d_basis(u, knots, degree)


_bspline1d_derivs_ctrl_jax = \
    jax.vmap(
        lambda u, control, knots, degree:
        np.squeeze(jax.jacfwd(bspline1d, argnums=1)(
            u, control, knots, degree
        )),
        (0, None, None, None),
    )


def _bspline2d_derivs_ctrl_hand(u, control, xknots, yknots, degree):
    if control.shape[1] != 2:
        raise ValueError("Hand computation not supported for control point shape"
                         "Use JAX version, or fix me.")

    basis = bspline2d_basis(
        u, xknots, yknots, degree)[:, np.newaxis, :, :, np.newaxis]
    zeros = np.zeros_like(basis)
    return np.concatenate([np.concatenate([basis, zeros], axis=1),
                           np.concatenate([zeros, basis], axis=1)],
                          axis=4)


_bspline2d_derivs_ctrl_jax = \
    jax.vmap(
        lambda u, control, xknots, yknots, degree:
        np.squeeze(jax.jacfwd(bspline2d, argnums=1)(
            u, control, xknots, yknots, degree
        )),
        (0, None, None, None, None),
    )


_bspline3d_derivs_ctrl_jax = \
    jax.vmap(
        lambda u, control, xknots, yknots, zknots, degree:
        np.squeeze(jax.jacfwd(bspline3d, argnums=1)(
            u, control, xknots, yknots, zknots, degree
        )),
        (0, None, None, None, None, None),
    )


# Hand-coded appears slightly faster, but don't use it unless we need the speed.
bspline1d_derivs_ctrl = _bspline1d_derivs_ctrl_hand
bspline2d_derivs_ctrl = _bspline2d_derivs_ctrl_jax  # fixme?
bspline3d_derivs_ctrl = _bspline3d_derivs_ctrl_jax


################################################################################
# Do some timing comparisons.
################################################################################


def compare_1d_basis_deriv_times():
    npr.seed(1)

    u = npr.rand(1000)
    num_control = 100
    degree = 3
    num_knots = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])

    def version1(): return _bspline1d_basis_derivs_hand(u, knots, degree)
    def version2(): return _bspline1d_basis_derivs_jax(u, knots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))


def compare_1d_deriv_times():
    npr.seed(1)

    u = npr.rand(100)
    control = npr.randn(10)
    degree = 3
    num_knots = control.shape[0] + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])

    def version1(): return _bspline1d_derivs_hand(u, control, knots, degree)
    def version2(): return _bspline1d_derivs_jax(u, control, knots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))


def compare_1d_deriv_ctrl_times():
    npr.seed(1)

    u = npr.rand(100)
    control = npr.randn(10)
    degree = 3
    num_knots = control.shape[0] + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])

    def version1(): return _bspline1d_derivs_ctrl_hand(u, control, knots, degree)
    def version2(): return _bspline1d_derivs_ctrl_jax(u, control, knots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))


def compare_2d_basis_deriv_times():
    npr.seed(1)

    u = npr.rand(100, 2)
    num_xcontrol = 10
    num_ycontrol = 11
    degree = 3
    num_xknots = num_xcontrol + degree + 1
    num_yknots = num_ycontrol + degree + 1
    xknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_xknots - 2*degree),
                        np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])

    def version1(): return _bspline2d_basis_derivs_hand(u, xknots, yknots, degree)
    def version2(): return _bspline2d_basis_derivs_jax(u, xknots, yknots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))


def compare_2d_deriv_times():
    npr.seed(1)

    u = npr.rand(100, 2)
    control = npr.randn(10, 11, 2)
    degree = 3
    num_xknots = control.shape[0] + degree + 1
    num_yknots = control.shape[1] + degree + 1
    xknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_xknots - 2*degree),
                        np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])

    def version1(): return _bspline2d_derivs_hand(
        u, control, xknots, yknots, degree)

    def version2(): return _bspline2d_derivs_jax(u, control, xknots, yknots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))


def compare_2d_deriv_ctrl_times():
    npr.seed(1)

    u = npr.rand(1000, 2)
    control = npr.randn(10, 11, 2)
    degree = 3
    num_xknots = control.shape[0] + degree + 1
    num_yknots = control.shape[1] + degree + 1
    xknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_xknots - 2*degree),
                        np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])

    def version1(): return _bspline2d_derivs_ctrl_hand(
        u, control, xknots, yknots, degree)

    def version2(): return _bspline2d_derivs_ctrl_jax(
        u, control, xknots, yknots, degree)

    version1_t = timeit.timeit(version1, number=1000)
    version2_t = timeit.timeit(version2, number=1000)

    print('Hand: %0.3fsec  JAX: %0.3fsec (%0.1fx)' % (version1_t, version2_t,
                                                      version2_t/version1_t))
