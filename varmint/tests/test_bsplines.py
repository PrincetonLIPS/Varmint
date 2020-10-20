import jax
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest
import unittest      as ut

from geomdl import BSpline

import bsplines

class TestBSplines(ut.TestCase):

  def test_divide00(self):
    # Verify basic functionality of the divide00 function, which is required for
    # the spline functions to do sensible things outside their support.

    x = np.array([ 2.0, 0.0, 0.0 ])
    y = np.array([ 1.0, 1.0, 0.0 ])

    div00 = bsplines.divide00(x, y)

    self.assertEqual(div00[0], 2.0)
    self.assertEqual(div00[1], 0.0)
    self.assertEqual(div00[2], 0.0)

  def test_divide00_jacfwd(self):
    # Forward mode needs to be well-behaved with divide00.

    divide00_jacx = jax.jacfwd(bsplines.divide00, argnums=0)
    divide00_jacy = jax.jacfwd(bsplines.divide00, argnums=1)

    x = np.array([ 1.0, 0.0, 0.0 ])
    y = np.array([ 2.0, 1.0, 0.0 ])

    jacx = divide00_jacx(x, y)

    self.assertEqual(jacx[0,0], 0.5)
    self.assertEqual(jacx[0,1], 0.0)
    self.assertEqual(jacx[0,2], 0.0)

    self.assertEqual(jacx[1,0], 0.0)
    self.assertEqual(jacx[1,1], 1.0)
    self.assertEqual(jacx[1,2], 0.0)

    self.assertEqual(jacx[2,0], 0.0)
    self.assertEqual(jacx[2,1], 0.0)
    self.assertEqual(jacx[2,2], 0.0)

  def test_divide00_jacrev(self):
    # Reverse mode needs to be well-behaved with divide00.

    divide00_jacx = jax.jacrev(bsplines.divide00, argnums=0)
    divide00_jacy = jax.jacrev(bsplines.divide00, argnums=1)

    x = np.array([ 1.0, 0.0, 0.0 ])
    y = np.array([ 1.0, 1.0, 0.0 ])

    jacx = divide00_jacx(x, y)

    self.assertEqual(jacx[0,0], 1.0)
    self.assertEqual(jacx[0,1], 0.0)
    self.assertEqual(jacx[0,2], 0.0)

    self.assertEqual(jacx[1,0], 0.0)
    self.assertEqual(jacx[1,1], 1.0)
    self.assertEqual(jacx[1,2], 0.0)

    self.assertEqual(jacx[2,0], 0.0)
    self.assertEqual(jacx[2,1], 0.0)
    self.assertEqual(jacx[2,2], 0.0)

  def test_bspline_1(self):
    npr.seed(1)

    crv            = BSpline.Curve()
    crv.degree     = 2
    crv.ctrlpts    = [[0,1], [2,4], [5,-1]]
    crv.knotvector = np.array([0, 0, 0, 1, 1, 1])

    npr.seed(1)
    for ii in range(100):
      u = npr.rand()
      point1 = crv.evaluate_single(u)
      point2 = bsplines.bspline1d(
        u,
        np.array(crv.ctrlpts),
        np.array(crv.knotvector),
        crv.degree,
      ).ravel()
      nptest.assert_almost_equal(point1, point2, decimal=6)

  def test_bspline_2(self):
    npr.seed(2)

    crv         = BSpline.Curve()
    crv.degree  = 2
    crv.ctrlpts = [[0, 0], [1, -1], [3, 2], [3,1]]
    crv.knotvector = [0, 0, 0, 1, 2, 2, 2]

    npr.seed(1)
    for ii in range(100):
      u = npr.rand()
      point1 = crv.evaluate_single(u)
      point2 = bsplines.bspline1d(
        u,
        np.array(crv.ctrlpts),
        np.array(crv.knotvector),
        crv.degree,
      ).ravel()
      nptest.assert_almost_equal(point1, point2, decimal=6)

  def test_bspline1d_basis_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100)
    degree      = 3
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline1d_basis(u, knots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], num_control)

  def test_bspline1d_basis_sizes_2(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100, 50)
    degree      = 3
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline1d_basis(u, knots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], 50)
    self.assertEqual(basis.shape[2], num_control)

  def test_bspline1d_basis_sizes_3(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100, 50, 25)
    degree      = 3
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline1d_basis(u, knots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], 50)
    self.assertEqual(basis.shape[2], 25)
    self.assertEqual(basis.shape[3], num_control)

  def test_bspline1d_basis_sizes_4(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100, 50, 25)
    degree      = 7
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline1d_basis(u, knots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], 50)
    self.assertEqual(basis.shape[2], 25)
    self.assertEqual(basis.shape[3], num_control)

  def test_bspline2d_basis_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u            = npr.randn(100,2)
    degree       = 3
    num_xcontrol = 10
    num_ycontrol = 11
    num_xknots    = num_xcontrol + degree + 1
    num_yknots    = num_ycontrol + degree + 1

    xknots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_xknots - 2*degree),
                       np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])

    basis = bsplines.bspline2d_basis(u, xknots, yknots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], num_xcontrol)
    self.assertEqual(basis.shape[2], num_ycontrol)

  def test_bspline3d_basis_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u            = npr.randn(100,3)
    degree       = 3
    num_xcontrol = 10
    num_ycontrol = 11
    num_zcontrol = 12
    num_xknots    = num_xcontrol + degree + 1
    num_yknots    = num_ycontrol + degree + 1
    num_zknots    = num_zcontrol + degree + 1

    xknots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_xknots - 2*degree),
                       np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])
    zknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_zknots - 2*degree),
                        np.ones(degree)])

    basis = bsplines.bspline3d_basis(u, xknots, yknots, zknots, degree)
    self.assertEqual(basis.shape[0], 100)
    self.assertEqual(basis.shape[1], num_xcontrol)
    self.assertEqual(basis.shape[2], num_ycontrol)
    self.assertEqual(basis.shape[3], num_zcontrol)

  def test_bspline1d_basis_sums(self):

    npr.seed(1)

    u           = npr.rand(100)
    degree      = 7
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline1d_basis(u, knots, degree)
    sums = np.sum(basis, axis=1)
    nptest.assert_array_almost_equal(sums, np.ones(sums.shape), decimal=6)

  def test_bspline2d_basis_sums(self):

    npr.seed(1)

    u           = npr.rand(100,2)
    degree      = 7
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline2d_basis(u, knots, knots, degree)
    sums = np.sum(basis, axis=(1,2,))
    nptest.assert_array_almost_equal(sums, np.ones(sums.shape), decimal=6)

  def test_bspline3d_basis_sums(self):

    npr.seed(1)

    u           = npr.rand(100,3)
    degree      = 7
    num_control = 10
    num_knots   = num_control + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    basis = bsplines.bspline3d_basis(u, knots, knots, knots, degree)
    sums = np.sum(basis, axis=(1,2,3,))
    nptest.assert_array_almost_equal(sums, np.ones(sums.shape), decimal=6)

  def test_bspline1d_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100)
    control     = npr.randn(10)
    degree      = 3
    num_knots   = control.shape[0] + degree + 1
    knots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_knots - 2*degree),
                       np.ones(degree)])
    func = bsplines.bspline1d(u, control, knots, degree)
    self.assertEqual(func.shape[0], 100)

  def test_bspline2d_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100, 2)
    control     = npr.randn(10, 11, 2)
    degree      = 3
    num_xknots  = control.shape[0] + degree + 1
    num_yknots  = control.shape[1] + degree + 1

    xknots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_xknots - 2*degree),
                       np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])

    funcs = bsplines.bspline2d(u, control, xknots, yknots, degree)
    self.assertEqual(len(funcs.shape), 2)
    self.assertEqual(funcs.shape[0], 100)
    self.assertEqual(funcs.shape[1], 2)

  def test_bspline3d_sizes_1(self):
    # Make sure we get sizes we expect.

    npr.seed(1)

    u           = npr.randn(100, 3)
    control     = npr.randn(10, 11, 13, 3)
    degree      = 3
    num_xknots  = control.shape[0] + degree + 1
    num_yknots  = control.shape[1] + degree + 1
    num_zknots  = control.shape[2] + degree + 1

    xknots = np.hstack([np.zeros(degree),
                       np.linspace(0, 1, num_xknots - 2*degree),
                       np.ones(degree)])
    yknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_yknots - 2*degree),
                        np.ones(degree)])
    zknots = np.hstack([np.zeros(degree),
                        np.linspace(0, 1, num_zknots - 2*degree),
                        np.ones(degree)])

    funcs = bsplines.bspline3d(u, control, xknots, yknots, zknots, degree)
    self.assertEqual(len(funcs.shape), 2)
    self.assertEqual(funcs.shape[0], 100)
    self.assertEqual(funcs.shape[1], 3)
