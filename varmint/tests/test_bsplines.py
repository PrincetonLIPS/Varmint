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
