import unittest      as ut
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest

from varmint.levmar import *

class Test_LevenbergMarquardt_More(ut.TestCase):


  # From section 8 of
  # Moré, J.J., 1978. The Levenberg-Marquardt algorithm: implementation
  # and theory. In Numerical analysis (pp. 105-116). Springer, Berlin,
  # Heidelberg.

  def test_p1_1(self):

    def p1(x):
      theta = (1/(2*np.pi)) * np.arctan(x[1]/x[0])
      theta = np.where(x[0] < 0, theta + 0.5, theta)
      return np.array([
        10*(x[2] - 10*theta),
        10*(np.sqrt(x[0]**2 + x[1]**2) - 1),
        x[2],
      ])
    arg_wrapper = lambda x, _: p1(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = np.array([-1., 0., 0.])

    xstar, res = optfun(x0, ())

    nptest.assert_array_almost_equal(xstar, np.array([1.0, 0.0, 0.0]))
    nptest.assert_array_almost_equal(res.Fx, np.zeros(3))
    self.assertLess(res.nFx, 1e-10)
    self.assertLess(res.nfev, 12)

  def test_p1_10(self):

    def p1(x):
      theta = (1/(2*np.pi)) * np.arctan(x[1]/x[0])
      theta = np.where(x[0] < 0, theta + 0.5, theta)
      return np.array([
        10*(x[2] - 10*theta),
        10*(np.sqrt(x[0]**2 + x[1]**2) - 1),
        x[2],
      ])
    arg_wrapper = lambda x, _: p1(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = 10 * np.array([-1., 0., 0.])

    xstar, res = optfun(x0, ())

    nptest.assert_array_almost_equal(xstar, np.array([1.0, 0.0, 0.0]))
    nptest.assert_array_almost_equal(res.Fx, np.zeros(3))
    self.assertLess(res.nFx, 1e-10)
    self.assertLess(res.nfev, 22)

  def test_p1_100(self):

    def p1(x):
      theta = (1/(2*np.pi)) * np.arctan(x[1]/x[0])
      theta = np.where(x[0] < 0, theta + 0.5, theta)
      return np.array([
        10*(x[2] - 10*theta),
        10*(np.sqrt(x[0]**2 + x[1]**2) - 1),
        x[2],
      ])
    arg_wrapper = lambda x, _: p1(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = 100 * np.array([-1., 0., 0.])

    xstar, res = optfun(x0, ())

    nptest.assert_array_almost_equal(xstar, np.array([1.0, 0.0, 0.0]))
    nptest.assert_array_almost_equal(res.Fx, np.zeros(3))
    self.assertLess(res.nFx, 1e-10)
    self.assertLess(res.nfev, 25)


  def test_p4_1(self):

    def p4(x):
      ti = (np.arange(20)+1.0) * 0.2
      fi = (x[0] + x[1]*ti - np.exp(ti))**2 \
        + (x[2] + x[3]*np.sin(ti) - np.cos(ti))**2
      return fi
    arg_wrapper = lambda x, _: p4(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = np.array([25.0, 5.0, -5.0, 1.0])

    _, res = optfun(x0, ())
    self.assertLess(res.nFx, 293.0)
    self.assertLess(res.nfev, 100)

  def test_p4_10(self):

    def p4(x):
      ti = (np.arange(20)+1.0) * 0.2
      fi = (x[0] + x[1]*ti - np.exp(ti))**2 \
        + (x[2] + x[3]*np.sin(ti) - np.cos(ti))**2
      return fi
    arg_wrapper = lambda x, _: p4(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = 10 * np.array([25.0, 5.0, -5.0, 1.0])

    _, res = optfun(x0, ())

    self.assertLess(res.nFx, 293.0)
    self.assertLess(res.nfev, 100)

  def test_p4_100(self):

    def p4(x):
      ti = (np.arange(20)+1.0) * 0.2
      fi = (x[0] + x[1]*ti - np.exp(ti))**2 \
        + (x[2] + x[3]*np.sin(ti) - np.cos(ti))**2
      return fi
    arg_wrapper = lambda x, _: p4(x)

    optfun = get_lmfunc(arg_wrapper, full_result=True)
    x0 = 100 * np.array([25.0, 5.0, -5.0, 1.0])

    _, res = optfun(x0, ())

    self.assertLess(res.nFx, 293.0)
    self.assertLess(res.nfev, 100)
