import unittest      as ut
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest

from varmint.levmar import *

class Test_LevenbergMarquardt(ut.TestCase):

  def test_more_p1_1(self):

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
