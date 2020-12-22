import unittest      as ut
import jax
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
    ''' Test More' Sec 8.1 with 1x '''

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
    ''' Test More' Sec 8.1 with 10x '''

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
    ''' Test More' Sec 8.1 with 100x '''

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
    ''' Test More' Sec 8.4 with 1x '''

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
    ''' Test More' Sec 8.4 with 10x '''

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
    ''' Test More' Sec 8.4 with 100x '''

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

class Test_LevenbergMarquardt_Jacobian(ut.TestCase):

  def test_quadratic_basic(self):
    ''' Verify the correct answer for a really simple quadratic. '''

    def quad(x, args):
      target, = args
      return x - target

    npr.seed(1)
    target = npr.randn(5)

    optfun = get_lmfunc(quad)
    x0 = npr.randn(5)

    x_star = optfun(x0, (target,))

    nptest.assert_array_almost_equal(x_star, target)


  def test_quadratic_meta(self):
    ''' Verify that we can close over the objective. '''

    def quad(x, args):
      target, = args
      return x - target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    npr.seed(1)
    for ii in range(10):
      target = npr.randn(5)
      x_star = opt_result(target)

      nptest.assert_array_almost_equal(x_star, target)


  def test_quadratic_jacfwd_1(self):
    ''' Forward-mode:
    Verify that we get the identity Jacobian for the closed function.
    '''

    def quad(x, args):
      target, = args
      return x - target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacfwd(opt_result)

    npr.seed(1)
    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, np.eye(5))


  def test_quadratic_jacfwd_2(self):
    ''' Forward-mode:
    Verify the Jacobian for a diagonal but non-linear situation.
    '''

    def quad(x, args):
      target, = args
      return x - target**2

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacfwd(opt_result)

    npr.seed(1)
    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, np.diag(2*target))


  def test_quadratic_jacfwd_3(self):
    ''' Forward-mode:
    Verify that we can use a non-diagonal Jacobian.
    '''

    npr.randn(1)
    A = npr.randn(5,5)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacfwd(opt_result)

    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)


  def test_quadratic_jacfwd_4(self):
    ''' Forward-mode:
    Verify non-diagonal Jacobian with larger output.
    '''

    npr.randn(1)
    A = npr.randn(5,4)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacfwd(opt_result)

    for ii in range(10):
      target = npr.randn(4)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)


  def test_quadratic_jacfwd_5(self):
    ''' Foward-mode:
    Verify non-diagonal Jacobian with larger input.
    '''

    npr.randn(1)
    A = npr.randn(5,6)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacfwd(opt_result)

    for ii in range(10):
      target = npr.randn(6)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)


  def test_quadratic_jacrev_1(self):
    ''' Reverse-mode:
    Verify that we get the identity Jacobian for the closed function.
    '''

    def quad(x, args):
      target, = args
      return x - target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacrev(opt_result)

    npr.seed(1)
    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, np.eye(5))


  def test_quadratic_jacrev_2(self):
    ''' Reverse-mode:
    Verify the Jacobian for a diagonal but non-linear situation.
    '''

    def quad(x, args):
      target, = args
      return x - target**2

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacrev(opt_result)

    npr.seed(1)
    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, np.diag(2*target))


  def test_quadratic_jacrev_3(self):
    ''' Reverse-mode:
    Verify that we can use a non-diagonal Jacobian.
    '''

    npr.randn(1)
    A = npr.randn(5,5)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacrev(opt_result)

    for ii in range(10):
      target = npr.randn(5)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)


  def test_quadratic_jacrev_4(self):
    ''' Reverse-mode:
    Verify non-diagonal Jacobian with larger output.
    '''

    npr.randn(1)
    A = npr.randn(5,4)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacrev(opt_result)

    for ii in range(10):
      target = npr.randn(4)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)


  def test_quadratic_jacrev_5(self):
    ''' Reverse-mode:
    Verify non-diagonal Jacobian with larger input.
    '''

    npr.randn(1)
    A = npr.randn(5,6)

    def quad(x, args):
      target, = args
      return x - A @ target

    optfun = get_lmfunc(quad)

    def opt_result(target):
      x0 = np.zeros(5)
      return optfun(x0, (target,))

    opt_result_jac = jax.jacrev(opt_result)

    for ii in range(10):
      target = npr.randn(6)
      jac = opt_result_jac(target)

      nptest.assert_array_almost_equal(jac, A)
