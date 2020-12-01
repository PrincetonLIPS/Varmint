import jax
import jax.numpy         as np
import jax.numpy.linalg  as npla
import matplotlib.pyplot as plt

from scipy.optimize import OptimizeResult

# Moré, J.J., 1978. The Levenberg-Marquardt algorithm: implementation
# and theory. In Numerical analysis (pp. 105-116). Springer, Berlin,
# Heidelberg.

def norm(v):
  return np.sqrt(np.sum(v**2))

def phi(alpha, D, J, f, delta):
  ''' Eq. 5.2 '''

  # Slow, but just seeing how to make this work.
  return norm(D @ npla.solve(J.T @ J + alpha * D.T @ D, J.T @ f)) - delta

def lm_param( D, J, f, delta, sigma=0.1 ):
  ''' Determine the \lambda parameter in L-M as discussed by Moré. '''

  vg_phi = jax.value_and_grad(phi, argnums=0)

  phi0, dphi0 = vg_phi(0., D, J, f, delta)

  if phi0 <= 0:
    return 0.0

  # D is diagonal...
  iD = npla.inv(D)

  upper = norm((J @ iD).T @ f) / delta
  #print(upper)

  # TODO: if J is low-rank, use lower=0
  lower = - phi0 / dphi0
  #print(lower)

  # start with alpha = 0?
  alpha = 0.0

  while True:
    print('\t\talpha = %f' % (alpha))
    print('\t\t[%f, %f]' % (lower, upper))

    if alpha <= lower or alpha >= upper:
      alpha = np.maximum( 0.001 * upper, np.sqrt(upper * lower) )
    #print(alpha)

    phik, dphik = vg_phi(alpha, D, J, f, delta)
    if phik < 0:
      upper = alpha

    print('\t\tphi(alpha) = %f (vs %f)' % (phik, sigma*delta))

    if np.abs(phik) < sigma*delta:
      #print('done! %f' % (phik))
      print('\t\talpha = %f' % (alpha))
      return alpha

    lower = np.maximum(lower, alpha - phik/dphik)
    print('\t\t[%f, %f]' % (lower, upper))

    alpha = alpha - ( (phik + delta)/delta ) * ( phik/dphik )
    #print(alpha)

def levenberg_marquardt(
    fun,
    x0,
    jacfun,
    init_lamb = 0.001,
    ftol = 1e-8,
    xtol = 1e-8,
    gtol = 1e-8,
    lambfac = 10,
    max_niters = 100,
):
  ''' Minimize the sum of squares of f(x) with respect to x. '''

  sigma = 0.1

  # Initialize lambda to its default.
  lamb = init_lamb

  # Statistics to track.
  nfev = 1
  njev = 1
  nit  = 0

  x   = x0
  dx  = np.inf
  fx  = fun(x)
  Jx  = jacfun(x)

  # Choose initial matrix D.
  D = np.diag(np.sqrt(np.diag(Jx.T @ Jx)))

  # Choose initial step bound.
  factor = 1.0
  delta = factor * norm(D @ x)

  while True:
    nit += 1
    print('\nIteration %d %f nfev=%d  njev=%d' % (nit, norm(fx), nfev, njev))
    print('\tdelta = %f' % (delta))

    print('\t%f vs %f' % (norm(D @ npla.solve(Jx.T@Jx, Jx.T@fx)), (1+sigma)*delta))

    lamb = lm_param(D, Jx, fx, delta)
    print('\tlambda = %f' % (lamb))

    # Precompute Jacobian-related quantities.
    # Move this around to not recompute.
    #Jx   = jacfun(x)
    JTJ  = Jx.T @ Jx
    dJTJ = np.diag(np.diag(JTJ))
    JTfx = Jx.T @ fx

    # Solve the pseudo-inverse system with the given lamb.
    p = -npla.lstsq(JTJ + lamb*(D.T@D), JTfx)[0]

    print('\t%f <= %f <= %f' % ((1-sigma)*delta, norm(D @ p), (1+sigma)*delta))

    # Check ftol
    if (norm(Jx @ p)/norm(fx))**2 + 2 * lamb * (norm(D @ p)/norm(fx))**2 <= ftol:
      print('ftol reached nfev=%d  njev=%d' % (nfev, njev))
      return x

    # Evaluate the quality of this location.
    new_x   = x + p
    new_fx  = fun(new_x)
    new_Jx  = jacfun(new_x)
    nfev += 1
    njev += 1

    # much savings to be had here...
    numer = 1-(norm(new_fx)/norm(fx))**2
    denom = (norm(Jx @ p)/norm(fx))**2 + 2*lamb*(norm(D @ p)/norm(fx))**2
    rho = numer / denom
    print('\trho = %f' % (rho))

    if rho <= 0.25:

      # So many recomputations...
      gamma = -((norm(Jx @ p)/norm(fx))**2 + lamb * (norm(D @ p)/norm(fx))**2)
      print('\tgamma = %f' % (gamma))

      mu = 0.5 * gamma / (gamma + 0.5*(1-(norm(new_fx)/norm(fx))**2))

      mu = np.clip(mu, 0.1, 0.5)

      print('\tmu = %f' % (mu))

    if rho > 0.0001:
      x  = new_x
      fx = new_fx
      Jx = new_Jx

    if rho <= 0.25:
      delta = delta * mu
    elif (rho <= 0.75 and lamb == 0) or (rho > 0.75):
      delta = 2 * norm(D @ p)

    D = np.diag(np.maximum(np.diag(D), np.sqrt(np.diag(Jx.T @ Jx))))

    if delta <= xtol * norm(D @ x):
      print('xtol reached nfev=%d  njev=%d' % (nfev, njev))
      return x


#D = 25
#y = np.arange(D)+1
#fun = lambda x: x**3 - y
#jacfun = jax.jacfwd(fun)
#x = levenberg_marquardt(fun, np.ones(D), jacfun)
#print(x)
#print(res)

def p1(x):
  theta = (1/(2*np.pi)) * np.arctan(x[1]/x[0])
  if x[0] < 0:
    theta = theta + 0.5
  return np.array([
    10*(x[2] - 10*theta),
    10*(np.sqrt(x[0]**2 + x[1]**2) - 1),
    x[2],
  ])
jac_p1 = jax.jacfwd(p1)
x0 = np.array([-1., 0., 0.])
x = levenberg_marquardt(p1, 10*x0, jac_p1)
print(x, norm(p1(x)))

'''
def p4(x):
  ti = (np.arange(20)+1.0) * 0.2
  fi = (x[0] + x[1]*ti - np.exp(ti))**2 + (x[2] + x[3]*np.sin(ti) - np.cos(ti))**2
  return fi
jac_p4 = jax.jacfwd(p4)
x0 = np.array([25.0, 5.0, -5.0, 1.0])
x = levenberg_marquardt(p4, x0, jac_p4)
print(x, norm(p4(x)))
'''
