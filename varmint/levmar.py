import jax
import jax.numpy        as np
import jax.numpy.linalg as npla

from scipy.optimize import OptimizeResult

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

  # Initialize lambda to its default.
  lamb = init_lamb

  # Statistics to track.
  nfev = 1
  njev = 1
  nit  = 0

  x   = x0
  dx  = np.inf
  fx  = fun(x)
  err = np.sqrt(np.mean(fx**2))

  while True:
    nit += 1

    # Precompute Jacobian-related quantities.
    Jx   = jacfun(x)
    JTJ  = Jx.T @ Jx
    dJTJ = np.diag(np.diag(JTJ))
    JTfx = Jx.T @ fx
    njev += 1

    # TODO: if JTfx is zero, this is messy.

    while True:

      # Solve the pseudo-inverse system with the given lamb.
      # step = npla.solve(JTJ + lamb*dJTJ, JTfx)
      step, _, _, _ = npla.lstsq(JTJ + lamb*dJTJ, JTfx)

      # Evaluate the quality of this location.
      new_x   = x - step
      new_fx  = fun(new_x)
      new_err = np.sqrt(np.mean(new_fx**2))
      nfev += 1

      # Apply the L-M heuristic.
      if new_err > err:
        # Increase lamb and don't update.
        lamb = lamb * lambfac

      else:
        # Decrease lamb and update.
        lamb = lamb / lambfac
        dx   = np.sqrt(np.mean((x - new_x)**2))
        x    = new_x
        fx   = new_fx
        err  = new_err
        break

    # Matching scipy.optimize.least_squares.
    # 0 : the maximum number of function evaluations is exceeded.
    # 1 : gtol termination condition is satisfied.
    # 2 : ftol termination condition is satisfied.
    # 3 : xtol termination condition is satisfied.
    # 4 : Both ftol and xtol termination conditions are satisfied

    if err < ftol:
      message = 'Error tolerance achived'
      success = True
      status  = 2
      break

    elif dx < xtol:
      message = 'Solution tolerance achieved'
      success = True
      status  = 3
      break

    elif np.sqrt(np.mean(Jx**2)) < gtol:
      message = 'Jacobian tolerance achieved'
      success = True
      status  = 1
      break

    elif nit >= max_niters:
      message = 'Maximum number of iterations reached'
      success = False
      status  = 0
      break

  result = OptimizeResult(
    x       = x,
    success = success,
    status  = status,
    message = message,
    fun     = fx,
    nfev    = nfev,
    njev    = njev,
    nit     = nit,
  )

  return result

D = 25
y = np.arange(D)+1
fun = lambda x: x**3 - y
jacfun = jax.jacfwd(fun)

res = levenberg_marquardt(fun, np.ones(D), jacfun)
print(res)
