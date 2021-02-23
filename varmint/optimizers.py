import jax
import jax.numpy as np

import scipy.optimize as spopt

from functools import partial

from varmint.levmar import get_lmfunc
from varmint.newtoncg import newtoncg

def get_optfun(residual_fun, kind='levmar', **optargs):
  if kind == 'levmar':
    maxiters = optargs.get('maxiters', 50)
    lmfunc = get_lmfunc(residual_fun, maxiters=maxiters)
    lmfunc = jax.jit(lmfunc)
    
    # We would like all the optimizer functions
    # to have the same signature. 
    def wrapped_lmfunc(x0, args, jac, jacp, hess, hessp):
      return lmfunc(x0, args)
    return wrapped_lmfunc

  elif kind == 'scipy-lm':
    residual_fun = jax.jit(residual_fun)
    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return spopt.least_squares(residual_fun, x0, args=(args,),
                                 method='lm', jac=jac).x
    return wrapped_optfun

  elif kind == 'newtoncg':
    def total_residual(new_q, args):
      return np.sum(np.square(residual_fun(new_q, args)))

    grad = jax.grad(total_residual)

    # Use the Gauss-Newton approximation for the Hessian.
    # Want: J^T J p
    def hessp(new_q, p, args):
      partial_res_fun = lambda q: residual_fun(q, args)
      _, vjp = jax.vjp(partial_res_fun, new_q)
      return vjp(jax.jvp(partial_res_fun, (new_q,), (p,))[1])[0]

    optfun = newtoncg(total_residual, grad, hessp)
    optfun = jax.jit(optfun)
    
    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return optfun(x0, args)

    return wrapped_optfun

  else:
    raise ValueError(f'Unknown LS solver kind {kind}')

