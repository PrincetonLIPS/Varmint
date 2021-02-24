import jax
import jax.numpy as np

import scipy.optimize as spopt
import scipy.stats

from functools import partial

from varmint.levmar import get_lmfunc
from varmint.newtoncg import newtoncg
from varmint.newtoncg_python import newtoncg_python

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
      return 0.5 * np.sum(np.square(residual_fun(new_q, args)))

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

  elif kind == 'newtoncg-python':
    def total_residual(new_q, args):
      return 0.5 * np.sum(np.square(residual_fun(new_q, args)))

    gradd = jax.grad(total_residual)

    # Use the Gauss-Newton approximation for the Hessian.
    # Want: J^T J p
    def hesspp(new_q, p, args):
      partial_res_fun = lambda q: residual_fun(q, args)
      _, vjp = jax.vjp(partial_res_fun, new_q)
      return vjp(jax.jvp(partial_res_fun, (new_q,), (p,))[1])[0]

    def hesspdirect(new_q, p, args):
      print('direct hessian')
      res_jac = jax.jacfwd(residual_fun)(new_q, args)

      gn = res_jac.T @ res_jac
      print('analyzing G-N matrix:')
      print(f'shape: {gn.shape}')
      print(f'rank: {np.linalg.matrix_rank(gn)}')
      print(f'eigenvals: {np.linalg.eigh(gn)[0]}')
      return res_jac.T @ (res_jac @ p)

    def full_hess(new_q, p, args):
      print('full hess')
      hess = jax.hessian(total_residual)(new_q, args)
      return hess @ p

    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return newtoncg_python(total_residual, gradd, hesspdirect, x0, args)

    return wrapped_optfun

  elif kind == 'newtoncg-scipy':
    def total_residual(new_q, args):
      return 0.5 * np.sum(np.square(residual_fun(new_q, args)))

    gradd = jax.grad(total_residual)

    # Use the Gauss-Newton approximation for the Hessian.
    # Want: J^T J p
    def hesspp(new_q, p, args):
      partial_res_fun = lambda q: residual_fun(q, args)
      _, vjp = jax.vjp(partial_res_fun, new_q)
      return vjp(jax.jvp(partial_res_fun, (new_q,), (p,))[1])[0]

    def hesspdirect(new_q, p, args):
      print('direct hessian')
      res_jac = jax.jacfwd(residual_fun)(new_q, args)

      gn = res_jac.T @ res_jac
      res = res_jac.T @ (res_jac @ p)
      print(f'curvature: {p.T @ res}')
      #print('analyzing G-N matrix:')
      #print(f'shape: {gn.shape}')
      #print(f'rank: {np.linalg.matrix_rank(gn)}')
      #print(f'eigenvals: {np.linalg.eigh(gn)[0]}')
      return res_jac.T @ (res_jac @ p)

    def full_hess(new_q, p, args):
      print('full hess')
      hess = jax.hessian(total_residual)(new_q, args)
      return hess @ p

    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return spopt.minimize(total_residual, x0, args=(args,), method='Newton-CG', jac=gradd,
                            hessp=hesspdirect).x

    return wrapped_optfun

  elif kind == 'trustncg-scipy':
    @jax.jit
    def total_residual(new_q, args):
      return 0.5 * np.sum(np.square(residual_fun(new_q, args)))

    gradd = jax.grad(total_residual)
    gradd = jax.jit(gradd)

    # Use the Gauss-Newton approximation for the Hessian.
    # Want: J^T J p
    @jax.jit
    def hesspp(new_q, p, args):
      partial_res_fun = lambda q: residual_fun(q, args)
      _, vjp = jax.vjp(partial_res_fun, new_q)
      return vjp(jax.jvp(partial_res_fun, (new_q,), (p,))[1])[0]

    def hesspdirect(new_q, p, args):
      print('direct hessian')
      res_jac = jax.jacfwd(residual_fun)(new_q, args)

      gn = res_jac.T @ res_jac
      res = res_jac.T @ (res_jac @ p)
      print(f'curvature: {p.T @ res}')
      print('analyzing G-N matrix:')
      print(f'shape: {gn.shape}')
      print(f'rank: {np.linalg.matrix_rank(gn)}')
      print(f'eigenvals: {np.linalg.eigh(gn)[0]}')
      return res_jac.T @ (res_jac @ p)

    @jax.jit
    def full_hess(new_q, p, args):
      print('full hess')
      hess = jax.hessian(total_residual)(new_q, args)
      return hess @ p

    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return spopt.minimize(total_residual, x0, args=(args,), method='trust-ncg', jac=gradd,
                            hessp=hesspp, tol=1e-8).x

    return wrapped_optfun

  else:
    raise ValueError(f'Unknown LS solver kind {kind}')

