import jax
import jax.numpy as np

import numpy.linalg as onpla

import scipy.optimize as spopt
import scipy.stats

import time
from functools import partial

from varmint.optimization.levmar import get_lmfunc
from varmint.optimization.newtoncg import newtoncg
from varmint.optimization.newtoncg_python import newtoncg_python
#from jaxoptkit.optkit.levmar import get_jittable_lm, lmoptimize

def get_optfun(residual_fun, kind='levmar', **optargs):
  if kind == 'levmar':
    maxiters = optargs.get('maxiters', 50)
    lmfunc = get_lmfunc(residual_fun, maxiters=maxiters)
    lmfunc = jax.jit(lmfunc)

    # We would like all the optimizer functions
    # to have the same signature. 
    def wrapped_lmfunc(x0, args):
      return lmfunc(x0, *args)
    return wrapped_lmfunc
  elif kind == 'levmarnewnojit':
    print('using new optimizer!')
    maxiters = optargs.get('maxiters', 100)

    # We would like all the optimizer functions
    # to have the same signature. 
    def wrapped_lmfunc(x0, args):
      return lmoptimize(residual_fun, x0, maxiters=maxiters, args=args)
    return wrapped_lmfunc
  
  elif kind == 'justnewton':
    print('using standard Newton method for nonlinear equations.')
    tol = 1e-12
    jac = jax.jacfwd(residual_fun)
    def newton_opt(x0, args):
      xk = x0
      print(f'Initial norm: {np.linalg.norm(residual_fun(xk, *args))}')

      for _ in range(10):
        print('starting iteration')
        t0 = time.time()
        rk = residual_fun(xk, *args)
        print(f'got residual at {time.time() - t0}')
        Jk = jac(xk, *args)
        print(f'got jacobian at {time.time() - t0}')


        pk = onpla.solve(Jk, -rk)
        print(f'got solution at {time.time() - t0}')

        xk = xk + pk
        current_norm = np.linalg.norm(residual_fun(xk, *args))
        print(f'Current norm: {current_norm}')
        if current_norm < tol:
          print('Reached tolerance. Breaking.')
          break
      print('Final norm:')
      print(np.linalg.norm(residual_fun(xk, *args)))
      return xk, None
    
    return newton_opt
  elif kind == 'justnewtonjit':
    jac = jax.jacfwd(residual_fun)
    def newton_opt(x0, args):
      xk = x0

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk = np.linalg.solve(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk = np.linalg.solve(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk = np.linalg.solve(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk = np.linalg.solve(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk = np.linalg.solve(Jk, -rk)
      xk = xk + pk

      return xk, None
    return newton_opt

  elif kind == 'justnewtonjitgmres':
    jac = jax.jacfwd(residual_fun)
    def newton_opt(x0, args):
      xk = x0

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk, _ = jax.scipy.sparse.linalg.gmres(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk, _ = jax.scipy.sparse.linalg.gmres(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk, _ = jax.scipy.sparse.linalg.gmres(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk, _ = jax.scipy.sparse.linalg.gmres(Jk, -rk)
      xk = xk + pk

      rk = residual_fun(xk, *args)
      Jk = jac(xk, *args)
      pk, _ = jax.scipy.sparse.linalg.gmres(Jk, -rk)
      xk = xk + pk

      return xk, None
    return newton_opt

  elif kind == 'levmarnew':
    print('using new optimizer!')
    maxiters = optargs.get('maxiters', 100)
    lmfunc = get_jittable_lm(residual_fun, maxiters=maxiters)
    lmfunc = jax.jit(lmfunc)

    # We would like all the optimizer functions
    # to have the same signature. 
    def wrapped_lmfunc(x0, args):
      return lmfunc(x0, args)
    return wrapped_lmfunc

  elif kind == 'scipy-lm':
    residual_fun = jax.jit(residual_fun)
    jac = jax.jacfwd(residual_fun)
    def wrapped_optfun(x0, args):
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
      #print('direct hessian')
      res_jac = jax.jacfwd(residual_fun)(new_q, args)

      gn = res_jac.T @ res_jac
      res = res_jac.T @ (res_jac @ p)
      #print(f'curvature: {p.T @ res}')
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
      #print(f'curvature: {p.T @ res}')
      #print('analyzing G-N matrix:')
      #print(f'shape: {gn.shape}')
      #print(f'rank: {np.linalg.matrix_rank(gn)}')
      #print(f'eigenvals: {np.linalg.eigh(gn)[0]}')
      return res_jac.T @ (res_jac @ p)

    @jax.jit
    def gndirect(new_q, args):
      print('full hess')
      res_jac = jax.jacfwd(residual_fun)(new_q, args)
      return res_jac.T @ res_jac

    def wrapped_optfun(x0, args, jac, jacp, hess, hessp):
      return spopt.minimize(total_residual, x0, args=(args,), method='trust-ncg', jac=gradd,
                            hessp=hesspp, tol=1e-8).x

    return wrapped_optfun

  else:
    raise ValueError(f'Unknown LS solver kind {kind}')


class MutableFunction:
  def __init__(self, func):
    self.func = func

  def __call__(self, q, p, ref_ctrl):
    return self.func(q, p, ref_ctrl)


def get_statics_optfun(loss_fun, grad_fun, hessp_gen_fun=None, kind='newton', optargs={}):
  if kind == 'newton':
    niters = optargs.get('niters', 10)

    def solve(q, ref_ctrl):
      # Try pure Newton iterations
      print('Beginning optimization with Newton solver...')
      start_t = time.time()

      def update(q):
        fun = hessp_gen_fun(q, ref_ctrl)
        def wrap(p):
          return fun(q, p, ref_ctrl)
        return wrap

      hessp_fun = update(q)

      for i in range(niters):
        print(f'Loss: {loss_fun(q, ref_ctrl)}')
        direction = -jax.scipy.sparse.linalg.cg(hessp_fun, grad_fun(q, ref_ctrl))[0]
        q = q + direction
        hessp_fun = update(q)
      end_t = time.time()
      print(f'Finished optimization. Took {niters} steps in {end_t - start_t} seconds')

      return q

    return solve
  elif kind in ['newtoncg-scipy', 'trustncg-scipy', 'bfgs-scipy']:
    if kind == 'newtoncg-scipy':
      method = 'Newton-CG'
    elif kind == 'trustncg-scipy':
      method = 'trust-ncg'
    elif kind == 'bfgs-scipy':
      method = 'bfgs'

    def solve(q, ref_ctrl):
      print(f'Beginning scipy optimization with {method} solver...')
      start_t = time.time()
      hessp_fun = MutableFunction(hessp_gen_fun(q, ref_ctrl))
      def callback(q):
        print('Iteration. Updating hessp.')
        hessp_fun.func = hessp_gen_fun(q, ref_ctrl)

      optim = spopt.minimize(loss_fun, q, args=(ref_ctrl,), method=method,
                             callback=callback, jac=grad_fun, hessp=hessp_fun)
      new_q = optim.x
      if not hasattr(optim, 'nhev'):
        optim.nhev = 0
      print(f'Optimization results:'
            f'\n\tSuccess: {optim.success}'
            f'\n\tMessage: {optim.message}'
            f'\n\tFinal loss: {optim.fun}'
            f'\n\tNumber of fun/grad/hess evals: {optim.nfev}/{optim.njev}/{optim.nhev}')
      end_t = time.time()
      print(f'Finished optimization. Took {end_t - start_t} seconds')

      return new_q

    return solve
  else:
    raise ValueError(f'Unknown statics solver kind {kind}')
