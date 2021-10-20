import os

import jax
from jax.core import InconclusiveDimensionOperation
import jax.numpy as np

import numpy as onp

import numpy.linalg as onpla

import scipy.optimize as spopt
import scipy.sparse.linalg
import scipy.stats

import time
from functools import partial

from varmint.optimization.levmar import get_lmfunc
from varmint.optimization.newtoncg import newtoncg
from varmint.optimization.newtoncg_python import newtoncg_python
#from jaxoptkit.optkit.levmar import get_jittable_lm, lmoptimize

class BaselineOptimizer:
  def __init__(self, niter=5):
    self.niter = niter

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      print('using gmres')
      pk, info = scipy.sparse.linalg.gmres(jac_lin_op, -rk, tol=1e-12)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    return xk, None


class CGOptimizer:
  def __init__(self, niter=5):
    self.niter = niter

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    jac = jac_fun(x0)
    print('maximum eigenvalue:')
    print(np.max(np.linalg.eigvalsh(jac)))


    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: -jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      print('using cg')
      pk, info = scipy.sparse.linalg.cg(jac_lin_op, rk, tol=1e-12)
      if info != 0:
        print(f'CG returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    return xk, None


class ILUOptimizer:
  def __init__(self, niter=5):
    self.niter = niter

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    lu_factor = scipy.sparse.linalg.spilu(jac)
    M_x = lambda x: lu_factor.solve(x)
    self.M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      print('using gmres')
      pk, info = scipy.sparse.linalg.gmres(jac_lin_op, -rk, tol=1e-12)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    return xk, None


class DiagonalEstimationOptimizer:
  def __init__(self, nsteps=5, bandsize=50, k=1000, niter=5):
    self.niter = niter
    self.bandsize = bandsize
    self.nsteps = nsteps
    self.k = k  # Number of draws to estimate a band.
    self.step_count = 0

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    def randjvp(v):
      return jvp_fun(x0, v)
    
    jvp_vmap = jax.jit(jax.vmap(randjvp))
    xk = x0
    N = x0.shape[0]

    if self.step_count % self.nsteps == 0:
      print('Constructing banded approximation.')
      bands = np.zeros((N, self.bandsize * 2 - 1))
      randvecs = onp.random.randn(self.k, N)
      products = jvp_vmap(randvecs)

      for b in range(self.bandsize):
        rolled = np.roll(randvecs, b, axis=1)
        D = np.mean(products * rolled, axis=0)

        bands = jax.ops.index_update(bands, jax.ops.index[:, self.bandsize - b - 1], D)

        if b > 0:
          D_back = np.roll(D, -b)
          bands = jax.ops.index_update(bands, jax.ops.index[:, self.bandsize + b - 1], D_back)

      M_x = lambda x: scipy.linalg.solve_banded((self.bandsize-1, self.bandsize-1), bands.T, x)
      self.M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)
      print('Constructed banded approximation.')

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      print('using minres')
      pk, info = scipy.sparse.linalg.minres(jac_lin_op, -rk, tol=1e-8, M=self.M_lin_op, maxiter=10000)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    self.step_count += 1

    return xk, None



class LearnedDiagonalOptimizer:
  def __init__(self, size, niter=5):
    self.diags = np.ones(size)
    self.niter = niter

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    # Do some iterations of optimization on the diagonal.
    print('Optimizing')
    t1 = time.time()
    N = x0.shape[0]
    opt_jac = lambda v: jvp_fun(x0, v)

    def resid(x, d):
      return np.linalg.norm(opt_jac(x) * d - x)
    
    def loss(x, d):
      return np.mean(jax.vmap(resid, (0, None), 0)(x, d))
    loss_grad = jax.grad(loss, argnums=1)
    jit_loss = jax.jit(loss)
    jit_grad = jax.jit(loss_grad)

    niters = 20
    lr = 0.1
    diags = self.diags
    for i in range(niters):
      z = onp.random.randn(10, N)
      loss_val = jit_loss(z, diags)

      diags = diags - lr * jit_grad(z, diags)
    self.diags = diags
    print(f'Optimization took {time.time() - t1} seconds.')
    print('Solving')

    # Do the GMRES optimization
    xk = x0
    M_x = lambda x: self.diags * x
    M_lin_op = scipy.sparse.linalg.LinearOperator((N, N), M_x)

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((N, N), Jk)

      pk, info = scipy.sparse.linalg.minres(jac_lin_op, -rk, tol=1e-12, M=M_lin_op)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    return xk, None

class InvertEveryFewStepsOptimizer:
  def __init__(self, niter=5, nsteps=20):
    self.niter = niter
    self.nsteps = nsteps
    self.step_count = 0

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    if self.step_count % self.nsteps == 0:
      print('computing jacobian')
      jac = jac_fun(xk).block_until_ready()
      print('computed jacobian')
      # inv_jac = np.linalg.inv(jac)
      # M_x = lambda x: inv_jac @ x
      print('lu factoring scipy')
      lu_factor = scipy.linalg.lu_factor(jac)
      print('done')
      M_x = lambda x: scipy.linalg.lu_solve(lu_factor, x)
      self.M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: -jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      pk, info = scipy.sparse.linalg.cg(jac_lin_op, rk, tol=1e-8, M=self.M_lin_op, maxiter=10000)
      if info != 0:
        print(f'GMRES returned info: {info}. Falling back to direct method.')
        # return np.nan * np.ones_like(x0), None
        jac = jac_fun(xk).block_until_ready()
        lu_factor = jax.scipy.linalg.lu_factor(jac)

        pk = jax.scipy.linalg.lu_solve(lu_factor, -rk)

      xk = xk + pk
    
    self.step_count += 1

    return xk, None

class SparseInvertEveryFewStepsOptimizer:
  def __init__(self, niter=20, nsteps=1, cell=None):
    self.niter = niter
    self.nsteps = nsteps
    self.step_count = 0
    self.cell = cell

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    tol = 1e-8
    vmap_jvp = jax.vmap(jvp_fun, in_axes=(None, 0), out_axes=0)
    for i in range(self.niter):
      rk = residual_fun(xk)
      current_norm = np.linalg.norm(rk)
      if current_norm < tol:
        print('Reached tolerance. Breaking.')
        break

      jvp_res = vmap_jvp(xk, self.cell.sparse_jvps_mat)

      sparse_jac = self.cell.sparse_reconstruct(jvp_res)
      ilu_factor = scipy.sparse.linalg.spilu(sparse_jac) #, fill_factor=5.0)

      M_x = lambda x: ilu_factor.solve(x)
      self.M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)

      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      global icount
      icount = 0
      def update_icount(x):
        global icount
        icount += 1
      pk, info = scipy.sparse.linalg.gmres(jac_lin_op, -rk, callback=update_icount, tol=tol, M=self.M_lin_op, maxiter=20)
      print(f'gmres did {icount} iterations.')

      if info != 0:
        print(f'GMRES returned info: {info}. Falling back to direct method.')
        # return np.nan * np.ones_like(x0), None
        lu_factor = scipy.sparse.linalg.splu(sparse_jac)

        pk = lu_factor.solve(-rk)
      xk = xk + pk
    
    self.step_count += 1

    return xk, None


class InvertOncePerStepOptimizer:
  def __init__(self, niter=5):
    self.niter = niter

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    jac = jac_fun(xk)
    inv_jac = np.linalg.inv(jac)
    M_x = lambda x: inv_jac @ x
    M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)

    for i in range(self.niter):
      rk = residual_fun(xk)
      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      pk, info = scipy.sparse.linalg.gmres(jac_lin_op, -rk, tol=1e-12, M=M_lin_op)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk

    return xk, None


class InvertEveryTimeOptimizer:
  def __init__(self, niter=5, save=False, savedir='/n/fs/mm-iga/Varmint/jacobianssmall/'):
    self.niter = niter
    self.iter_num = 0
    self.savedir = savedir
    self.save = save

    if self.save:
      print(f'Saving jacobians and residuals into {self.savedir}.')
      if not os.path.exists(self.savedir):
        print('\t- Creating directory')
        os.mkdir(self.savedir)

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    for i in range(self.niter):
      jac = jac_fun(xk)
      lu_factor = jax.scipy.linalg.lu_factor(jac)
      print('done')
      M_x = lambda x: jax.scipy.linalg.lu_solve(lu_factor, x)

      rk = residual_fun(xk)

      # Save Jacobians
      if self.save:
        np.save(os.path.join(self.savedir, f'jac{self.iter_num}'), jac)
        np.save(os.path.join(self.savedir, f'res{self.iter_num}'), -rk)

      Jk = lambda v: jvp_fun(xk, v)
      jac_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), Jk)

      M_lin_op = scipy.sparse.linalg.LinearOperator((x0.shape[0], x0.shape[0]), M_x)
      pk, info = scipy.sparse.linalg.gmres(jac_lin_op, -rk, tol=1e-12, M=M_lin_op)
      if info != 0:
        print(f'GMRES returned info: {info}. Exiting.')
        return np.nan * np.ones_like(x0), None

      xk = xk + pk
      self.iter_num += 1

    return xk, None


class SuperLUOptimizer:
  def __init__(self, niter=5):
    self.niter = niter
    self.iter_num = 0

  def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
    xk = x0

    for i in range(self.niter):
      jac = jac_fun(xk).block_until_ready()
      rk = residual_fun(xk).block_until_ready()

      sjac = scipy.sparse.csc_matrix(jac)
      B = scipy.sparse.linalg.splu(sjac)
      pk = B.solve(onp.array(-rk))

      xk = xk + pk
      self.iter_num += 1

    return xk, None


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
