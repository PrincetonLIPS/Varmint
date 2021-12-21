import os

import jax
from jax.core import InconclusiveDimensionOperation
import jax.numpy as np
import jax.scipy.linalg

import numpy as onp

import numpy.linalg as onpla

import scipy.optimize as spopt
import scipy.sparse
import scipy.sparse.linalg

import scipy.stats

import time
from functools import partial

from varmintv2.geometry.geometry import Geometry


class SuperLUOptimizer:
    def __init__(self, geometry: Geometry, niter=200):
        self.niter = niter
        self.iter_num = 0
        self.geometry = geometry

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents

        self.stats = {
            'factorization_time': 0.0,
            'gmres_time': 0.0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'num_gmres_calls': 0,
            'total_gmres_iters': 0,
        }


    def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
        xk = x0

        vmap_jvp = jax.vmap(jvp_fun, in_axes=(None, 0), out_axes=0)
        tol = 1e-6
        for i in range(self.niter):
            jvp_res = vmap_jvp(
                xk, self.geometry.jac_reconstruction_tangents).block_until_ready()
            rk = residual_fun(xk).block_until_ready()

            current_norm = np.linalg.norm(rk)
            if current_norm < tol:
                print(f"breaking after {i} iterations")
                break

            sjac = self.sparse_reconstruct(jvp_res)
            B = scipy.sparse.linalg.splu(sjac)
            pk = B.solve(onp.array(-rk))

            xk = xk + pk
            self.iter_num += 1
        else:
            print("ran out of iterations")
            print(f"norm ended up {current_norm}\n")

        return xk, None


class LUOptimizer:
    def __init__(self, niter=5):
        self.niter = niter
        self.iter_num = 0

        self.stats = {
            'factorization_time': 0.0,
            'gmres_time': 0.0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'num_gmres_calls': 0,
            'total_gmres_iters': 0,
        }


    def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
        xk = x0

        for i in range(self.niter):
            jac = jac_fun(xk).block_until_ready()
            rk = residual_fun(xk).block_until_ready()

            lu, piv = jax.scipy.linalg.lu_factor(jac)
            pk = jax.scipy.linalg.lu_solve((lu, piv), -rk)

            xk = xk + pk
            self.iter_num += 1

        return xk, None


class ILUPreconditionedOptimizer:
    def __init__(self, geometry: Geometry, niter=20, nsteps=1):
        self.niter = niter
        self.nsteps = nsteps
        self.step_count = 0
        self.geometry = geometry

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents

        self.stats = {
            'factorization_time': 0.0,
            'gmres_time': 0.0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'num_gmres_calls': 0,
            'total_gmres_iters': 0,
        }

    def optimize(self, x0, residual_fun, jvp_fun, jac_fun):
        xk = x0

        tol = 1e-8
        vmap_jvp = jax.vmap(jvp_fun, in_axes=(None, 0), out_axes=0)
        for i in range(self.niter):
            rk = residual_fun(xk)
            current_norm = np.linalg.norm(rk)
            if current_norm < tol:
                break

            t = time.time()
            jvp_res = vmap_jvp(
                xk, self.geometry.jac_reconstruction_tangents).block_until_ready()
            self.stats['jvps_time'] += time.time() - t

            t = time.time()
            sparse_jac = self.sparse_reconstruct(jvp_res)
            self.stats['jac_reconstruction_time'] += time.time() - t

            t = time.time()
            try:
                ilu_factor = scipy.sparse.linalg.spilu(sparse_jac)
            except:
                print('Jacobian singular. Returning NaN.')
                return np.nan * np.ones_like(x0), None
            finally:
                self.stats['factorization_time'] += time.time() - t

            def M_x(x): return ilu_factor.solve(x)
            self.M_lin_op = scipy.sparse.linalg.LinearOperator(
                (x0.shape[0], x0.shape[0]), M_x)

            def Jk(v): return jvp_fun(xk, v)
            jac_lin_op = scipy.sparse.linalg.LinearOperator(
                (x0.shape[0], x0.shape[0]), Jk)

            global icount
            icount = 0

            def update_icount(x):
                global icount
                icount += 1

            t = time.time()
            pk, info = scipy.sparse.linalg.gmres(
                jac_lin_op, -rk, callback=update_icount, tol=tol, M=self.M_lin_op, maxiter=200)
            self.stats['gmres_time'] += time.time() - t
            self.stats['total_gmres_iters'] += icount
            self.stats['num_gmres_calls'] += 1

            if info != 0:
                print(
                    f'GMRES returned info: {info}. Falling back to direct method.')
                # return np.nan * np.ones_like(x0), None
                lu_factor = scipy.sparse.linalg.splu(sparse_jac)

                pk = lu_factor.solve(-rk)
            xk = xk + pk

        self.step_count += 1

        return xk, None


class MutableFunction:
    def __init__(self, func):
        self.func = func

    def __call__(self, q, p, ref_ctrl):
        return self.func(q, p, ref_ctrl)


def hvp(f, primals, tangents, args):
    def f_with_args(x):
        return f(x, *args)
    return jax.jvp(jax.grad(f_with_args), (primals,), (tangents,))[1]


class SparseNewtonSolver:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        @jax.jit
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.jit(jax.grad(loss_fun))


    def optimize(self, x0, args=()):
        xk = x0

        tol = self.tol
        
        for i in range(self.max_iter):
            g = onp.array(self.grad_fun(xk, *args))

            if np.linalg.norm(g) < tol:
                return xk, True

            hvp_res = self.sparse_entries_fun(xk, args)
            sparse_hess = self.sparse_reconstruct(hvp_res)
            lu = scipy.sparse.linalg.splu(sparse_hess)

            dx = lu.solve(-g)
            xk = xk + dx * self.step_size

        g = onp.array(self.grad_fun(xk, *args))

        print(f'Reached max iters. Ended up with norm {np.linalg.norm(g)}')
        return xk, False


class DenseNewtonSolver:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol

        def loss_hess(x, args):
            return jax.hessian(loss_fun)(x, *args)

        self.loss_fun = jax.jit(loss_fun)
        self.loss_hess = jax.jit(loss_hess)
        self.grad_fun = jax.jit(jax.grad(loss_fun))


    def optimize(self, x0, args=()):
        xk = x0

        tol = 1e-8
        
        for i in range(self.max_iter):
            g = onp.array(self.grad_fun(xk, *args))

            if np.linalg.norm(g) < tol:
                return xk, True

            #sparse_hess = scipy.sparse.csc_matrix(self.loss_hess(xk, args))
            dense_hess = self.loss_hess(xk, args)
            print('Checking for NaN:')
            print(onp.any(onp.isnan(dense_hess)))
            print(f'Loss value is: {self.loss_fun(xk, *args)}')

            try:
                lu = scipy.linalg.lu_factor(dense_hess)
                #lu = scipy.sparse.linalg.splu(sparse_hess)

                dx = scipy.linalg.lu_solve(lu, -g)
                #dx = lu.solve(-g)
                xk = xk + dx * self.step_size
            except ValueError:
                print("ValueError")
                return xk, False

        return xk, False
        

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
                direction = - \
                    jax.scipy.sparse.linalg.cg(
                        hessp_fun, grad_fun(q, ref_ctrl))[0]
                q = q + direction
                hessp_fun = update(q)
            end_t = time.time()
            print(
                f'Finished optimization. Took {niters} steps in {end_t - start_t} seconds')

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
