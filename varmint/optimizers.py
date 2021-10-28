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


class ILUPreconditionedOptimizer:
    def __init__(self, niter=20, nsteps=1, cell=None):
        self.niter = niter
        self.nsteps = nsteps
        self.step_count = 0
        self.cell = cell

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
                xk, self.cell.sparse_jvps_mat).block_until_ready()
            self.stats['jvps_time'] += time.time() - t

            t = time.time()
            sparse_jac = self.cell.sparse_reconstruct(jvp_res)
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
