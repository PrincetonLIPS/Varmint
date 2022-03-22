from collections import namedtuple
import os
import gc

import jax
from jax.core import InconclusiveDimensionOperation
import jax.numpy as np
import jax.scipy.linalg

import jax.experimental.host_callback as hcb

import numpy as onp

import numpy.linalg as onpla

import scipy.optimize as spopt
import scipy.sparse
import scipy.sparse.linalg

import scipy.stats

import ilupp

import time
from functools import partial

from sympy import solve

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

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        @jax.jit
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = jax.jit(loss_hvp)
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.jit(jax.grad(loss_fun))


    def optimize(self, x0, args=()):
        xk = x0

        tol = self.tol
        
        for i in range(self.max_iter):
            g = onp.array(self.grad_fun(xk, *args))
            print(f'Gradient norm: {np.linalg.norm(g)}')

            if np.linalg.norm(g) < tol:
                return xk, True

            t1 = time.time()
            hvp_res = self.sparse_entries_fun(xk, args).block_until_ready()
            self.stats['jvps_time'] += time.time() - t1
            self.stats['num_hess_calls'] += 1

            t1 = time.time()
            sparse_hess = self.sparse_reconstruct(hvp_res)
            self.stats['jac_reconstruction_time'] += time.time() - t1

            t1 = time.time()
            lu = scipy.sparse.linalg.splu(sparse_hess)
            #sparse_hess2 = sparse_hess @ sparse_hess
            #ichol = ilupp.ICholTPreconditioner(sparse_hess2, add_fill_in=20004)
            """
            ilu_factor = scipy.sparse.linalg.spilu(sparse_hess)
            """
            self.stats['factorization_time'] += time.time() - t1

            t1 = time.time()
            dx = lu.solve(-g)
            """
            def M_x(x): return ilu_factor.solve(x)
            #def M_x(x): return ichol @ x
            self.M_lin_op = scipy.sparse.linalg.LinearOperator(
                (x0.shape[0], x0.shape[0]), M_x)

            def Jk(v): return self.loss_hvp(xk, v, args)
            def JJk(v): return Jk(Jk(v))
            jac_lin_op = scipy.sparse.linalg.LinearOperator(
                (x0.shape[0], x0.shape[0]), Jk)
            dx, info = scipy.sparse.linalg.gmres(
                jac_lin_op, -g, tol=tol, M=self.M_lin_op, maxiter=200)

            if info != 0:
                print(
                    f'GMRES returned info: {info}. Falling back to direct method.')
                # return np.nan * np.ones_like(x0), None
                lu_factor = scipy.sparse.linalg.splu(sparse_hess)

                dx = lu_factor.solve(-g)
            """
            self.stats['solve_time'] += time.time() - t1

            xk = xk + dx * self.step_size

        g = onp.array(self.grad_fun(xk, *args))

        print(f'Reached max iters. Ended up with norm {np.linalg.norm(g)}')
        return xk, False


# This function runs on the host.
def host_sparse_lu_solve(inputs):
    data, row_indices, col_indptr, g = inputs

    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
    lu = scipy.sparse.linalg.splu(sparse_hess)
    return lu.solve(-g)

def host_sparse_lu_savemat(inputs):
    data, row_indices, col_indptr, g, inum = inputs

    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
    scipy.sparse.save_npz(f'sparse_mats_small/spmat_{inum}.npz', sparse_hess)
    lu = scipy.sparse.linalg.splu(sparse_hess)
    return lu.solve(-g)

# This function runs on the device.
def device_sparse_lu(data, row_indices, col_indptr, g, inum):
    inputs = (data, row_indices, col_indptr, g)
    return hcb.call(host_sparse_lu_solve, inputs,
                    result_shape=g)


SparseNewtonSolverHCBState = namedtuple('NewtonState', [
    'xk', 'g', 'inum', 'total_inum'
])


class SparseNewtonSolverHCBPrint:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                hcb.id_print(np.array([1010101]))
                hcb.id_print(self.loss_fun(state.xk, *args))
                hcb.id_print(np.linalg.norm(state.g))
                hcb.id_print(np.array([2020202]))
                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):
                hcb.id_print(np.array([-2, -2, -2, -2, -2]))
                hcb.id_print(np.array([10000]))
                hcb.id_print(np.sum(state.xk))
                hcb.id_print(np.sum(args[0]))
                hcb.id_print(np.sum(args[2]))
                hcb.id_print(np.max(args[2] - args[0]))
                hcb.id_print(np.array([state.inum, self.max_iter]))
                hcb.id_print(np.array([20000]))

                hvp_res = self.sparse_entries_fun(state.xk, args)
                hcb.id_print(np.sum(hvp_res))
                hcb.id_print(np.array([30000]))

                # Sparse reconstruct gives a csc matrix format.
                data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
                dx = device_sparse_lu(data, row_indices, col_indptr, state.g)
                hcb.id_print(np.array([-1, -1, -1, -1, -1]))

                xk = state.xk + dx * self.step_size
                hcb.id_print(np.sum(xk))
                hcb.id_print(np.array([-10, -10, -10, -10, -10]))

                return SparseNewtonSolverHCBState(
                    xk=xk,
                    g=self.grad_fun(xk, *args),
                    inum=state.inum+1,
                    total_inum=0,
                )
            
            init_val = SparseNewtonSolverHCBState(
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
                total_inum=0
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk #, final_state.inum < self.max_iter

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)


            hcb.id_print(np.array([-3, -3, -3, -3, -3]))
            hcb.id_print(np.array([10000]))
            hcb.id_print(np.sum(xk))
            hcb.id_print(np.sum(args[0]))
            hcb.id_print(np.sum(args[2]))
            hcb.id_print(np.max(args[2] - args[0]))
            hcb.id_print(np.array([20000]))
            hcb.id_print(np.array([-4, -4, -4, -4, -4]))

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res
            # TODO compute adjoint
            hcb.id_print(np.array([40000]))
            hcb.id_print(np.sum(xk))
            hcb.id_print(np.sum(args[0]))
            hcb.id_print(np.sum(args[2]))
            hcb.id_print(np.max(args[2] - args[0]))
            hcb.id_print(np.array([50000]))

            hvp_res = self.sparse_entries_fun(xk, args)
            hcb.id_print(np.sum(hvp_res))
            hcb.id_print(np.array([60000]))

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_sparse_lu(data, row_indices, col_indptr, g)

            #hcb.id_print(np.array([5, 6, 7, 8, 9]))

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize


class SparseNewtonSolverHCB:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):
                hvp_res = self.sparse_entries_fun(state.xk, args)

                # Sparse reconstruct gives a csc matrix format.
                data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
                dx = device_sparse_lu(data, row_indices, col_indptr, state.g, state.inum)

                xk = state.xk + dx * self.step_size

                return SparseNewtonSolverHCBState(
                    xk=xk,
                    g=self.grad_fun(xk, *args),
                    inum=state.inum+1,
                    total_inum=0,
                )
            
            init_val = SparseNewtonSolverHCBState(
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
                total_inum=0,
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk #, final_state.inum < self.max_iter

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res
            # TODO compute adjoint

            hvp_res = self.sparse_entries_fun(xk, args)

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_sparse_lu(data, row_indices, col_indptr, g, -1)

            #hcb.id_print(np.array([5, 6, 7, 8, 9]))

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize


class SparseNewtonSolverHCBLineSearch:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8, ls_backtrack=0.95):
        print('Using Newton with backtracking line search.')
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False
        self.ls_backtrack = ls_backtrack

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):

                hvp_res = self.sparse_entries_fun(state.xk, args)

                # Sparse reconstruct gives a csc matrix format.
                data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
                dx = device_sparse_lu(data, row_indices, col_indptr, state.g, state.inum)


                # Do simple backtracking line search here.
                # Find furthest away point that does not give NaN

                def ls_cond_fun(step_size):
                    xk_try = state.xk + dx * step_size
                    #hcb.id_print(step_size)
                    return np.logical_and(np.isnan(self.loss_fun(xk_try, *args)), step_size > 0.0001)
                def ls_body_fun(step_size):
                    return step_size * self.ls_backtrack
                step_size = jax.lax.while_loop(ls_cond_fun, ls_body_fun, self.step_size)
                xk = state.xk + dx * step_size * 0.8  # Go at 0.95 the step size for some slack.

                #hcb.id_print(np.array([111]))
                #hcb.id_print(np.sum(xk))
                #hcb.id_print(np.array([222]))

                return SparseNewtonSolverHCBState(
                    xk=xk,
                    g=self.grad_fun(xk, *args),
                    inum=state.inum+1,
                    total_inum=0,
                )
            
            init_val = SparseNewtonSolverHCBState(
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
                total_inum=0,
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk #, final_state.inum < self.max_iter

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res
            # TODO compute adjoint

            hvp_res = self.sparse_entries_fun(xk, args)

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_sparse_lu(data, row_indices, col_indptr, g, -1)

            #hcb.id_print(np.array([5, 6, 7, 8, 9]))

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize

SparseNewtonSolverHCBRestartState = namedtuple('SparseNewtonSolverHCBRestartState', [
    'x0', 'xk', 'g', 'inum', 'total_inum', 'step_size'
])


class SparseNewtonSolverHCBRestart:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8, ls_backtrack=0.95):
        print('Using Newton with backtracking line search.')
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False
        self.ls_backtrack = ls_backtrack

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                #hcb.id_print(np.array([111]))
                #hcb.id_print(np.sum(state.xk))
                #hcb.id_print(np.sum(self.loss_fun(state.xk, *args)))
                #hcb.id_print(np.array([222]))

                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):

                hvp_res = self.sparse_entries_fun(state.xk, args)

                # Sparse reconstruct gives a csc matrix format.
                data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
                dx = device_sparse_lu(data, row_indices, col_indptr, state.g, state.inum)

                # Construct the step. If the step gives NaN (true case), restart the optimization.
                xk = state.xk + dx * state.step_size
                return jax.lax.cond(
                    np.logical_and(np.isnan(self.loss_fun(xk, *args)), state.step_size > 0.001),
                    lambda _: SparseNewtonSolverHCBRestartState(
                        x0=state.x0,
                        xk=state.x0,
                        g=self.grad_fun(state.x0, *args),
                        inum=state.inum+1,
                        total_inum=0,
                        step_size=state.step_size * 0.95,
                    ),
                    lambda _: SparseNewtonSolverHCBRestartState(
                        x0=state.x0,
                        xk=xk,
                        g=self.grad_fun(xk, *args),
                        inum=state.inum+1,
                        total_inum=0,
                        step_size=state.step_size,
                    ),
                    operand=None,
                )
            
            init_val = SparseNewtonSolverHCBRestartState(
                x0=x0,
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
                total_inum=0,
                step_size=self.step_size,
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk #, final_state.inum < self.max_iter

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res
            # TODO compute adjoint

            hvp_res = self.sparse_entries_fun(xk, args)

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_sparse_lu(data, row_indices, col_indptr, g, -1)

            #hcb.id_print(np.array([5, 6, 7, 8, 9]))

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize

SparseNewtonSolverHCBRestartPreconditionState = namedtuple('SparseNewtonSolverHCBRestartPreconditionState', [
    'x0', 'xk', 'g', 'inum', 'total_inum', 'step_size', 'lu_precond', 'lu_precond0'
])

class SolveFnContainer:
    lu = None
    def __init__(self, size):
        self.lu = None
        self.size = size

        def M_x_global(x):
            return x
        self.M = scipy.sparse.linalg.LinearOperator(
            (self.size, self.size), lambda x: x)

    def update_linear_op(self, lu):
        print('Updating linear op')

        if self.lu is not None:
            del self.lu
        self.lu = lu

        del self.M.__dict__
        self.M = scipy.sparse.linalg.LinearOperator(
            (self.size, self.size), self.lu.solve)



def host_sparse_lu_factor(inputs):
    t = time.time()
    data, row_indices, col_indptr, L_data_shape, U_data_shape = inputs

    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
    lu = scipy.sparse.linalg.spilu(sparse_hess)

    global solver_container
    solver_container.update_linear_op(lu)
    return 0.0
"""
    # Annoyingly, we have to invert the permutations
    inv_r = np.arange(lu.perm_r.shape[0], dtype=np.int32)
    inv_r = inv_r.at[lu.perm_r].set(inv_r)

    inv_c = np.arange(lu.perm_c.shape[0], dtype=np.int32)
    inv_c = inv_c.at[lu.perm_c].set(inv_c)

    # We also have to convert from csc to csr.
    csr_L = lu.L.tocsr()
    csr_U = lu.U.tocsr()

    # Pad to splu_shape
    L_data_padded = np.pad(csr_L.data, (0, L_data_shape - csr_L.data.size), constant_values=0)
    L_indices_padded = np.pad(csr_L.indices, (0, L_data_shape - csr_L.data.size), constant_values=0)

    U_data_padded = np.pad(csr_U.data, (0, U_data_shape - csr_U.data.size), constant_values=0)
    U_indices_padded = np.pad(csr_U.indices, (0, U_data_shape - csr_U.data.size), constant_values=0)

    print(f'factored {time.time() - t}')
    return ((L_data_padded, L_indices_padded, csr_L.indptr),
            (U_data_padded, U_indices_padded, csr_U.indptr),
            inv_r, lu.perm_c)
"""

# This function runs on the device.
def device_sparse_lu_factor(data, row_indices, col_indptr, splu_shape):
    inputs = (data, row_indices, col_indptr, splu_shape[0][0].shape[0], splu_shape[1][0].shape[0])
    return hcb.call(host_sparse_lu_factor, inputs,
                    result_shape=jax.ShapeDtypeStruct((), np.float64))

def host_preconditioned_gmres(inputs):
    lu_precond, data, row_indices, col_indptr, g = inputs
    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

    #L = scipy.sparse.csr_matrix(lu_precond[0], shape=(g.size, g.size))
    #U = scipy.sparse.csr_matrix(lu_precond[1], shape=(g.size, g.size))
    #inv_r = lu_precond[2]
    #perm_c = lu_precond[3]

    #def M_x(x):
    #    x = x[inv_r]
    #    x = scipy.sparse.linalg.spsolve_triangular(L, x, lower=True, unit_diagonal=True)
    #    x = scipy.sparse.linalg.spsolve_triangular(U, x, lower=False)
    #    return x[perm_c]

    #global solver_container

    global icount
    icount = 0

    def update_icount(x):
        global icount
        icount += 1

    t = time.time()

    #SolveFnContainer.lu = scipy.sparse.linalg.spilu(sparse_hess)
    lu = scipy.sparse.linalg.spilu(sparse_hess)
    M = scipy.sparse.linalg.LinearOperator(
        (g.size, g.size), lu.solve)
    #res, info = scipy.sparse.linalg.gmres(sparse_hess, -g, tol=1e-8, M=solver_container.M, maxiter=20, callback=update_icount)
    #print('solving')
    res, info = scipy.sparse.linalg.gmres(sparse_hess, -g, tol=1e-8, M=M, maxiter=20, callback=update_icount)
    #print(info, icount, f'{time.time() - t}')
    if info > 0:
        # Fall back to LU decomposition
        #print('Falling back to LU.')
        SolveFnContainer.lu = scipy.sparse.linalg.spilu(sparse_hess)

        # Clean super lu object to prevent memory leak.
        #if super_lu is not None:
        #    del super_lu.L
        #    del super_lu.U
        #    del super_lu.perm_c
        #    del super_lu.perm_r
        #print(globals())
        M = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), SolveFnContainer.lu.solve)
        icount = 0
        t = time.time()
        res, info = scipy.sparse.linalg.gmres(sparse_hess, -g, tol=1e-8, M=M, maxiter=100, callback=update_icount)
        #print(info, icount, f'{time.time() - t}')

        if info > 0:
            print('Falling back to standard LU.')

    return res

# This function runs on the device.
def device_preconditioned_gmres(lu_precond, data, row_indices, col_indptr, g):
    inputs = (lu_precond, data, row_indices, col_indptr, g)
    return hcb.call(host_preconditioned_gmres, inputs,
                    result_shape=g)



class SparseNewtonSolverHCBRestartPrecondition:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10):
        print(f'Using Newton with backtracking line search. Using GMRES and updating preconditioner every {update_every} steps.')
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False
        self.ls_backtrack = ls_backtrack
        self.update_every = update_every

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

        # Initialize the global solve function
        #global solver_container
        #solver_container = SolveFnContainer(geometry.n_dof)

    def get_optimize_fn(self, x_test, args_test):
        # Need the shape of the two lu factors. Since they are represented
        # as sparse matrices, we do a test factorization to see what the shape
        # would be.
        hvp_res = self.sparse_entries_fun(x_test, args_test)
        data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
        sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
        lu = scipy.sparse.linalg.spilu(sparse_hess)
        
        self.splu_shape = ((jax.ShapeDtypeStruct((lu.L.data.size + 10000,), lu.L.data.dtype), jax.ShapeDtypeStruct((lu.L.indices.size + 10000,), lu.L.indices.dtype), jax.ShapeDtypeStruct(lu.L.indptr.shape, lu.L.indptr.dtype)),
                           (jax.ShapeDtypeStruct((lu.U.data.size + 10000,), lu.U.data.dtype), jax.ShapeDtypeStruct((lu.U.indices.size + 10000,), lu.U.indices.dtype), jax.ShapeDtypeStruct(lu.U.indptr.shape, lu.U.indptr.dtype)),
                           jax.ShapeDtypeStruct(lu.perm_r.shape, lu.perm_r.dtype),
                           jax.ShapeDtypeStruct(lu.perm_c.shape, lu.perm_c.dtype))

        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                #hcb.id_print(np.array([111]))
                #hcb.id_print(np.sum(state.xk))
                #hcb.id_print(np.sum(self.loss_fun(state.xk, *args)))
                #hcb.id_print(np.array([222]))

                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):
                hvp_res = self.sparse_entries_fun(state.xk, args)

                # Sparse reconstruct gives a csc matrix format.
                data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

                #hcb.id_print(np.array([20000]))
                def f1(_):
                    #hcb.id_print(np.array([30000]))
                    #return device_sparse_lu_factor(data, row_indices, col_indptr, self.splu_shape)
                    return state.lu_precond
                def f2(_):
                    #hcb.id_print(np.array([40000]))
                    return state.lu_precond
                lu_precond = jax.lax.cond(
                    #np.logical_and(state.inum > 0, state.inum % self.update_every == 0),
                    state.inum % self.update_every == 0,
                    lambda _: f1(_),
                    lambda _: f2(_),
                    operand=None,
                )

                dx = device_preconditioned_gmres(lu_precond, data, row_indices, col_indptr, state.g)
                # Construct the step. If the step gives NaN (true case), restart the optimization.
                xk = state.xk + dx * state.step_size

                return jax.lax.cond(
                    np.logical_and(np.isnan(self.loss_fun(xk, *args)), state.step_size > 0.001),
                    lambda _: SparseNewtonSolverHCBRestartPreconditionState(
                        x0=state.x0,
                        xk=state.x0,
                        g=self.grad_fun(state.x0, *args),
                        inum=0,
                        total_inum=state.total_inum+1,
                        step_size=state.step_size * 0.95,
                        lu_precond=state.lu_precond0,
                        lu_precond0=state.lu_precond0,
                    ),
                    lambda _: SparseNewtonSolverHCBRestartPreconditionState(
                        x0=state.x0,
                        xk=xk,
                        g=self.grad_fun(xk, *args),
                        inum=state.inum+1,
                        total_inum=state.total_inum+1,
                        step_size=state.step_size,
                        lu_precond=lu_precond,
                        lu_precond0=state.lu_precond0,
                    ),
                    operand=None,
                )

            #hvp_res = self.sparse_entries_fun(x0, args)
            # Sparse reconstruct gives a csc matrix format.
            #data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
            #hcb.id_print(np.array([10000]))
            #lu_precond = device_sparse_lu_factor(data, row_indices, col_indptr, self.splu_shape)
            lu_precond = 0.0
            init_val = SparseNewtonSolverHCBRestartPreconditionState(
                x0=x0,
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
                total_inum=0,
                step_size=self.step_size,
                lu_precond=lu_precond,
                lu_precond0=lu_precond,
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk #, final_state.inum < self.max_iter

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res
            # TODO compute adjoint

            hvp_res = self.sparse_entries_fun(xk, args)

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_preconditioned_gmres(0.0, data, row_indices, col_indptr, -g)

            #hcb.id_print(np.array([5, 6, 7, 8, 9]))

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize


class ExplicitSolver:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=200000, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol

        self.stats = {
            'factorization_time': 0.0,
            'num_hess_calls': 0,
            'jvps_time': 0.0,
            'jac_reconstruction_time': 0.0,
            'solve_time': 0.0,
        }

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents
        
        def loss_hvp(x, tangents, args):
            return hvp(loss_fun, x, tangents, args)

        vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
        def sparse_entries(x, args):
            return vmap_loss_hvp(x, self.sparsity_tangents, args)

        self.diag_mass_matrix = np.squeeze(geometry.sp_mass_matrix.sum(axis=0))

        self.loss_fun = loss_fun
        self.loss_hvp = loss_hvp
        self.sparse_entries_fun = sparse_entries
        self.grad_fun = jax.grad(loss_fun)

    def get_optimize_fn(self):
        #@jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):
                xk = state.xk - 1./self.diag_mass_matrix * self.step_size * state.g

                return SparseNewtonSolverHCBState(
                    xk=xk,
                    g=self.grad_fun(xk, *args),
                    inum=state.inum+1,
                )
            
            init_val = SparseNewtonSolverHCBState(
                xk=x0,
                g=self.grad_fun(x0, *args),
                inum=0,
            )

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
            return final_state.xk, final_state.inum < self.max_iter, final_state.inum
        return optimize

"""
        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)
            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            (xk, args) = res
            # TODO compute adjoint
            hvp_res = self.sparse_entries_fun(xk, args)
            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)

            # Compute adjoint wrt upstream adjoints.
            adjoint = device_sparse_lu(data, row_indices, col_indptr, g)

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)
"""


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

        tol = self.tol
        
        for i in range(self.max_iter):
            g = onp.array(self.grad_fun(xk, *args))

            if np.linalg.norm(g) < tol:
                return xk, True

            #sparse_hess = scipy.sparse.csc_matrix(self.loss_hess(xk, args))
            dense_hess = self.loss_hess(xk, args)
            #print('Checking for NaN:')
            #print(onp.any(onp.isnan(dense_hess)))
            #print(f'Loss value is: {self.loss_fun(xk, *args)}')

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


NewtonState = namedtuple('NewtonState', [
    'xk', 'g', 'hess', 'inum', 'total_num',
])

class DenseNewtonSolverJittable:
    def __init__(self, geometry: Geometry, loss_fun,
                 max_iter=20, step_size=1.0, tol=1e-8):
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol

        def loss_hess(x, args):
            return jax.hessian(loss_fun)(x, *args)

        self.loss_fun = loss_fun
        self.loss_hess = loss_hess
        self.grad_fun = jax.grad(loss_fun)

    def optimize(self, x0, args=()):
        tol = self.tol

        def cond_fun(state):
            return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)
        
        def body_fun(state):
            dx = jax.numpy.linalg.solve(state.hess, -state.g)

            xk = state.xk + dx * self.step_size

            return NewtonState(
                xk=xk,
                g=self.grad_fun(xk, *args),
                hess=self.loss_hess(xk, args),
                inum=state.inum+1,
            )
        
        init_val = NewtonState(
            xk=x0,
            g=self.grad_fun(x0, *args),
            hess=self.loss_hess(x0, args),
            inum=0,
        )

        final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
        return final_state.xk, final_state.inum < self.max_iter


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
