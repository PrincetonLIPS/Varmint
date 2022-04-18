from collections import namedtuple
import os
import gc

import jax
from jax.core import InconclusiveDimensionOperation
import jax.numpy as np
import jax.numpy.linalg as npla
import jax.scipy.linalg
import jax.scipy.sparse.linalg

import jax.experimental.host_callback as hcb

import numpy as onp

import numpy.linalg as onpla

import scipy.optimize as spopt
import scipy.sparse
import scipy.sparse.linalg

import scipy.stats

import time
from functools import partial

from sympy import solve

from varmintv2.geometry.geometry import Geometry

SparseNewtonSolverHCBRestartPreconditionState = namedtuple('SparseNewtonSolverHCBRestartPreconditionState', [
    'x0', 'xk', 'g', 'inum', 'total_inum', 'step_size'
])


class SolveFnContainer:
    lu = None


def hvp(f, primals, tangents, args):
    def f_with_args(x):
        return f(x, *args)
    return jax.jvp(jax.grad(f_with_args), (primals,), (tangents,))[1]


def host_sparse_lu_factor(inputs):
    t = time.time()
    data, row_indices, col_indptr = inputs

    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
    SolveFnContainer.lu = scipy.sparse.linalg.spilu(sparse_hess)

    return 0.0


# This function runs on the device.
def device_sparse_lu_factor(data, row_indices, col_indptr):
    inputs = (data, row_indices, col_indptr)
    return hcb.call(host_sparse_lu_factor, inputs,
                    result_shape=jax.ShapeDtypeStruct((), np.float64))


def host_sparse_lu_solve(inputs):
    g = inputs

    if SolveFnContainer.lu is None:
        return g
    else:
        return SolveFnContainer.lu.solve(g)


# This function runs on the device.
def device_sparse_lu_solve(g):
    inputs = g
    return hcb.call(host_sparse_lu_solve, inputs,
                    result_shape=g)


def host_preconditioned_gmres(inputs):
    data, row_indices, col_indptr, g = inputs
    sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

    global icount
    icount = 0

    def update_icount(x):
        global icount
        icount += 1

    t = time.time()

    if SolveFnContainer.lu is None:
        solve = lambda x: x
    else:
        solve = SolveFnContainer.lu.solve
    M = scipy.sparse.linalg.LinearOperator(
        (g.size, g.size), solve)

    res, info = scipy.sparse.linalg.gmres(sparse_hess, -g, tol=1e-8, M=M, maxiter=20, callback=update_icount)

    if info > 0:
        # Fall back to LU decomposition
        #print('Falling back to LU.')
        SolveFnContainer.lu = scipy.sparse.linalg.spilu(sparse_hess)

        M = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), SolveFnContainer.lu.solve)
        icount = 0
        t = time.time()
        res, info = scipy.sparse.linalg.gmres(sparse_hess, -g, tol=1e-8, M=M, maxiter=100, callback=update_icount)

        if info > 0:
            print('Falling back to standard LU.')

    return res


# This function runs on the device.
def device_preconditioned_gmres(data, row_indices, col_indptr, g):
    inputs = (data, row_indices, col_indptr, g)
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

    def get_optimize_fn(self, x_test, args_test):
        # Need the shape of the two lu factors. Since they are represented
        # as sparse matrices, we do a test factorization to see what the shape
        # would be.
        hvp_res = self.sparse_entries_fun(x_test, args_test)
        data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
        sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
        lu = scipy.sparse.linalg.spilu(sparse_hess)

        @jax.custom_vjp
        def optimize(x0, args=()):
            tol = self.tol

            def cond_fun(state):
                return np.logical_and(np.linalg.norm(state.g) > tol, state.inum < self.max_iter)

            def body_fun(state):
                def hvp_fun(v):
                    return self.loss_hvp(state.xk, v, args)
                
                dx, _ = jax.scipy.sparse.linalg.gmres(hvp_fun, -state.g, tol=1e-8, M=device_sparse_lu_solve, maxiter=20)
                def not_conv_fn():
                    hvp_res = self.sparse_entries_fun(state.xk, args)
                    data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
                    _ = device_sparse_lu_factor(data, row_indices, col_indptr)

                    dx, _ = jax.scipy.sparse.linalg.gmres(hvp_fun, -state.g, tol=1e-8, M=device_sparse_lu_solve, maxiter=20)

                    return state.xk + dx * state.step_size
                
                xk = jax.lax.cond(
                    npla.norm(hvp_fun(dx) + state.g) < 1e-8 * npla.norm(state.g),
                    lambda: state.xk + dx * state.step_size,
                    not_conv_fn
                )

                # Sparse reconstruct gives a csc matrix format.
                #dx = device_preconditioned_gmres(data, row_indices, col_indptr, state.g)

                # Construct the step. If the step gives NaN (true case), restart the optimization.
                #xk = state.xk + dx * state.step_size

                return jax.lax.cond(
                    np.logical_and(np.isnan(self.loss_fun(xk, *args)), state.step_size > 0.001),
                    lambda _: SparseNewtonSolverHCBRestartPreconditionState(
                        x0=state.x0,
                        xk=state.x0,
                        g=self.grad_fun(state.x0, *args),
                        inum=0,
                        total_inum=state.total_inum+1,
                        step_size=state.step_size * 0.95,
                    ),
                    lambda _: SparseNewtonSolverHCBRestartPreconditionState(
                        x0=state.x0,
                        xk=xk,
                        g=self.grad_fun(xk, *args),
                        inum=state.inum+1,
                        total_inum=state.total_inum+1,
                        step_size=state.step_size,
                    ),
                    operand=None,
                )

            init_val = SparseNewtonSolverHCBRestartPreconditionState(
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
            def hvp_fun(v):
                return self.loss_hvp(xk, v, args)

            # TODO compute adjoint

            hvp_res = self.sparse_entries_fun(xk, args)

            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)


            # Compute adjoint wrt upstream adjoints.
            adjoint = device_preconditioned_gmres(data, row_indices, col_indptr, -g)
            #adjoint, _ = jax.scipy.sparse.linalg.gmres(hvp_fun, -g, tol=1e-8, M=device_sparse_lu_solve, maxiter=20)
            #def not_conv_fn():
            #    hvp_res = self.sparse_entries_fun(xk, args)
            #    data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
            #    _ = device_sparse_lu_factor(data, row_indices, col_indptr)

            #    adjoint, _ = jax.scipy.sparse.linalg.gmres(hvp_fun, -g, tol=1e-8, M=device_sparse_lu_solve, maxiter=20)

            #    return adjoint

            #adjoint = jax.lax.cond(
            #    npla.norm(hvp_fun(adjoint) + g) < 1e-8 * npla.norm(g),
            #    lambda: adjoint,
            #    not_conv_fn
            #)

            # Take vjp wrt args
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]

            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize