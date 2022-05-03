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

from varmintv2.geometry.geometry import Geometry, SingleElementGeometry

SparseNewtonSolverHCBRestartPreconditionState = namedtuple('SparseNewtonSolverHCBRestartPreconditionState', [
    'x0', 'xk', 'g', 'inum', 'total_inum', 'step_size'
])


class SolveFnContainer:
    lu = None


def hvp(f, primals, tangents, args):
    def f_with_args(x):
        return f(x, *args)
    return jax.jvp(jax.grad(f_with_args), (primals,), (tangents,))[1]


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

        def sparse_hess_construct(x, args):
            vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
            hvp_res = vmap_loss_hvp(x, self.sparsity_tangents, args)
            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
            return data, row_indices, col_indptr

        self.loss_fun = jax.jit(loss_fun)
        self.grad_fun = jax.jit(jax.grad(loss_fun))
        self.loss_hvp = jax.jit(loss_hvp)

        self.hvp_counter = 0
        def hvp_with_counter(*args, **kwargs):
            self.hvp_counter += 1
            return jax.jit(loss_hvp)(*args, **kwargs)
        #self.loss_hvp = hvp_with_counter

        self.sparse_hess_construct = jax.jit(sparse_hess_construct)

        # Probably won't use this, but include it anyway.
        self.hessian_fun = jax.jit(jax.hessian(loss_fun))
        self.preconditioner = None

        def adjoint_op(xk, adjoint, args):
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]
            return args_bar
        
        self.adjoint_op = jax.jit(adjoint_op)
    
    def linear_solve(self, xk, args, g):
        if self.preconditioner is None:
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
            self.preconditioner = scipy.sparse.linalg.spilu(sparse_hess)

        M = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), self.preconditioner.solve)
        A = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), lambda v: self.loss_hvp(xk, v, args))
        res, info = scipy.sparse.linalg.lgmres(A, -g, tol=1e-5, M=M, inner_m=3, maxiter=5)

        if info > 0:
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

            try:
                self.preconditioner = scipy.sparse.linalg.spilu(sparse_hess)
            except RuntimeError as e:
                print('Found singular matrix. Saving to disk.')
                scipy.sparse.save_npz('singular_matrix.npz', sparse_hess)
                with open('singular_point.npz', 'wb') as f:
                    onp.savez(f, xk)
                with open('singular_args.npz', 'wb') as f:
                    onp.savez(f, *args)
                raise
            res, info = scipy.sparse.linalg.lgmres(sparse_hess, -g, tol=1e-5, M=M, maxiter=100)
            if info > 0:
                print('WARNING: Direct ILU factorization was not good enough at 100 iterations.')
        return res

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, args=()):
            xk = x0
            g = self.grad_fun(x0, *args)
            inum = 0
            total_inum = 0
            step_size = self.step_size

            while np.linalg.norm(g) > self.tol and inum < self.max_iter:
                dx = self.linear_solve(xk, args, g)
                xk = xk + dx * step_size

                if np.isnan(self.loss_fun(xk, *args)) and step_size > 0.001:
                    xk = x0
                    inum = 0
                    step_size = step_size * 0.95
                
                g = self.grad_fun(xk, *args)
                inum = inum + 1
                total_inum = total_inum + 1

            return xk

        def optimize_fwd(x0, args=()):
            xk = optimize(x0, args)
            #return (xk, success), (xk, args)

            return xk, (xk, args)
        
        def optimize_bwd(res, g):
            #hcb.id_print(np.array([0, 1, 2, 3, 4]))
            (xk, args) = res

            # Compute adjoint wrt upstream adjoints.
            adjoint = self.linear_solve(xk, args,  -g)
            args_bar = self.adjoint_op(xk, adjoint, args)
            return None, args_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize


class SparseNewtonIncrementalSolver:
    def __init__(self, geometry: SingleElementGeometry, loss_fun, base_incs=10,
                 max_iter=20, step_size=1.0, tol=1e-8, ls_backtrack=0.95, update_every=10, dev_id=0, save_mats=0, save_mats_path='saved_mats/'):
        print(f'Using Newton with backtracking line search. Using GMRES and updating preconditioner every {update_every} steps.')
        self.max_iter = max_iter
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False
        self.ls_backtrack = ls_backtrack
        self.update_every = update_every

        # This function takes in a vector of increments and forms a dictionary
        # mapping the increments to the boundary conditions.
        self.fixed_locs_from_dict = jax.jit(geometry.fixed_locs_from_dict, device=jax.devices()[dev_id])
        self.base_incs = base_incs

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

        def sparse_hess_construct(x, args):
            vmap_loss_hvp = jax.vmap(loss_hvp, in_axes=(None, 0, None))
            hvp_res = vmap_loss_hvp(x, self.sparsity_tangents, args)
            data, row_indices, col_indptr = self.sparse_reconstruct(hvp_res)
            return data, row_indices, col_indptr

        self.loss_fun = jax.jit(loss_fun, device=jax.devices()[dev_id])
        self.grad_fun = jax.jit(jax.grad(loss_fun), device=jax.devices()[dev_id])
        self.loss_hvp = jax.jit(loss_hvp, device=jax.devices()[dev_id])

        self.hvp_counter = 0
        def hvp_with_counter(*args, **kwargs):
            self.hvp_counter += 1
            return jax.jit(loss_hvp, device=jax.devices()[dev_id])(*args, **kwargs)
        #self.loss_hvp = hvp_with_counter

        self.sparse_hess_construct = jax.jit(sparse_hess_construct, device=jax.devices()[dev_id])

        # Probably won't use this, but include it anyway.
        self.hessian_fun = jax.jit(jax.hessian(loss_fun), device=jax.devices()[dev_id])
        self.preconditioner = None
        self.saved_mats = 0
        self.save_mats = save_mats
        self.save_mats_path = save_mats_path

        def adjoint_op(xk, adjoint, args):
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]
            return args_bar
        
        self.adjoint_op = jax.jit(adjoint_op, device=jax.devices()[dev_id])
    
    def linear_solve(self, xk, args, g, solve_tol=1e-5):
        if self.preconditioner is None:
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
            self.preconditioner = scipy.sparse.linalg.spilu(sparse_hess)

        if self.save_mats > 0 and self.saved_mats < self.save_mats:
            print('Saving mat to', os.path.join(self.save_mats_path, f'saved{self.saved_mats}.npz'))
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
            scipy.sparse.save_npz(os.path.join(self.save_mats_path, f'saved{self.saved_mats}.npz'), sparse_hess)
            onp.savez(os.path.join(self.save_mats_path, f'saved_grad_{self.saved_mats}.npz'), -g)
            self.saved_mats += 1

        M = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), self.preconditioner.solve)
        A = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), lambda v: self.loss_hvp(xk, v, args))
        res, info = scipy.sparse.linalg.lgmres(A, -g, tol=solve_tol, M=M, inner_m=3, maxiter=5)

        if info > 0:
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

            try:
                self.preconditioner = scipy.sparse.linalg.spilu(sparse_hess)
            except RuntimeError as e:
                print('Found singular matrix. Saving to disk.')
                scipy.sparse.save_npz('singular_matrix.npz', sparse_hess)
                with open('singular_point.npz', 'wb') as f:
                    onp.savez(f, xk)
                with open('singular_args.npz', 'wb') as f:
                    onp.savez(f, *args)
                raise
            M = scipy.sparse.linalg.LinearOperator(
                (g.size, g.size), self.preconditioner.solve)
            res, info = scipy.sparse.linalg.lgmres(sparse_hess, -g, tol=solve_tol, M=M, maxiter=10)
            if info > 0:
                print('WARNING: Direct ILU factorization was not good enough at 100 iterations.')
        return res

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, increment_dict, tractions, ref_ctrl):
            # Magic numbers to tune
            succ_mult = 1.05
            fail_mult = 0.8
            n_trial_iters = 5
            min_step_size = 0.001
            step_size_anneal = 0.95
            tol_switch_threshold = 0.9
            low_tol = 1e-1

            # `increment_dict` should be a pytree with increments on the appropriate
            # boundaries. tree_map should be used to scale it for incremental solution.
            x_inc = x0
            proposed_increment = 1. / self.base_incs
            solved_increment = 0.0
            all_xs = []
            all_fixed_locs = []

            def try_increment(increment, x_s, tol):
                fixed_displacements = jax.tree_util.tree_map(
                    lambda x: increment * x, increment_dict)
                fixed_locs = self.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
                args = (fixed_locs, tractions, ref_ctrl)

                xk = x_s
                g = self.grad_fun(xk, *args)
                inum = 0
                total_inum = 0
                step_size = self.step_size

                while np.linalg.norm(g) > tol and total_inum < n_trial_iters:
                    dx = self.linear_solve(xk, args, g, solve_tol=tol)
                    xk = xk + dx * step_size

                    if np.isnan(self.loss_fun(xk, *args)) and step_size > min_step_size:
                        xk = x0
                        inum = 0
                        step_size = step_size * step_size_anneal

                    g = self.grad_fun(xk, *args)
                    inum = inum + 1
                    total_inum = total_inum + 1
                if np.isnan(self.loss_fun(xk, *args)) or total_inum >= n_trial_iters and np.linalg.norm(g) > tol:
                    success = False
                else:
                    success = True

                return xk, success, fixed_locs

            while solved_increment < (1.0 - 1e-8):
                increment = min(1.0, solved_increment + proposed_increment)
                if increment > tol_switch_threshold:
                    tol = self.tol
                else:
                    tol = low_tol

                xk, success, fixed_locs = try_increment(increment, x_inc, tol)
                args = (fixed_locs, tractions, ref_ctrl)

                if success:
                    solved_increment = min(1.0, solved_increment + proposed_increment)
                    x_inc = xk
                    proposed_increment = proposed_increment * succ_mult
                    #print(f'solved up to increment {solved_increment}')
                    all_xs.append(x_inc)
                    all_fixed_locs.append(fixed_locs)
                else:
                    proposed_increment = proposed_increment * fail_mult
                    #print(f'failed. new increment is {proposed_increment}')

            return x_inc, all_xs, all_fixed_locs

        def optimize_fwd(x0, increment_dict, tractions, ref_ctrl):
            xk, all_xs, all_fixed_locs = optimize(x0, increment_dict, tractions, ref_ctrl)
            #return (xk, success), (xk, args)

            return (xk, all_xs, all_fixed_locs), (xk, increment_dict, tractions, ref_ctrl)

        def optimize_bwd(res, g_all):
            #import pdb
            xk, increment_dict, tractions, ref_ctrl = res
            (g, _, _) = g_all
            #pdb.set_trace()

            def preprocess(increment_dict, tractions, ref_ctrl):
                fixed_locs = self.fixed_locs_from_dict(ref_ctrl, increment_dict)
                return (fixed_locs, tractions, ref_ctrl)

            args, f_vjp = jax.vjp(preprocess, increment_dict, tractions, ref_ctrl)
            #pdb.set_trace()
            # Compute adjoint wrt upstream adjoints.
            #print('computing adjoint')
            adjoint = self.linear_solve(xk, args,  -g, solve_tol=self.tol)
            args_bar = self.adjoint_op(xk, adjoint, args)
            increment_dict_bar, tractions_bar, ref_ctrl_bar = f_vjp(args_bar)
            return None, increment_dict_bar, tractions_bar, ref_ctrl_bar
        
        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize
