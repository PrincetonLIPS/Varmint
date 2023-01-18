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

from varmint.geometry.geometry import Geometry, SingleElementGeometry
from varmint.utils.timer import Timer, Time
from varmint.utils.ad_utils import hvp


class SparseNewtonIncrementalSolver:
    def __init__(self, geometry: SingleElementGeometry, loss_fun, base_incs=10,
                 step_size=1.0, tol=1e-8, dev_id=0, save_mats=0, save_mats_path='saved_mats/', print_runtime_stats=False):
        self.iter_num = 0
        self.geometry = geometry
        self.step_size = step_size
        self.tol = tol
        self.save=False
        self.print_runtime_stats = print_runtime_stats

        # This function takes in a vector of increments and forms a dictionary
        # mapping the increments to the boundary conditions.
        self.fixed_locs_from_dict = jax.jit(geometry.fixed_locs_from_dict, device=jax.devices()[dev_id])
        self.tractions_from_dict = jax.jit(geometry.tractions_from_dict, device=jax.devices()[dev_id])
        self.base_incs = base_incs

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

        self.timer = Timer()

        def adjoint_op(xk, adjoint, args):
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]
            return args_bar

        self.adjoint_op = jax.jit(adjoint_op, device=jax.devices()[dev_id])

    def linear_solve(self, xk, args, g, solve_tol=1e-5):
        if self.preconditioner is None:
            with Time(self.timer, 'hessian_construction'):
                data, row_indices, col_indptr = jax.block_until_ready(self.sparse_hess_construct(xk, args))
                sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
            with Time(self.timer, 'lu_decomposition'):
                self.preconditioner = scipy.sparse.linalg.splu(sparse_hess)

        if self.save_mats > 0 and self.saved_mats < self.save_mats:
            print('Saving mat to', os.path.join(self.save_mats_path, f'saved{self.saved_mats}.npz'))
            data, row_indices, col_indptr = self.sparse_hess_construct(xk, args)
            sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))
            scipy.sparse.save_npz(os.path.join(self.save_mats_path, f'saved{self.saved_mats}.npz'), sparse_hess)
            onp.savez(os.path.join(self.save_mats_path, f'saved_grad_{self.saved_mats}.npz'), -g)
            self.saved_mats += 1

        def _precondition_solve(x):
            with Time(self.timer, 'preconditioner_solve'):
                return jax.block_until_ready(self.preconditioner.solve(x).astype(np.float64))
        M = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), _precondition_solve, dtype=np.float64)

        def _loss_hvp(v):
            with Time(self.timer, 'loss_hvp'):
                return jax.block_until_ready(self.loss_hvp(xk, v.astype(np.float64), args))
        A = scipy.sparse.linalg.LinearOperator(
            (g.size, g.size), _loss_hvp, dtype=np.float64)
        with Time(self.timer, 'gmres'):
            res, info = scipy.sparse.linalg.lgmres(A, -g, tol=solve_tol, M=M, inner_m=3, maxiter=5)

        if info > 0:
            with Time(self.timer, 'hessian_construction'):
                data, row_indices, col_indptr = jax.block_until_ready(self.sparse_hess_construct(xk, args))
                sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

            try:
                with Time(self.timer, 'lu_decomposition'):
                    self.preconditioner = scipy.sparse.linalg.splu(sparse_hess)
            except RuntimeError as e:
                print('Found singular matrix. Saving to disk.')
                scipy.sparse.save_npz('singular_matrix.npz', sparse_hess)
                with open('singular_point.npz', 'wb') as f:
                    onp.savez(f, xk)
                with open('singular_args.npz', 'wb') as f:
                    onp.savez(f, *args)
                raise

            def _precondition_solve(x):
                with Time(self.timer, 'preconditioner_solve'):
                    return self.preconditioner.solve(x).astype(np.float64)
            M = scipy.sparse.linalg.LinearOperator(
                (g.size, g.size), _precondition_solve, dtype=np.float64)

            with Time(self.timer, 'gmres'):
                res, info = scipy.sparse.linalg.lgmres(sparse_hess, -g, tol=solve_tol, M=M, maxiter=10)
            if info > 0:
                print('WARNING: Direct ILU factorization was not good enough at 100 iterations.')
        return res

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, increment_dict, tractions_dict, ref_ctrl, mat_params):
            self.timer.reset()

            # Magic numbers to tune
            succ_mult = 1.05
            fail_mult = 0.8
            n_trial_iters = 50
            min_step_size = 0.001
            step_size_anneal = 0.95
            tol_switch_threshold = 0.90
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
                tractions_disp = jax.tree_util.tree_map(
                    lambda x: increment * x, tractions_dict)
                fixed_locs = self.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
                tractions = self.tractions_from_dict(tractions_disp)
                args = (fixed_locs, tractions, ref_ctrl, mat_params)

                xk = x_s
                with Time(self.timer, 'gradient'):
                    g = jax.block_until_ready(self.grad_fun(xk, *args))
                inum = 0
                total_inum = 0
                step_size = self.step_size

                while np.linalg.norm(g) > tol and inum < n_trial_iters:
                    with Time(self.timer, 'linear_solve'):
                        dx = self.linear_solve(xk, args, g, solve_tol=tol)

                    with Time(self.timer, 'loss_fun'):
                        test_loss = jax.block_until_ready(self.loss_fun(xk + dx * step_size, *args))
                    if not np.isnan(test_loss):
                        # Even if we find a step that doesn't get NaN, step 0.8 of the way there.
                        xk = xk + dx * step_size * 0.8
                    elif step_size > min_step_size:
                        xk = x0
                        inum = 0
                        step_size = step_size * step_size_anneal

                    with Time(self.timer, 'gradient'):
                        g = jax.block_until_ready(self.grad_fun(xk, *args))
                    inum = inum + 1
                    total_inum = total_inum + 1

                with Time(self.timer, 'loss_fun'):
                    loss = jax.block_until_ready(self.loss_fun(xk, *args))
                if np.isnan(loss) or total_inum >= n_trial_iters and np.linalg.norm(g) > tol:
                    success = False
                else:
                    success = True

                return xk, success, fixed_locs, tractions

            while solved_increment < (1.0 - 1e-8):
                if proposed_increment < 1e-4:
                    print(f'WARNING: Existed at increment {solved_increment}.', flush=True)
                    if self.print_runtime_stats:
                        self.timer.summarize()
                    return x_inc, all_xs, all_fixed_locs, solved_increment
                increment = min(1.0, solved_increment + proposed_increment)
                if increment > tol_switch_threshold:
                    tol = self.tol
                else:
                    tol = low_tol

                xk, success, fixed_locs, tractions = try_increment(increment, x_inc, tol)
                args = (fixed_locs, tractions, ref_ctrl)

                if success:
                    solved_increment = min(1.0, solved_increment + proposed_increment)

                    x_inc = xk
                    proposed_increment = proposed_increment * succ_mult
                    all_xs.append(x_inc)
                    all_fixed_locs.append(fixed_locs)
                else:
                    proposed_increment = proposed_increment * fail_mult

            if self.print_runtime_stats:
                self.timer.summarize()
            return x_inc, all_xs, all_fixed_locs, solved_increment

        def optimize_fwd(x0, increment_dict, tractions, ref_ctrl, mat_params):
            xk, all_xs, all_fixed_locs, solved_increment = optimize(x0, increment_dict, tractions, ref_ctrl, mat_params)
            fixed_displacements = jax.tree_util.tree_map(
                lambda x: solved_increment * x, increment_dict)

            return (xk, all_xs, all_fixed_locs, solved_increment), (xk, fixed_displacements, tractions, ref_ctrl, mat_params)

        def optimize_bwd(res, g_all):
            xk, increment_dict, tractions, ref_ctrl, mat_params = res
            (g, _, _, _) = g_all

            def preprocess(increment_dict, tractions_dict, ref_ctrl, mat_params):
                fixed_locs = self.fixed_locs_from_dict(ref_ctrl, increment_dict)
                tractions = self.tractions_from_dict(tractions_dict)
                return (fixed_locs, tractions, ref_ctrl, mat_params)

            args, f_vjp = jax.vjp(preprocess, increment_dict, tractions, ref_ctrl, mat_params)

            # Compute adjoint wrt upstream adjoints.
            adjoint = self.linear_solve(xk, args,  -g, solve_tol=self.tol)
            args_bar = self.adjoint_op(xk, adjoint, args)
            increment_dict_bar, tractions_bar, ref_ctrl_bar, mat_params_bar = f_vjp(args_bar)
            return None, increment_dict_bar, tractions_bar, ref_ctrl_bar, mat_params_bar

        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize
