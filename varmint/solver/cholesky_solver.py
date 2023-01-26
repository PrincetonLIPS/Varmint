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

import sksparse.cholmod as cholmod


class SparseCholeskyLinearSolver:
    def __init__(self, geometry, loss_fun, dev_id=0):
        self.geometry = geometry

        self.sparse_reconstruct = geometry.get_jac_reconstruction_fn()
        self.sparsity_tangents = geometry.jac_reconstruction_tangents

        self.fixed_locs_from_dict = jax.jit(geometry.fixed_locs_from_dict, device=jax.devices()[dev_id])
        self.tractions_from_dict = jax.jit(geometry.tractions_from_dict, device=jax.devices()[dev_id])

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

        self.sparse_hess_construct = jax.jit(sparse_hess_construct, device=jax.devices()[dev_id])

        self.timer = Timer()

        def adjoint_op(xk, adjoint, args):
            def partial_grad(args):
                return self.grad_fun(xk, *args)
            
            _, f_pgrad = jax.vjp(partial_grad, args)
            args_bar = f_pgrad(-adjoint)[0]
            return args_bar

        self.adjoint_op = jax.jit(adjoint_op, device=jax.devices()[dev_id])
        self.factor = None

    def get_optimize_fn(self):
        @jax.custom_vjp
        def optimize(x0, increment_dict, tractions_dict, ref_ctrl, mat_params):
            self.timer.reset()

            fixed_locs = self.fixed_locs_from_dict(ref_ctrl, increment_dict)
            tractions = self.tractions_from_dict(tractions_dict)
            args = (fixed_locs, tractions, ref_ctrl, mat_params)

            with Time(self.timer, 'hessian_construction'):
                data, row_indices, col_indptr = jax.block_until_ready(self.sparse_hess_construct(x0, args))
                sparse_hess = scipy.sparse.csc_matrix((data, row_indices, col_indptr))

            with Time(self.timer, 'gradient'):
                g = jax.block_until_ready(self.grad_fun(x0, *args))

            if self.factor is None:
                with Time(self.timer, 'analysis'):
                    self.factor = cholmod.analyze(sparse_hess)
            
            with Time(self.timer, 'solving'):
                self.factor.cholesky_inplace(sparse_hess)
                dx = self.factor(-g)

            return x0 + dx

        def optimize_fwd(x0, increment_dict, tractions, ref_ctrl, mat_params):
            xk = optimize(x0, increment_dict, tractions, ref_ctrl, mat_params)

            return xk, (xk, increment_dict, tractions, ref_ctrl, mat_params)

        def optimize_bwd(res, g):
            xk, increment_dict, tractions, ref_ctrl, mat_params = res

            def preprocess(increment_dict, tractions_dict, ref_ctrl, mat_params):
                fixed_locs = self.fixed_locs_from_dict(ref_ctrl, increment_dict)
                tractions = self.tractions_from_dict(tractions_dict)
                return (fixed_locs, tractions, ref_ctrl, mat_params)

            args, f_vjp = jax.vjp(preprocess, increment_dict, tractions, ref_ctrl, mat_params)

            # Compute adjoint wrt upstream adjoints.
            adjoint = self.factor(g)
            args_bar = self.adjoint_op(xk, adjoint, args)
            increment_dict_bar, tractions_bar, ref_ctrl_bar, mat_params_bar = f_vjp(args_bar)
            return None, increment_dict_bar, tractions_bar, ref_ctrl_bar, mat_params_bar

        optimize.defvjp(optimize_fwd, optimize_bwd)

        return optimize
