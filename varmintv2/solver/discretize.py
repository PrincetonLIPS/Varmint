from typing import Callable
import jax
import time
import jax.numpy as np

from varmintv2.geometry.geometry import Geometry
from varmintv2.utils.vmap_utils import map_jacfwd

from varmintv2.solver.optimization import (
    ILUPreconditionedOptimizer,
    SuperLUOptimizer,
)


def discretize_eulag(L):
    def Ld(q1, q2, dt):
        q = (q1 + q2) / 2
        qdot = (q2 - q1) / dt
        return L(q, qdot)

    grad_Ld_q1 = jax.grad(Ld, argnums=0)
    grad_Ld_q2 = jax.grad(Ld, argnums=1)

    def DEL(q1, t1, q2, t2, q3, t3):
        return grad_Ld_q1(q2, q3, t3-t2) + grad_Ld_q2(q1, q2, t2-t1)

    return DEL


def discretize_hamiltonian(L):
    def Ld(q1, q2, dt, args):
        q = (q1 + q2) / 2
        qdot = (q2 - q1) / dt
        return L(q, qdot, *args)

    grad_Ld_q1 = jax.grad(Ld, argnums=0)
    grad_Ld_q2 = jax.grad(Ld, argnums=1)

    return grad_Ld_q1, grad_Ld_q2


class HamiltonianStepper:
    def __init__(self, L: Callable, geometry: Geometry,
                 F: Callable = None, save: bool = False):
        # For thinking about forces, see West thesis:
        # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
        # Page 16, Sec 1.5.6.
        # Could include in optimization or in momentum update, I think.
        # Momentum update seems much easier.
        self.D0_Ld, self.D1_Ld = discretize_hamiltonian(L)
        self.F = F
        self.save = save
        self.L = L
        self.geometry = geometry

    def residual_fun(self, new_q, args):
        old_q, p, dt, l_args = args

        if self.F is None:
            return p + self.D0_Ld(old_q, new_q, dt, l_args)
        else:
            q = (old_q + new_q)/2.0
            qdot = (new_q-old_q) / dt

            return p + self.D0_Ld(old_q, new_q, dt, l_args) + self.F(q, qdot, *l_args)

    def construct_stepper(self, size, strategy, strategy_params={}):
        if strategy == 'ilu_preconditioning':
            print('Using ILU Preconditioning.')
            optimizer = ILUPreconditionedOptimizer(geometry=self.geometry)
        elif strategy == 'superlu':
            print('Using SuperLUOptimizer')
            optimizer = SuperLUOptimizer()
        else:
            raise ValueError(
                f'Unsupported preconditioning strategy: {strategy}.')

        @jax.jit
        def jit_resid(xk, args):
            return self.residual_fun(xk, args)

        @jax.jit
        def jit_res_jvp(xk, v, args):
            def res(q):
                return self.residual_fun(q, args)
            out, deriv = jax.jvp(res, (xk,), (v,))
            return deriv

        @jax.jit
        def jit_jac(xk, args):
            def res(q):
                return self.residual_fun(q, args)
            return map_jacfwd(res, size)(xk)
            # return jax.jacfwd(res)(xk)

        def step_q(q, p, dt, args):
            def res_fun(xk):
                return jit_resid(xk, (q, p, dt, args))

            global ncalls
            ncalls = 0

            def jvp(xk, v):
                global ncalls
                ncalls += 1
                return jit_res_jvp(xk, v, (q, p, dt, args))

            def jac(xk):
                return jit_jac(xk, (q, p, dt, args))

            new_q, _ = optimizer.optimize(
                jax.lax.stop_gradient(q), res_fun, jvp, jac)
            return new_q, None

        print(f'Using CPU hybrid strategies. End to end compilation not supported.')

        def step_p(q1, q2, dt, args):
            if self.F is None:
                return self.D1_Ld(q1, q2, dt, args)
            else:
                q = (q1 + q2) / 2
                qdot = (q2 - q1) / dt

                return self.D1_Ld(q1, q2, dt, args) + self.F(q, qdot, *args)

        @jax.jit
        def update_p(new_q, q, p, dt, *args):
            return jax.lax.cond(
                np.all(np.isfinite(new_q)),
                lambda _: step_p(q, new_q, dt, args),
                lambda _: np.ones_like(p) + np.nan,
                np.float64(0.0),
            )

        def stepper(q, p, dt, *args):
            new_q, aux = step_q(q, p, dt, args)
            new_p = update_p(new_q, q, p, dt, *args)

            return new_q, new_p

        return stepper, optimizer
