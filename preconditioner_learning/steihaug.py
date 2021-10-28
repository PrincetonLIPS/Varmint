from functools import partial
import jax
import jax.numpy as np
import jax.numpy.linalg as npla

import matplotlib.pyplot as plt

from utils import radius_bounds

from collections import namedtuple


def _cg_steihaug_nojit(grad, x, delta, epsilon, Hv_func, precond, hv_args=()):
    D_apply, D_T_apply = precond

    z = np.zeros_like(x)
    grad = D_T_apply(grad)

    def Hv_func_precond(v, p):
        return D_T_apply(Hv_func(v, D_apply(p), args=hv_args))

    r = grad
    d = -r
    ncalls = 0

    if npla.norm(r) < epsilon:
        return x, False

    while True:
        Bd = Hv_func_precond(x, d)
        ncalls += 1

        dBd = d.T @ Bd

        if dBd <= 0.0:
            print('getting negative curvature')
            print('curvature:', dBd)
            # Compute the two boundaries using this direction.
            alpha_dn, alpha_up = radius_bounds(z, d, delta)

            x_dn = z + alpha_dn*d
            x_up = z + alpha_up*d

            f_dn = 0.5*x_dn.T @ Hv_func_precond(x, x_dn) + x_dn.T @ grad
            f_up = 0.5*x_up.T @ Hv_func_precond(x, x_up) + x_up.T @ grad

            if f_dn < f_up:
                return x + D_apply(x_dn), True, ncalls
            else:
                return x + D_apply(x_up), True, ncalls

        rTr = r.T @ r
        alpha = rTr / dBd
        z_new = z + alpha * d

        if npla.norm(z_new) >= delta:
            print('out of bounds here')
            alpha_dn, alpha_up = radius_bounds(z, d, delta)
            x_up = z + alpha_up * d
            return x + D_apply(x_up), True, ncalls

        r_new = r + alpha * Bd

        if npla.norm(r_new) < epsilon:
            return x + D_apply(z_new), False, ncalls

        beta = (r_new.T @ r_new) / rTr
        d = - r_new + beta * d
        z = z_new
        r = r_new


CGSteihaugState = namedtuple('CGSteihaugState', [
    'x', 'd', 'z', 'r', 'd_old', 'z_old', 'r_old', 'dBd', 'nHvcalls'
])


def _cg_steihaug_jit(grad, x, delta, epsilon, Hv_func, precond, hv_args=()):
    #grad = grad / precond
    grad = precond(grad)

    def Hv_func_precond(v, p):
        return precond(Hv_func(v, precond(p), args=hv_args))
        # return 1. / precond * Hv_func(v, p / precond, args=hv_args)

    Bd = Hv_func_precond(x, -grad)
    dBd = -grad.T @ Bd

    init_state = CGSteihaugState(
        x=x,
        z=np.zeros_like(x),
        r=grad,
        d=-grad,
        z_old=np.zeros_like(x),
        r_old=grad,
        d_old=-grad,
        dBd=dBd,
        nHvcalls=0,
    )

    def negative_curvature_cond(state):
        return state.dBd <= 0

    def negative_curvature_body(state):
        # Compute the two boundaries using this direction.
        alpha_dn, alpha_up = radius_bounds(state.z_old, state.d_old, delta)

        dx_dn = state.z_old + alpha_dn * state.d_old
        dx_up = state.z_old + alpha_up * state.d_old

        f_dn = 0.5 * dx_dn.T @ Hv_func_precond(x, dx_dn) + dx_dn.T @ grad
        f_up = 0.5 * dx_up.T @ Hv_func_precond(x, dx_up) + dx_up.T @ grad

        return jax.lax.cond(
            f_dn < f_up,
            lambda _: (x + precond(dx_dn), True, state),
            lambda _: (x + precond(dx_up), True, state),
            operand=None,
        )

    def out_of_region_cond(state):
        return npla.norm(state.z) >= delta

    def out_of_region_body(state):
        _, alpha_up = radius_bounds(state.z_old, state.d_old, delta)
        dx_up = state.z_old + alpha_up * state.d_old
        return state.x + precond(dx_up), True, state

    def converged_cond(state):
        return npla.norm(state.r) < epsilon

    def converged_body(state):
        return state.x + precond(state.z), False, state

    def loop_body(state):
        Bd = Hv_func_precond(state.x, state.d)
        dBd = state.d.T @ Bd

        rTr = state.r.T @ state.r
        alpha = rTr / dBd

        z_new = state.z + alpha * state.d
        r_new = state.r + alpha * Bd
        beta = (r_new.T @ r_new) / rTr
        d_new = -r_new + beta * state.d

        return CGSteihaugState(
            x=x,
            z=z_new,
            r=r_new,
            d=d_new,
            z_old=state.z,
            r_old=state.r,
            d_old=state.d,
            dBd=dBd,
            nHvcalls=state.nHvcalls + 1,
        )

    def main_loop(init_state):
        final_state = jax.lax.while_loop(
            lambda s: np.logical_not(
                np.logical_or(
                    negative_curvature_cond(s),
                    np.logical_or(
                        out_of_region_cond(s),
                        converged_cond(s)
                    )
                )
            ),
            loop_body,
            init_state,
        )

        # Is there a cleaner way to do this?? :(
        ind = jax.lax.cond(converged_cond(final_state),
                           lambda _: 2, lambda _: -1, None)
        ind = jax.lax.cond(out_of_region_cond(final_state),
                           lambda _: 1, lambda _: ind, None)
        ind = jax.lax.cond(negative_curvature_cond(
            final_state), lambda _: 0, lambda _: ind, None)

        funcs = [
            negative_curvature_body,
            out_of_region_body,
            converged_body,
        ]

        return jax.lax.switch(ind, funcs, final_state)

    # We must check whether we converged in the very beginning, otherwise
    # we might run into a case like grad = 0, in which case we would accidentally
    # report negative curvature.
    return jax.lax.cond(
        converged_cond(init_state),
        converged_body,
        main_loop,
        operand=init_state,
    )


def get_cg_steihaug_solver(Hv_func, precond=False, jittable=True):
    if jittable:
        fn = _cg_steihaug_jit
    else:
        fn = _cg_steihaug_nojit

    if precond:
        return partial(fn, Hv_func=Hv_func)
    else:
        return partial(fn, Hv_func=Hv_func, precond=lambda x: x)
