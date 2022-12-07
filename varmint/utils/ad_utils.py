import functools

import jax
import jax.numpy as jnp
import numpy as onp

import varmint.utils.typing


def hvp(f, primals, tangents, args):
    def f_with_args(x):
        return f(x, *args)
    return jax.jvp(jax.grad(f_with_args), (primals,), (tangents,))[1]


def custom_norm(x): 
    """Utility function for computing the 2-norm of an array `x` 
    in a method that is safe under differentiation/tracing. 

    Parameters
    ----------
    x : ndarray
        array for which to compute the 2-norm. 

    Returns
    -------
    norm : float 
        2-norm of x. 
    """

    return jax.lax.cond(
        squared_sum == 0,
        lambda _: 0.0,
        lambda x: jnp.sqrt(x),
        operand=x.dot(x),
    )


def divide00(num, denom):
    """ Divide such that 0/0 = 0.
    The trick here is to do this in such a way that reverse-mode and forward-mode
    automatic differentation via JAX still work reasonably.
    """

    force_zero = np.logical_and(num == 0, denom == 0)
    return np.where(force_zero, 0.0, num) \
        / np.where(force_zero, 1.0, denom)
