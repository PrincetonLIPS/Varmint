import functools

import jax
import jax.numpy as jnp
import numpy as onp


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

    squared_sum = x.dot(x)
    return jax.lax.cond(
        squared_sum == 0,
        lambda: jnp.zeros_like(squared_sum),
        lambda: jnp.sqrt(squared_sum),
    )


def divide00(num, denom):
    """ Divide such that 0/0 = 0.
    The trick here is to do this in such a way that reverse-mode and forward-mode
    automatic differentation via JAX still work reasonably.
    """

    force_zero = jnp.logical_and(num == 0, denom == 0)
    return jnp.where(force_zero, 0.0, num) \
        / jnp.where(force_zero, 1.0, denom)


def zero_one_sign(arr):
    """Returns an array of the same shape as the input with 
    the value 1. where the input array is greater than or equal 
    to zero and 0. where the input array is less than zero. 

    Parameters
    ----------
    arr : ndarray 
        input array 

    Returns 
    -------
    binary_arr : ndarray 
        result with shape of `arr` and value 1. where arr >= 0. 
        and value 0. otherwise. 
    """
    binary_arr = 0.5 * (1.0 + jnp.sign(arr))
    return binary_arr

