import jax
import jax.numpy as np

from functools import partial


def get_vjp_fun(func):
    """Generate a function that computes the vjp of func at input x and vector p.
    """

    def vjp(x, p, args=()):
        def partial_fun(xin): return func(xin, *args)
        _, vjp_func = jax.vjp(partial_fun, x)
        return vjp_func(p)

    return vjp


def get_JTJv(func):
    """Generate a function that performs J.T @ J @ v for arbitrary input.
    """

    def JTJv(x, v, args=()):
        def partial_fun(xin): return func(xin, *args)
        def jvp_func(v): return partial(jax.jvp, partial_fun, (x,))((v,))[1]
        _, vjp_func = jax.vjp(partial_fun, x)
        return vjp_func(jvp_func(v))[0]

    return JTJv


@jax.jit
def radius_bounds(loc, direction, radius):
    """Compute the intersection of a line and a hypersphere at the origin.

    Args:
      loc: A location assumed to be inside the hypersphere.
      direction: A vector specifying a direction.
      radius: The radius of the hypersphere.

    Returns:
      Computes the two values of alpha such that
          ||loc + alpha*direction|| = radius
      These values are returned in increasing order.
      Returns nans if the loc is outside the hypersphere.

    """

    # This is just computing the quadratic formula.
    a = direction.T @ direction
    b = 2 * direction.T @ loc
    c = loc.T @ loc - radius**2

    disc = np.sqrt(np.maximum(b**2 - 4*a*c, 0.0))
    up = (-b + disc) / (2*a)
    dn = (-b - disc) / (2*a)

    return jax.lax.cond(
        c < 0,
        lambda _: np.sort(np.array([dn, up])),
        lambda _: np.nan * np.ones(2),
        operand=None,
    )
