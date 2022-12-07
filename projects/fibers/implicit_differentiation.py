"""This module contains procedures for implicit differentiation setups in 
conjunction with fiber sampling applications.

Taken directly from fibers/src/implicit_differentiation.py
"""
import functools 

import jax 
import jax.numpy as jnp
from jaxopt.implicit_diff import custom_root 

from varmint.utils.typing import *
from varmint.utils.ad_utils import divide00


def bisect(f: callable, fiber: ndarray, num_iterations: int=10) -> float: 
    """Batchable bisection solver running for a specified number of steps."""

    interpolant: callable = lambda x: fiber[0] + x * (fiber[1] - fiber[0])
    h: callable = lambda x: f(interpolant(x))

    # --- standardize so the 'left' endpoint has negative value 
    endpoints: ndarray = jax.lax.cond(h(0.) > 0., lambda _: jnp.array([1., 0]), lambda _: jnp.array([0., 1.]), operand=None)

    def _bisect(endpoints: ndarray) -> float: 
        left, right = endpoints 
        midpoint: float = (left + right) / 2. 
        return jax.lax.cond(h(midpoint) < 0., lambda _: jnp.array([midpoint, right]), lambda _: jnp.array([left, midpoint]), operand=midpoint)

    for _ in range(num_iterations): 
        endpoints: ndarray = _bisect(endpoints)

    return interpolant(endpoints[0])


def on_fiber(fiber: ndarray, x: ndarray) -> float:
    projection: callable = lambda u, v: (u.dot(v) / jnp.linalg.norm(v)) * v
    projection_vector: ndarray = projection(x, fiber[1] - fiber[0]) - x
    distance_from_fiber: ndarray = projection_vector.dot(projection_vector)
    return distance_from_fiber


def bind_optimality_condition(f: callable) -> ndarray: 
    def optimality_condition(x: ndarray, params: dict, fiber: ndarray) -> ndarray: 
        """Computes the ndarray of length 2 representing the vector-value of the 
        `constraint function` which takes on the value of the zero-vector when 
        the constrain is satisfied. 

        Parameters 
        ----------
        x: ndarray 
            ndarray (of length 2, nominally), representing the point in the domain 
            of `scalar_field` to be evaluated. 
        params: tuple 
            auxiliary parameters for `scalar_field`. 
        fiber: ndarray 
            fiber included as the second constraint; the constraint is zero when 
            `x` lies on this fiber, otherwise the value of the constraint is the 
            distance between `x` and the fiber. 

        Returns 
        -------
        constraint_value: ndarray 
            first element is the value of the first constraint (i.e., the value of 
            the scalar field evaluated at x); second element is the distance between 
            `x` and the fiber, it is zero when `x` is coincident with the fiber. 

        Note: where the constraint is satisfied, this means `x` is both coincident 
        with `fiber` and lies on the (zero) level-set of the function `scalar_field`. 
        """

        constraint_value: ndarray = jnp.array([f(params, x), on_fiber(fiber, x)])
        return constraint_value
    return optimality_condition


def bind_solver(f: callable) -> callable: 
    @custom_root(bind_optimality_condition(f))
    def bisection_solver(x_init: ndarray, params: dict, fiber: ndarray) -> ndarray: 
        """Computes the (intersection) point that is (1) on the line segment represented 
        by `fiber` and (2) at which the scalar field takes on the value zero. 

        Parameters 
        ----------
        x_init: ndarray
            unused but necessary for signature of jaxopt.custom_root. 
        params: tuple 
            auxiliary parameters for the bound callable `f`. 
        fiber: ndarray 
            line segment for which the scalar_field, when evaluated at each endpoint, 
            returns a value with opposing sign. 

        Returns 
        -------
        fixed_point: ndarray 
            intersection point which (1) lies on the line segment specified by `fiber` 
            and (2) satisfies f(params, fixed_point) == 0. 
        """

        fixed_point: ndarray = bisect(functools.partial(f, params), fiber)
        return fixed_point
    return bisection_solver


def projection(u: ndarray, v: ndarray) -> ndarray:
    """Computes the projection of vector `u` onto vector `v`."""

    norm_zero: bool = v.dot(v) == 0. 
    result: ndarray = jax.lax.cond(norm_zero, lambda u: jnp.zeros(2), lambda u: divide00((u.dot(v)), (v.dot(v))) * v, operand=u)
    return result 


def on_fiber_constraint(fiber: ndarray, x: ndarray) -> float:
    """Computes the distance between a point `x` and a fiber `fiber`."""

    fiber_line_direction: ndarray = fiber[1] - fiber[0]
    projection_vector: ndarray = projection(x, fiber_line_direction) - x
    distance_from_fiber: ndarray = projection_vector.dot(projection_vector)
    return distance_from_fiber


def bisection_constraint(f: callable, x: ndarray, params: tuple, fiber: ndarray) -> ndarray:
    """Computes the ndarray of length 2 representing the vector-value of the 
    `constraint function` which takes on the value of the zero-vector when 
    the constrain is satisfied. 

    Parameters 
    ----------
    scalar_field: callable 
        scalar field function whose level-set at zero is included as one of 
        the constraints; the field is evaluated at `x` and `params`. 
    x: ndarray 
        ndarray (of length 2, nominally), representing the point in the domain 
        of `scalar_field` to be evaluated. 
    params: tuple 
        auxiliary parameters for `scalar_field`. 
    fiber: ndarray 
        fiber included as the second constraint; the constraint is zero when 
        `x` lies on this fiber, otherwise the value of the constraint is the 
        distance between `x` and the fiber. 

    Returns 
    -------
    constraint_value: ndarray 
        first element is the value of the first constraint (i.e., the value of 
        the scalar field evaluated at x); second element is the distance between 
        `x` and the fiber, it is zero when `x` is coincident with the fiber. 
    Note: where the constraint is satisfied, this means `x` is both coincident 
    with `fiber` and lies on the (zero) level-set of the function `scalar_field`. 
    """

    field_constraint: float = jnp.squeeze(f(params, x))
    return jnp.array([field_constraint, on_fiber_constraint(fiber, x)])


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def bisection_solver(params: pytree, fiber: ndarray, f: callable) -> ndarray:
    """Computes the (intersection) point that is (1) on the line segment represented 
    by `fiber` and (2) at which the scalar field takes on the value zero. 

    Parameters 
    ----------
    f: callable 
        function which should return a real-valued scalar when applied like: 
        f(params, x). 
    fiber: ndarray 
        line segment for which the scalar_field `f`, when evaluated at each endpoint, 
        returns a value with opposing sign. 
    params: tuple 
        auxiliary parameters for `f`. 

    Returns 
    -------
    fixed_point: ndarray 
        intersection point which (1) lies on the line segment specified by `fiber` 
        and (2) satisfies f(params, fixed_point) == 0. 
    See also: src.implicit_differentiation.bisection_solver{forward, backward}
    """

    fixed_point: ndarray = bisect(functools.partial(f, params), fiber)
    return fixed_point


def bisection_solver_forward(params: tuple, fiber: ndarray, f: callable) -> ndarray:
    """Uses `src.implicit_differentiation.bisection_solver` to determine the 
    intersection between the fiber and the zero level-set of the scalar field. 

    Parameters 
    ----------
    scalar_field: callable 
        function which should return a real-valued scalar when applied like: 
        scalar_field(params, x). 
    fiber: ndarray 
        line segment for which the scalar_field, when evaluated at each endpoint, 
        returns a value with opposing sign. 
    params: tuple 
        auxiliary parameters for `scalar_field`. 

    Returns
    -------
    payload: tuple 
        first element is the ndarray containing the intersection point; second element 
        is the collection of residuals used in the backward pass to compute the vjp. 
    """

    # --- determine the intersection point 
    fixed_point: ndarray = bisection_solver(params, fiber, f)

    # --- collect the residuals to be used in the backward pass
    residuals: tuple = (params, fixed_point, fiber)

    payload: tuple = (fixed_point, residuals) 
    return payload 


def bisection_solver_backward(f: callable, residuals: tuple, incoming_gradient: ndarray) -> ndarray:
    """Computes the vector-Jacobian project associated with the bisection solver 
    procedure, using implicit differentiation. 

    Parameters 
    ----------
    scalar_field: callable 
        function which should return a real-valued scalar when applied like: 
        scalar_field(params, x). 
    fiber: ndarray 
        line segment for which the scalar_field, when evaluated at each endpoint, 
        returns a value with opposing sign. 
    residuals: tuple 
        first element is the parameters associated with the scalar_field (assumed 
        fixed) when the bisection solver was invoked; second element is the intersection 
        point. 
    incoming_gradient: ndarray 
        gradient signal arising from some downstream (from the perspective of the 
        forward pass, that is) computation for which autodiff has already produced 
        derivative values; these are used to correctly proceed with the chain rule 
        back upstream of wherever `bisection_solver` was called. 

    Returns
    -------
    final_vjp: ndarray
        array containing the local derivatives associated with the bisection solver 
        and multiplied with the incoming (dowstream) derivatives. 
    """

    # --- unpack residuals
    params, fixed_point, fiber = residuals

    # --- f's univariate analogues
    f_params: callable = lambda _params: bisection_constraint(f, fixed_point, _params, fiber)
    f_spatial: callable = lambda _x: bisection_constraint(f, _x, params, fiber)

    # --- partial vjps (w.r.t. params and the spatial variable)
    _, vjp_params = jax.vjp(f_params, params)
    _, vjp_spatial = jax.vjp(f_spatial, fixed_point)

    # --- solve for the intermediate vjp
    jacobian_f_fn: callable = jax.jacobian(f_spatial)
    jacobian_f: ndarray = jacobian_f_fn(fixed_point)

    A: ndarray = jacobian_f.T
    b: ndarray = incoming_gradient
    intermediate_vjp: ndarray = -1.0 * jnp.linalg.solve(A, b)
    final_vjp: ndarray = vjp_params(intermediate_vjp)

    return (final_vjp[0], None)


# --- solver custom vjp binding
bisection_solver.defvjp(bisection_solver_forward, bisection_solver_backward)
