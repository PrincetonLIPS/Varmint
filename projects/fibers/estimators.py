"""This module contains the core procedures for actually 'sampling' fibers from 
a domain, as well as differentiable Monte Carlo estimators for use with simple 
geometric primitives like regular polygons/convex sets and implicit functions. 
"""
import functools
import os
from typing import Tuple

from jax import vmap, jit
import jax.numpy as np
import jax.random as npr

from numpy.polynomial.legendre import leggauss

from projects.fibers.implicit_differentiation import bind_solver, bisection_solver, bisect

from varmint.utils.typing import *
from varmint.utils.ad_utils import zero_one_sign, custom_norm


@functools.partial(jit, static_argnums=tuple(range(2, 6)))
def sample_fibers(key: ndarray, bounds: ndarray, num_fibers: int, length: float, fiber_dim: int = 2, numeric_type: type = FP64) -> ndarray:
    """Samples fibers according to a jointly uniform distribution over the starts of 
    the fibers and their angle; the endpoints are then determined by the fibers length.
    Parameters 
    ----------
    key: ndarray
        psuedo-random number generation key/seed array (via jax.random.PRNGKey). 
    num_fibers: int 
        strictly positive number of fibers to sample. 
    length: float 
        strictly positive fiber length. 
    bounds: ndarray 
        sampling domain specified by a 4-array with elements corresponding to 
        (min_x, min_y, max_y, max_y). 
    fiber_dim: int 
        dimensionality of the fibers (default: 2). 
    numeric_type: type 
        numeric type to use for the fibers (default: FP64). 
    Returns 
    -------
    fibers: ndarray 
        an array of shape (num_fibers, fiber_dim, fiber_dim) containing the fibers 
        along axis 0. 
    """
    # --- instantiate subseeds 
    next_key, location_key, angular_key = npr.split(key, 3)
    x_key, y_key = npr.split(location_key, 2)

    starts: ndarray = np.array(
            (
            npr.uniform(x_key, shape=(num_fibers,), dtype=numeric_type, minval=bounds[0], maxval=bounds[2]), 
            npr.uniform(y_key, shape=(num_fibers,), dtype=numeric_type, minval=bounds[1], maxval=bounds[3])
            )
        ).T

    # --- given fiber starting coordinates, sample an angle 
    angles: ndarry = npr.uniform(angular_key, shape=(num_fibers,), dtype=numeric_type, minval=-np.pi, maxval=np.pi)

    # --- determine ends given fibers' starting coordinate and angle 
    ends: ndarray = starts + (length * np.array([np.cos(angles), np.sin(angles)]).T)

    # --- organize fibers in a (num_fibers, fiber_dim, fiber_dim) ndarray 
    fibers: ndarray = np.stack((starts, ends), axis=-fiber_dim)
    return fibers, next_key


def sample_points(key, bounds, num_points):
    next_key, location_key = npr.split(key)

    x_key, y_key = npr.split(location_key)
    points = np.array((
            npr.uniform(x_key, shape=(num_points,), minval=bounds[0], maxval=bounds[2]),
            npr.uniform(y_key, shape=(num_points,), minval=bounds[1], maxval=bounds[3]),
    )).T

    return points, next_key


def estimate_field_value_mc(domain_oracle, integrand, points: ndarray, params: pytree, negative=True):
    domain_oracle_params, integrand_params = params
    f: callable = jax.vmap(lambda x: integrand(integrand_params, x))
    domain: callable = jax.vmap(lambda x: domain_oracle(domain_oracle_params, x))

    valid_points = domain(points) < 0 if negative else domain(points) > 0
    area_estimate = points[valid_points].shape[0] / points.shape[0]
    field_values = f(points[valid_points])

    return np.mean(field_values, axis=0) * area_estimate

def estimate_field_length(field: callable, fibers: ndarray, params: tuple, negative: bool = True) -> float:
    """Estimates the total fiber length for which a given scalar `field` takes on positive/negative
    value. 
    Parameters
    ----------
    field: callable[[...], float]
        scalar, real-valued callable which takes auxiliary data `params` and fiber endpoints as 
        input argument(s); if fibers are dimension 2 for example, field takes ndarrays of size 2 
        and `params` to produce a real-valued output. 
    fibers: ndarray 
        ndarray of shape (num_fibers, fiber_dim, fiber_dim) of fibers. 
    params: pytree 
        auxiliary data provided to the field (e.g., parameters). 
    negative: bool 
        estimate the total fiber length for which `field` takes on negative values, if True; 
        if instead False, estimate the total fiber length for which `field` takes on positive 
        values. 
    Returns 
    -------
    total_length: float 
        nonnegative Monte Carlo estimate of the total fiber length for which `field` takes on negative/positive 
        values (negative by default). 
        
    Note: this estimator assumes that `field` changes sign on a lengthscale larger than the length of each
    fiber.
    """
    # --- vectorize and partially evaluate `field`
    vmap_field: callable = functools.partial(vmap(field, in_axes=(None, 0)), params)
    solver_base: callable = bind_solver(field)
    solver: callable = jit(vmap(lambda fiber, params: solver_base(np.empty(0), params, fiber), in_axes=(0, None)))

    # --- compute the sign of field(x) where x is each fiber endpoint
    start_points, end_points = fibers[:, 0], fibers[:, 1]
    start_values, end_values = (vmap_field(start_points).ravel(), vmap_field(end_points).ravel())
    start_signs, end_signs = zero_one_sign(start_values), zero_one_sign(end_values)

    # --- (default) 0: field(x) < 0 --- 1: field(x) >= 0
    if negative:
        negative: float = 0.0
        positive: float = 1.0
    else:
        negative: float = 1.0
        positive: float = 0.0

    # --- associated boolean condition ndarrays
    count_entire_fiber_cond: ndarray = np.logical_and(start_signs == negative, end_signs == negative)
    count_none_fiber_cond: ndarray = np.logical_and(start_signs == positive, end_signs == positive)
    count_from_start_cond: ndarray = np.logical_and(start_signs == negative, end_signs == positive)
    count_from_end_cond: ndarray = np.logical_and(start_signs == positive, end_signs == negative)

    # --- initialize the total field length
    total_length: float = 0.0

    # --- case: count the entire fiber length
    count_all_fibers: ndarray = np.where(count_entire_fiber_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))
    total_length += np.linalg.norm(count_all_fibers[:, 1] - count_all_fibers[:, 0], axis=1).sum()

    # --- case: count from the start of the fiber to the intersection point
    count_from_start: ndarray = np.where(count_from_start_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))
    total_length += np.linalg.norm(solver(count_from_start, params) - count_from_start[:, 0], axis=1).sum()

    # --- case: count from the end of the fiber to the intersection point
    count_from_end: ndarray = np.where(count_from_end_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))
    total_length += np.linalg.norm(count_from_end[:, 1] - solver(count_from_end, params), axis=1).sum()

    return total_length

def estimate_field_area(field: callable, fibers: ndarray, params: pytree, negative: bool = True) -> float:
    """Estimates the total area for which a scalar `field` takes on positive/negative
    value (negative, by default).
    Parameters
    ----------
    field: callable[[...], float]
        scalar, real-valued callable which takes auxiliary data `params` and fiber endpoints as 
        input argument(s); if fibers are dimension 2 for example, field takes ndarrays of size 2 
        and `params` to produce a real-valued output. 
    fibers: ndarray 
        ndarray of shape (num_fibers, fiber_dim, fiber_dim) of fibers. 
    params: tuple 
        auxiliary data provided to the field (e.g., parameters). 
    negative: bool 
        estimate the total area for which `field` takes on negative values, if True; 
        if instead False, estimate the area for which `field` takes on positive 
        values. 
    Returns 
    -------
    total_area: float 
        nonnegative Monte Carlo estimate of the total fiber area for which `field` takes on negative/positive 
        values (negative by default). 
        
    Note: this estimator assumes that `field` changes sign on a lengthscale larger than the length of each
    fiber.
    """
    cumulative_fiber_length: float = np.linalg.norm(fibers[:, 1] - fibers[:, 0], axis=1).sum()
    total_length: float = estimate_field_length(field, fibers, params, negative=negative)
    total_field_area: float = total_length / cumulative_fiber_length
    return total_field_area

@functools.partial(jit, static_argnums=(0, 3))
def clip_to_field(field: callable, fibers: ndarray, params: pytree, negative: bool=True) -> ndarray: 
    # --- vectorize and partially evaluate `field`
    vmap_field: callable = functools.partial(vmap(field, in_axes=(None, 0)), params)
    solver: callable = vmap(lambda fiber: bisection_solver(params, fiber, field))

    # --- compute the sign of field(x) where x is each fiber endpoint
    start_points, end_points = fibers[:, 0], fibers[:, 1]
    start_values, end_values = (vmap_field(start_points).ravel(), vmap_field(end_points).ravel())
    start_signs, end_signs = zero_one_sign(start_values), zero_one_sign(end_values)

    # --- (default) 0: field(x) < 0 --- 1: field(x) >= 0
    if negative:
        negative: float = 0.0
        positive: float = 1.0
    else:
        negative: float = 1.0
        positive: float = 0.0

    # --- associated boolean condition ndarrays
    count_entire_fiber_cond: ndarray = np.logical_and(start_signs == negative, end_signs == negative)
    count_none_fiber_cond: ndarray = np.logical_and(start_signs == positive, end_signs == positive)
    count_from_start_cond: ndarray = np.logical_and(start_signs == negative, end_signs == positive)
    count_from_end_cond: ndarray = np.logical_and(start_signs == positive, end_signs == negative)

    # --- case: count the entire fiber length
    inside_fibers: ndarray = np.where(count_entire_fiber_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))

    # --- case: clip from the start of the fiber to the intersection point
    count_from_start: ndarray = np.where(count_from_start_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))
    solver_cond: callable = lambda predicates, fibers: vmap(lambda predicate, fiber: jax.lax.cond(predicate, lambda fiber: solver(fiber[None, :, :]), lambda fiber: np.zeros((1, 2)), operand=fiber))(predicates, fibers)
    start_clipped_fibers: ndarray = np.dstack((count_from_start[:, 0], np.squeeze(solver_cond(count_from_start_cond, count_from_start)))).swapaxes(2, 1).reshape(-1, 2, 2)

    # --- case: count from the end of the fiber to the intersection point
    count_from_end: ndarray = np.where(count_from_end_cond.reshape(-1, 1, 1), fibers, np.zeros_like(fibers))
    end_clipped_fibers: ndarray = np.dstack((np.squeeze(solver_cond(count_from_end_cond, count_from_end)), count_from_end[:, 1])).swapaxes(2, 1).reshape(-1, 2, 2)

    # --- aggregate all the valid fibers 
    clipped_fibers: ndarray = np.vstack((inside_fibers, start_clipped_fibers, end_clipped_fibers))
    return clipped_fibers


def _compute_line_integral(f: callable, fiber: ndarray, degree: int=10) -> float: 
    start_point, end_point = fiber 
    # input to f is still 2D bc start, end pts are 2D
    f_parametric: callable = lambda t: f(t * start_point + (1 - t) * end_point)

    # eval_pts is (degree,) (so are weights)--these should be 2D?
    evaluation_points, weights = leggauss(degree)

    # leggauss gives points for [-1, 1]. Rescale for [0, 1].
    evaluation_points = (evaluation_points + 1.0) / 2.0
    weights = weights / 2.0

    evals = vmap(f_parametric)(evaluation_points)
    line_integral: float = evals.T @ weights

    # Rescale to coordinates in physical space.
    line_integral = line_integral * custom_norm(end_point - start_point)

    # If the start point and end point are the same (fiber has been collapsed),
    # then do not evaluate line integral.
    return jax.lax.cond(
        custom_norm(start_point - end_point) < 1e-8,
        lambda: 0.0,
        lambda: line_integral,
    )


def estimate_field_value(domain_oracle: callable, integrand: callable, fibers: ndarray, params: pytree, negative: bool=True): 
    """Estimate the integral of integrand over the field, conditioned on where domain_oracle is negative."""

    domain_oracle_params, integrand_params = params 
    valid_fibers: ndarray = clip_to_field(domain_oracle, fibers, domain_oracle_params, negative)

    f: callable = lambda x: integrand(integrand_params, x) 
    field_value: float = vmap(_compute_line_integral, in_axes=(None, 0))(f, valid_fibers).sum()

    # We should divide by the cumulative fiber length to estimate the integral properly.
    cumulative_fiber_length: float = np.linalg.norm(fibers[:, 1] - fibers[:, 0], axis=1).sum()
    return field_value / cumulative_fiber_length


def compute_shape_derivative(objective_fn: callable, domain_oracle: callable, integrand: callable, fibers: ndarray, params: pytree, negative: bool=True): 
    """Compute the shape derivative with respect to an objective function at random points along the shape boundary.
    
    The `domain_oracle` defines the domain through the implicit function. The `integrand` defines the integral
    over the domain that we want to use in the objective function. The objective is of the form J(I) where I is

    $$I = \int_{\Omega} integrand(int_params, x) dx$$

    and \Omega is the domain defined by the domain oracle. Hence, `objective_fn`: R -> R.
    This function will compute intersections of each fiber with the domain, evaluate the shape derivative at those
    locations, and then use that information to compute new values of the domain oracle at the intersection points.
    The return value will be size (num_fibers,). It is then the job of the user to optimize the parameters of the
    domain to match these points.
    """

    domain_oracle_params, integrand_params = params 
    I = estimate_field_value(domain_oracle, integrand, fibers, params)
    upstream_grad = jax.grad(objective_fn)(I).squeeze()  # Should just be a scalar.

    def domain_with_params(point):
        return domain_oracle(domain_oracle_params, point)

    # Sample a bunch of points on the current surface.
    surface_points = jax.vmap(bisect, in_axes=(None, 0))(domain_with_params, fibers)
    is_surface = jnp.abs(jax.vmap(domain_with_params)(surface_points)) < 1e-8

    # Compute the energy density at each point.
    energy_densities = jax.vmap(integrand, in_axes=(None, 0))(integrand_params, surface_points)

    # Compute the gradient norm of the implicit surface at each point.
    implicit_grad_norm = jax.vmap(lambda p: jnp.linalg.norm(jax.grad(domain_with_params)(p)))

    # Compute the update to the domain at each surface point.
    if negative:
        domain_updates = energy_densities * implicit_grad_norm(surface_points)
    else:
        domain_updates = -energy_densities * implicit_grad_norm(surface_points)

    return upstream_grad * domain_updates, surface_points, is_surface