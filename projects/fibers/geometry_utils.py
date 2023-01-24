from functools import partial
import sys

import chex
import jax
from jax import jit, vmap
import jax.numpy as np
import jax.random as jnpr
import numpy as static_np
import numpy.random as npr
from scipy.optimize import least_squares
import shapely

from varmint.utils.typing import *
from jax_utils import _vectorize, _jit_vectorize, custom_norm

def sample_from_ball(num_samples: int, dimension: int=2, radius: float=1., translation: ndarray=np.zeros(2), disk: bool=False) -> ndarray: 
    if dimension == 2: 
        angles: ndarray = npr.uniform(-np.pi, np.pi, size=(num_samples,))
        if disk: 
            radius: ndarray = npr.uniform(0., radius, size=(num_samples, 1))
        samples: ndarray = radius * np.vstack((
            np.cos(angles), 
            np.sin(angles)
            )).T
        samples += translation 
    elif dimension == 3: 
        pass
    else: 
        raise NotImplementedError(f"dimension {dimension} not supported.")
    return samples

def get_contours(boundary: tuple, f: callable, resolution: int = 100, dimension: int=2) -> tuple:
    """Given a callable `density` and a domain defined by `boundary`, evaluates 
    the callable on a grid to numerically determine its contours/level-sets. 
    Parameters
    ----------
    boundary: Tuple[float, ...]
        tuple of four floats encoding the rectangular boundary to evaluate the `density` 
        on, in the format (min_x, min_y, max_x, max_y). 
    f: callable 
        function to be evaluated within the boundary (the one we want to get 
        contours for); takes in arrays of size 2 nominally. 
    resolution: int
        grid-resolution to evaluate the density. 
    dimension: int 
        dimension of the input points (and thus the contour). 
    Returns
    -------
    evaluations: Tuple[ndarray, ...]
        tuple containing the input coordinates (x, first element and y, second element) 
        and the output value (z, third element)
    """
    if dimension == 2: 
        # --- unpack the boundary values 
        min_x, min_y, max_x, max_y = boundary

        # --- construct the grid of input points
        _x = np.linspace(min_x, max_x, num=resolution)
        _y = np.linspace(min_y, max_y, num=resolution)
        x, y = np.meshgrid(_x, _y)
        inputs: ndarray = np.dstack((x, y)).reshape(-1, 2)
        z = f(inputs)
        z = z.reshape(resolution, resolution)
    elif dimension == 3: 
        pass 
    else: 
        raise NotImplementedError(f"dimension {dimension} not supported.")
    return (_x, _y, z)

def uniform_in_rectangle_jax(num_samples: int, bounds: ndarray, key: ndarray) -> ndarray: 
    """Sample `num_samples` values according to a uniform distribution 
    over the rectangle defined by `bounds`.
    Parameters 
    ----------
    num_samples: int 
        number of samples to generate. 
    bounds: Union[Tuple, ndarray]
        tuple or ndarray of either four (for 2D) or six (for 3D) floats encoding 
        the rectangular boundary, in the format (min_x, min_y, max_x, max_y). 
    Returns
    -------
    samples: ndarray 
        ndarray of samples of shape (num_samples, 2). 
    """
    x_key, y_key = jnpr.split(key, 2)
    x_low, y_low, x_high, y_high = bounds 

    # --- sample the x and y-coordinate marginals (themselves distributed uniform) 
    x: ndarray = jnpr.uniform(x_key, shape=(num_samples,), minval=x_low, maxval=x_high)
    y: ndarray = jnpr.uniform(y_key, shape=(num_samples,), minval=y_low, maxval=y_high)

    # --- collect the samples
    samples: ndarray = np.vstack((x, y)).T

    return samples


def uniform_in_rectangle(num_samples: int, bounds: tuple) -> ndarray: 
    """Sample `num_samples` values according to a uniform distribution 
    over the rectangle defined by `bounds`.
    Parameters 
    ----------
    num_samples: int 
        number of samples to generate. 
    bounds: Union[Tuple, ndarray]
        tuple or ndarray of either four (for 2D) or six (for 3D) floats encoding 
        the rectangular boundary, in the format (min_x, min_y, max_x, max_y). 
    Returns
    -------
    samples: ndarray 
        ndarray of samples of shape (num_samples, 2). 
    """
    # --- unpack the boundary values 
    if (type(bounds) == ndarray and bounds.size == 6) or (len(bounds) == 6): 
        x_low, y_low, x_high, y_high, z_low, z_high = bounds 

        # --- sample the x and y-coordinate marginals (themselves distributed uniform) 
        x: ndarray = static_np.random.uniform(x_low, x_high, size=(num_samples,))
        y: ndarray = static_np.random.uniform(y_low, y_high, size=(num_samples,))
        z: ndarray = static_np.random.uniform(z_low, z_high, size=(num_samples,))

        # --- collect the samples
        samples: ndarray = np.stack((x, y, z)).T
    elif (type(bounds) == ndarray and bounds.size == 4) or (len(bounds) == 4):
        x_low, y_low, x_high, y_high = bounds 

        # --- sample the x and y-coordinate marginals (themselves distributed uniform) 
        x: ndarray = static_np.random.uniform(x_low, x_high, size=(num_samples,))
        y: ndarray = static_np.random.uniform(y_low, y_high, size=(num_samples,))

        # --- collect the samples
        samples: ndarray = np.stack((x, y)).T
    else: 
        raise NotImplementedError(f"bounds must be an ndarray of size {4, 6} or a tuple of len {4, 6}.")

    return samples

@jit
def triangle_area(triangle: ndarray) -> float: 
    """Computes the area of a triangle; the vertices 
    of the triangle are contained in the input array. 
    """
    vertex_1, vertex_2, vertex_3 = triangle 
    area: float = 0.5 * custom_norm(np.cross((vertex_2 - vertex_1), (vertex_3 - vertex_1)))
    return area 

@jit
def circle_area(radius: float) -> float: 
    """Computes the area of a circle of given radius.
    """
    area: float = np.pi * np.power(radius, 2)
    return area 

@_jit_vectorize(signature="(2,2),(2,2)->(2)")
def fiber_fiber_intersection(x: ndarray, y: ndarray) -> ndarray:
    raise NotImplementedError("this function is currently not supported.")
    s_1x: ndarray = x[1][0] - x[0][0]
    s_1y: ndarray = x[1][1] - x[0][1]
    s_2x: ndarray = y[1][0] - y[0][0]
    s_2y: ndarray = y[1][1] - y[0][1]

    s: float = (-s_1y * (x[0][0] - y[0][0]) + s_1x * (x[0][1] - y[0][1])) / (-s_2x * s_1y + s_1x * s_2y)
    t: float = (s_2x * (x[0][1] - y[0][1]) - s_2y * (x[0][0] - y[0][0])) / (-s_2x * s_1y + s_1x * s_2y)

    args: tuple = (x, t, s_1x, s_1y)

    def solve_fn(args):
        x, t, s_1x, s_1y = args
        return np.array([x[0][0] + (t * s_1x), x[0][1] + (t * s_1y)])

    solve_fn: callable = lambda args: np.array([args[0][0][0] + (args[1] * args[2]), args[0][0][1] + (args[1] * args[3])])
    if_unsafe: callable = lambda _: np.array([np.inf, np.inf])
    return jax.lax.cond(
        np.logical_and(np.logical_and(s >= 0, s <= 1), np.logical_and(t >= 0, t <= 1)),
        solve_fn,
        if_unsafe,
        operand=args,
    )

def _get_hull_orientation(hull: ndarray) -> float:
    """Returns -1 if the provided hull is oriented clockwise and 
    +1 otherwise.
    """
    chex.assert_shape(hull, (..., None, 2))
    return np.sign(np.cross(hull[1] - hull[0], hull[2] - hull[0]))

def _reduce_clip_params(clips: ndarray, axis: int=-2) -> ndarray:
    if axis == -1 or axis == clips.ndim - 1:
        raise ValueError(f"Do not reduce_clip_params over coordinate axis.")
    clips: ndarray = np.moveaxis(clips, axis, 0)
    lo, hi = np.moveaxis(clips, -1, 0)
    clip_params: ndarray = np.stack([lo.max(0), hi.min(0)], axis=-1)
    return clip_params 

def _hull_to_walls(hull: ndarray) -> ndarray:
    """Computes the 'walls' (line-segments) associated with the given `hull` 
    containing the vertices of the hull. 
    Parameters 
    ----------
    hull: ndarray 
        ndarray of vertices comprising the hull in question. 
    Returns
    -------
    walls: ndarray 
        ndarray of 'walls' (line segments) corresponding to the hull. 
    """
    chex.assert_shape(hull, (..., None, 2))
    walls: ndarray = np.stack([hull, np.roll(hull, -1, axis=-2)], axis=-2)
    return walls

def _in01(x: ndarray, open: bool=True) -> ndarray:
    """Returns a boolean-valued ndarray representing whether elements 
    of `x` lie in the interval (0, 1] or (0, 1); the latter if `open`=True.
    """
    return np.logical_and(0 < x, x < 1) if open else np.logical_and(0 <= x, x <= 1)

@partial(np.vectorize, signature="(n,2),()->(n,2)")
def _orient_hull(hull: ndarray, target_orientation: int):
    """Given a hull and a target orientation, ensures the hull 
    is organized with that orientation; if it doesn't it reverses 
    the hull's orientation; the correctly oriented hull is returned 
    """
    chex.assert_shape(hull, (None, 2))

    # --- determine whether the hull ought to be reversed
    should_reverse = target_orientation != _get_hull_orientation(hull)

    # --- specify how the hull should be transformed on both sides of the branch 
    reverse: callable = lambda hull: hull[::-1]
    dont_reverse: callable = lambda hull: hull 

    # --- organize oriented hull 
    oriented_hull: ndarray = jax.lax.cond(should_reverse, reverse, dont_reverse, operand=hull)
    return oriented_hull

@_jit_vectorize(signature="(2,2),(2)->(2,2)")
def apply_fiber_clip(fiber: ndarray, clip_params: ndarray) -> ndarray:
    """Clips the provided fibers according to the clip parameters. 
    """
    clipped_fibers: ndarray = interpolate(clip_params, fiber)
    return clipped_fibers

def orient_hull_ccw(hull: ndarray) -> ndarray:
    """Returns the same hull as provided but guaranteed to be 
    oriented counter-clockwise.
    """
    oriented_hull: ndarray = _orient_hull(hull, +1)
    return oriented_hull 

def orient_hull_cw(hull: ndarray) -> ndarray:
    """Returns the same hull as provided but guaranteed to be 
    oriented clockwise.
    """
    oriented_hull: ndarray = _orient_hull(hull, -1)
    return oriented_hull

@_vectorize(signature="(2,2),(2,2)->(2)")
def intersect_segments(first_segment: ndarray, second_segment: ndarray) -> ndarray:
    """Computes the intersection between two line segments. 
    Note: returns np.array([np.inf, np.inf]) if there is no intersection. 
    """
    # --- setup the linear system 
    A: ndarray = np.stack([first_segment[1] - first_segment[0], second_segment[0] - second_segment[1]], axis=-1)
    b: ndarray = second_segment[0] - first_segment[0]

    # --- determine if the coefficient matrix is singular
    abs_determinant: ndarray = np.abs(np.linalg.det(A))

    # --- specify both sides of the branch 
    linear_solve: callable = partial(np.linalg.solve, A)
    is_no_solution: callable = lambda _: np.array([np.inf, np.inf])

    # --- compute the intersection 
    is_solvable: bool = abs_determinant > 0 
    intersection: ndarray = jax.lax.cond(is_solvable, linear_solve, is_no_solution, operand=b)
    return intersection

@_jit_vectorize(signature="(2,2),(2,2)->(2),(2),()")
def clip_wrt_wall(fiber: ndarray, wall: ndarray) -> ndarray:
    """Clips a fiber with respect to an oriented line segment `wall`. 
    Parameters 
    ----------
    fiber: ndarray 
        ndarray of shape (2, fiber_dim); the fiber. 
    wall: 
        ndarray of shape (2, fiber_dim) containing the endpoints of the wall. 
    Returns
    -------
    clip_parameters: ndarray 
        interpolation parameters for clipped fiber of shape 2. 
    endpoint_sides: int {1, -1}
        indicates which side of wall each fiber endpoint resides. 
    has_intersection: bool 
        indicates if the fiber and the wall had non-empty intersection.
    """
    # --- closure taking a query point and returning which side of the wall it falls on 
    which_side: callable = lambda query_point: np.sign(np.cross(wall[1] - wall[0], query_point - wall[0]))

    # --- which side of the wall each endpoint of the fiber falls on
    endpoint_sides: ndarray = np.stack([which_side(f) for f in fiber], axis=0)
    start_sides, end_sides = endpoint_sides

    # --- find the intersection (if it exists) 
    intersection: ndarray = intersect_segments(fiber, wall)
    has_intersection: ndarray = _in01(intersection, open=True).all(-1)
    alpha, _ = np.moveaxis(intersection, -1, 0)

    intersection_select: ndarray = jax.lax.select(
        start_sides > end_sides,
        np.stack([0.0, alpha]),
        np.stack([alpha, 1.0]),
    )

    # --- determine the clip parameters 
    clip_parameters: ndarray = jax.lax.select(has_intersection, intersection_select, np.array([0.0, 1.0]))
    return clip_parameters, endpoint_sides, has_intersection

def _clip_inside_convex_hull_safe(fiber: ndarray, hull: ndarray) -> ndarray:
    """Clips fibers to lie within a convex hull, checks first that the fibers 
    have not 'collapsed' (have start point == end point). 
    """
    collapsed_fiber: ndarray = (fiber[:, 0] == fiber[:, 1]).all() 
    args: tuple = (fiber, hull)
    is_collapsed: callable = lambda _: static_np.zeros(2) 
    not_collapsed: callable = lambda args: _clip_inside_convex_hull(args[0], args[1])
    clipped_fibers: ndarray = jax.lax.cond(collapsed_fiber, is_collapsed, not_collapsed, operand=args)
    return clipped_fibers 

@_jit_vectorize(signature="(2,2),(n,2)->(2)")
def _clip_inside_convex_hull(fiber: ndarray, hull: ndarray) -> ndarray:
    """Clip fiber(s) to inside of a convex hull.
    Parameters
    ----------
    fiber: 
        ndarray of shape (2, fiber_dim) containing the fiber. 
    hull: 
        ndarray of shape (num_vertices, fiber_dim) containing the vertices 
        of the hull. 
    # TODO example 
    Returns
    -------
    clip_parameters: ndarray 
        array of clipping parameters. 
    """
    # --- ensure we have a valid hull with greater than 2 vertices 
    num_vertices, _ = hull.shape
    assert num_vertices > 2, f"Hull must contain more than 2 vertices (got hull with {num_vertices} vertices)."

    # --- orient the hull and obtain its walls
    hull: ndarray = orient_hull_ccw(hull)
    walls = _hull_to_walls(hull)

    # --- clip the fibers with respect to the walls of the hull 
    clips, endpoint_sides, has_intersections = clip_wrt_wall(fiber, walls)
    clip: ndarray = _reduce_clip_params(clips)

    # --- determine if there are any intersections 
    any_intersections: ndarray = has_intersections.any(-1)

    is_fully_inside: ndarray = (endpoint_sides > 0).all((-1, -2))
    is_fully_outside: ndarray = np.logical_and(~is_fully_inside, ~any_intersections)
    
    # --- compute the clipping parameters
    clip_parameters: ndarray = jax.lax.select(is_fully_outside, np.zeros_like(clip), clip)
    return clip_parameters


def clip_inside_convex_hull(fibers: ndarray, hull: ndarray) -> ndarray:
    """Wraps src.utils.geometry_utils._clip_inside_convex_hull and 
    applies the clipping parameters to the fibers. 
    """
    clip_parameters = _clip_inside_convex_hull(fibers, hull)
    clipped_fibers: ndarray = apply_fiber_clip(fibers, clip_parameters)
    return clipped_fibers

def _plusminus(x, y, axis=0):
    return np.stack([x+y, x-y], axis=axis)

def _sq_norm(x, axis=-1):
    return np.square(x).sum(axis)

@_jit_vectorize(signature='(2,2),(2),()->(2),()')
def _clip_inside_circle(fiber: ndarray, center: ndarray, radius: ndarray) -> ndarray:
    """Clips a fiber with respect to a circle parameterized by its center and 
    radius. 
    Parameters
    ----------
    fiber: ndarray
        ndarray of shape (2, fiber_dim) containing the endpoints of the fiber. 
    center: ndarray 
        ndarray of shape (fiber_dim); the center of the circle. 
    radius: float 
        positive radius of the circle. 
    Returns
    -------
    clip_params: ndarray 
        pair of clipping parameters (shape=(2, 2)).
    branch_idx: int 
        integer branch index for debugging / testing.
    Reference: https://github.com/PrincetonLIPS/cnc_toolpaths_dev20/blob/8d9651fc4d79ed11d23cedd5abbb739f94c5ddf6/v0x6/utils/fiberlib.py#L217
    """
    zeros, ones, zero_one = np.zeros(2), np.ones(2), np.array([0., 1.])
    np_and: callable  = np.logical_and
    stack: callable = lambda *_a: np.stack(_a, axis=0)

    endpts_in_radius = _sq_norm(fiber - center) < radius**2
    chex.assert_shape(endpts_in_radius, (2,))
    f0_in, f1_in= endpts_in_radius

    alphas = intersect_segment_circle(fiber, center, radius)  # shape=[2]
    no_soln = ~np.logical_and(np.isfinite(alphas).all(), _in01(alphas).any())

    branch = np.array(
      [False,                      # else-case: two intersections
       endpts_in_radius.all(-1),   # fiber fully contained in the circle 
       np_and(f0_in, ~f1_in),      # clip one side
       np_and(~f0_in, f1_in),      # clip other side
       no_soln,                    # no intersection
       ])

    branch_idx = np.argmax(branch)
    branches = [
        lambda _a: _a,
        lambda _a: zero_one,
        lambda _a: np.array([0., _a[1]]),
        lambda _a: np.array([_a[0], 1.]), 
        lambda _a: zeros, 
    ]

    clip_params = jax.lax.switch(branch_idx, branches, operand=alphas)
    return clip_params, branch_idx

def clip_inside_circle(fibers: ndarray, center: ndarray, radius: float, apply=False) -> ndarray:
    """Wraps src.utils.geometry_utils._clip_inside_circle to clip fibers inside of a circle 
    parameterized by its center and radius.
    """
    clip_parameters, branch_idx = _clip_inside_circle(fibers, center, radius)
    if apply:
        return apply_fiber_clip(fibers, clip_parameters)
    else: 
        return clip_parameters, branch_idx

@_jit_vectorize(signature='(2,2),(2),()->(2)')
def intersect_segment_circle(segment: ndarray, center: ndarray, radius: ndarray) -> ndarray:
    """Computes the intersection parameter: how far along `segment` starting from its 
    start point segment[0] that it intersects with the circle parameterized by circle and radius.
    Parameters 
    ----------
    segment: ndarray 
        segment (e.g., fiber) for which to compute the intersection 
    center: ndarray 
        location of the center of the circle. 
    radius: float 
        positive radius of the circle. 
    Returns 
    -------
    intersection_parameters: ndarray 
        TODO document. 
    """
    # Solve quadratic equation: |f(alpha) - center|^2 = R^2.
    p0, p1 = segment - center
    delta = p1 - p0
    A = (delta**2).sum(-1)
    #A = np.maximum(A, 1e-8)  #FIXME
    B = 2 * (delta * p0).sum(-1)
    C = (p0**2).sum(-1) - radius**2
    disc = B**2 - 4*A*C

    alphas = jax.lax.cond(  # cond + np.abs to avoid imaginary numbers
        0 < disc,
        lambda _: _plusminus(-B, np.sqrt(np.abs(disc))) / (2 * A),
        lambda _: np.inf + np.ones([2]),
        operand=None)
    intersection_parameters: ndarray = alphas[::-1]  # --- reversed, so that alpha0 < alpha1
    return intersection_parameters

@_jit_vectorize(signature='(2,2),(k,2)->(k,2,2)')
def apply_fiber_multiclip(fiber, clip_params):
    return interpolate(clip_params, fiber)

@_jit_vectorize(signature="(),(2,2)->(2)")
def interpolate(alpha: float, segment: np.ndarray) -> np.ndarray:
    """Interpolates `alpha` amount along `segment`.
    """
    x0, x1 = np.moveaxis(segment, -2, 0)
    interpolated: ndarray = (1 - alpha) * x0 + alpha * x1
    return interpolated

def spherical_to_rectangular(spherical_coordinates: ndarray) -> ndarray: 
    theta, psi, r = spherical_coordinates[:, 0], spherical_coordinates[:, 1], spherical_coordinates[:, 2]
    x = r * np.sin(psi) * np.cos(theta) 
    y = r * np.sin(psi) * np.sin(theta) 
    z = r * np.cos(psi)
    rectangular_coordinates: ndarray = np.hstack((x, y, z))
    return rectangular_coordinates

def sample_from_unit_sphere(num_samples: int) -> ndarray: 
    thetas: ndarray = npr.uniform(-np.pi, np.pi, size=(num_samples,))
    psis: ndarray = npr.uniform(-np.pi, np.pi, size=(num_samples,))
    rs: ndarray = np.ones((num_samples,))
    spherical_coordinates: ndarray = np.hstack((thetas, psis, rs))
    rectangular_coordinates: ndarray = spherical_to_rectangular(spherical_coordinates)
    return rectangular_coordinates

def sample_from_boundary(bounds: ndarray, samples_per_wall: int) -> ndarray: 
    min_x, min_y, max_x, max_y = bounds
    left_edge_samples: ndarray = np.vstack((np.full((samples_per_wall,), min_x), npr.uniform(min_y, max_y, (samples_per_wall,)))).T
    right_edge_samples: ndarray = np.vstack((np.full((samples_per_wall,), max_x), npr.uniform(min_y, max_y, (samples_per_wall,)))).T
    top_edge_samples: ndarray = np.vstack((npr.uniform(min_x, max_x, (samples_per_wall,)), np.full((samples_per_wall,), max_y))).T
    bottom_edge_samples: ndarray = np.vstack((npr.uniform(min_y, max_x, (samples_per_wall,)), np.full((samples_per_wall,), min_y))).T
    samples: ndarray = np.vstack((
        left_edge_samples, 
        right_edge_samples, 
        top_edge_samples, 
        bottom_edge_samples
        ))
    return samples
