import jax.numpy as jnp
import numpy as np
import jax

from scipy.spatial import KDTree

from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel
from varmint.utils.geometry_utils import generate_constraints

import estimators as est
import geometry_utils as geo_utils

from . import mesher as mshr


def construct_beam(domain_oracle, params, len_x, len_y, fidelity, quad_degree, material, negative=True):
    # Create knot vectors with 1-D splines and 2 control points.
    spline_degree = 1
    patch_ncp = 2

    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    # Use the pixel mesher with the implicit function, and then convert output
    # to ctrl points understandable by Varmint.
    coords, cells, occupied_pixels, find_patch = mshr.find_occupied_pixels(domain_oracle, params, len_x, len_y, fidelity)
    all_ctrls = coords[cells].reshape(cells.shape[0], patch_ncp, patch_ncp, coords.shape[-1])

    # Dirichlet labels
    group_1 = np.abs(all_ctrls[..., 0] - 0.0) < 1e-9
    group_2 = np.abs(all_ctrls[..., 0] - len_x) < 1e-9
    group_3 = (np.abs(all_ctrls[..., 0] - len_x) < 1e-14) * (np.abs(all_ctrls[..., 1] - 0.0) < 1e-14)

    # Example traction group: Right side
    traction_group = np.abs(all_ctrls[..., 0] - len_x) < 1e-14

    # Keep any element that contains a node on the traction group.
    traction_group = np.any(traction_group, axis=(1, 2))

    # Elements have 4 boundaries: left, top, right, bottom in that order.
    # Here we want to set the right boundary.
    traction_group = traction_group.reshape(-1, 1) * np.array([[0, 0, 1, 0]])

    dirichlet_groups = {
        #'1': (group_1, np.array([1, 0])),
        '1': group_1,
        #'2': (group_2, np.array([0, 1])),
        #'3': (group_3, np.array([0, 1])),
    }

    traction_groups = {
        #'A': traction_group,
    }

    # Find all constraints with a KDTree. Should take O(n log n) time.
    flat_ctrls = all_ctrls.reshape((-1, 2))
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))

    @jax.jit
    def gen_stratified_fibers(key):
        """Return 2 fibers per cell. Shape is (n_cells, 2, fiber_dim, fiber_dim)."""

        key, subkey = jax.random.split(key)
        a = jax.random.uniform(subkey, shape=(cells.shape[0], 2))

        h_fibers = jnp.stack([
                        jnp.stack([jnp.zeros_like(a[:, 0]), a[:, 0]], axis=-1),
                        jnp.stack([jnp.ones_like(a[:, 0]), a[:, 0]], axis=-1)
                   ], axis=-2)
        v_fibers = jnp.stack([
                        jnp.stack([a[:, 1], jnp.zeros_like(a[:, 1])], axis=-1),
                        jnp.stack([a[:, 1], jnp.ones_like(a[:, 1])], axis=-1)
                   ], axis=-2)

        cell_xmax = jnp.max(coords[cells][:, :, 0], axis=-1) # get max x coordinate in each cell
        cell_xmin = jnp.min(coords[cells][:, :, 0], axis=-1) # get min x coordinate in each cell

        cell_ymax = jnp.max(coords[cells][:, :, 1], axis=-1) # get max y coordinate in each cell
        cell_ymin = jnp.min(coords[cells][:, :, 1], axis=-1) # get min y coordinate in each cell

        scaled_h_fibers = jnp.stack([
                h_fibers[..., 0] * (cell_xmax[:, None] - cell_xmin[:, None]) + cell_xmin[:, None],
                h_fibers[..., 1] * (cell_ymax[:, None] - cell_ymin[:, None]) + cell_ymin[:, None]
        ], axis=-1)

        scaled_v_fibers = jnp.stack([
                v_fibers[..., 0] * (cell_xmax[:, None] - cell_xmin[:, None]) + cell_xmin[:, None],
                v_fibers[..., 1] * (cell_ymax[:, None] - cell_ymin[:, None]) + cell_ymin[:, None]
        ], axis=-1)

        return key, jnp.stack([scaled_h_fibers, scaled_v_fibers], axis=1)

    @jax.jit
    def gen_per_cell_fibers(key, n_fibers=20):
        n_cells = cells.shape[0]
        all_keys = jax.random.split(key, n_cells)

        cell_xmax = jnp.max(coords[cells][:, :, 0], axis=-1) # get max x coordinate in each cell
        cell_xmin = jnp.min(coords[cells][:, :, 0], axis=-1) # get min x coordinate in each cell

        cell_ymax = jnp.max(coords[cells][:, :, 1], axis=-1) # get max y coordinate in each cell
        cell_ymin = jnp.min(coords[cells][:, :, 1], axis=-1) # get min y coordinate in each cell

        len_fiber = jnp.maximum(cell_xmax[0] - cell_xmin[0], cell_ymax[0] - cell_ymin[0])
        all_fibers = jax.vmap(est.sample_fibers, in_axes=(0, 0, None, None))(all_keys, (cell_xmin, cell_ymin, cell_xmax, cell_ymax), n_fibers, len_fiber)[0]

        # Rearrange cell to make convex hull.
        hull_cells = np.stack((cells[:, 0], cells[:, 1], cells[:, 3], cells[:, 2]), axis=-1)
        all_fibers = jax.vmap(geo_utils.clip_inside_convex_hull, in_axes=(0, 0))(all_fibers, coords[hull_cells])

        return all_fibers

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    ), all_ctrls, occupied_pixels, find_patch, gen_per_cell_fibers, coords
