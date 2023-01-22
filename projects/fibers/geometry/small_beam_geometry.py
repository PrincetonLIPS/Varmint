import jax.numpy as jnp
import numpy as np
import jax

from scipy.spatial import KDTree

from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel
from varmint.utils.geometry_utils import generate_constraints

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
        '1': (group_1, np.array([1, 0])),
        #'2': (group_2, np.array([0, 1])),
        '3': (group_3, np.array([0, 1])),
    }

    traction_groups = {
        'A': traction_group,
    }

    # Find all constraints with a KDTree. Should take O(n log n) time.
    flat_ctrls = all_ctrls.reshape((-1, 2))
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    ), all_ctrls, occupied_pixels, find_patch
