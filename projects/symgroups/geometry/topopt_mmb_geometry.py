from collections import defaultdict
from functools import partial
import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp

from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel
from varmint.utils.geometry_utils import fast_generate_constraints


def su_corners_beam(l,h):

    all_ctrls = []
    all_ctrls.append(np.array([[0.0,0.0],[0.0,h],[l,h],[l,0.0]]))  # middle

    return np.stack(all_ctrls, axis=0)


def construct_mmb_beam(geo_params, numx, numy, patch_ncp, quad_degree, spline_degree,
                       material: PhysicsModel):
    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)
    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    num_x = numx
    num_y = numy

    W_b = geo_params[0]
    H_b = geo_params[1]

    def construct_ctrl():
        all_ctrls = []
        w_b = W_b/num_x
        h_b = H_b/num_y

        for j in range(num_y):
            print('j = ', j)
            sq_patches_corners = su_corners_beam(w_b,h_b)

            ctrls = []
            for k in range(sq_patches_corners.shape[0]):
                l1 = jnp.linspace(sq_patches_corners[k, 1],
                            sq_patches_corners[k, 2], patch_ncp)
                l2 = jnp.linspace(sq_patches_corners[k, 0],
                            sq_patches_corners[k, 3], patch_ncp)
                l3 = jnp.linspace(l1, l2, patch_ncp)

                ctrls.append(l3)

            # Construct a single template for the material.
            template_ctrls = jnp.stack(ctrls, axis=0)
            template_ctrls = jnp.transpose(template_ctrls, (0, 2, 1, 3))
            template_ctrls = template_ctrls[:, :, ::-1, :]

            # Use the template to construct a a cellular structure with offsets.
            for i in range(num_x):
                 xy_offset = jnp.array([i*w_b, j*h_b])
                 all_ctrls.append(jnp.array(template_ctrls, copy=True) + xy_offset)


        all_ctrls = jnp.concatenate(all_ctrls, axis=0)
        return all_ctrls

    all_ctrls = construct_ctrl()
    flat_ctrls = all_ctrls.reshape((-1, 2))
    constraints = fast_generate_constraints(flat_ctrls)

    # Dirichlet labels
    group_1 = np.abs(all_ctrls[..., 0] - 0.0) < 1e-14
    group_2 = np.abs(all_ctrls[..., 0] - W_b) < 1e-14
    #group_2 = (np.abs(all_ctrls[..., 0] - 0.0) < 1e-14) * (np.abs(all_ctrls[..., 1] - H_b) < 1e-14)
    group_3 = (np.abs(all_ctrls[..., 0] - W_b) < 1e-14) * (np.abs(all_ctrls[..., 1] - 0.0) < 1e-14)

    # Example traction group: Right side
    traction_group = np.abs(all_ctrls[..., 0] - W_b) < 1e-14

    # Keep any element that contains a node on the traction group.
    traction_group = np.any(traction_group, axis=(1, 2))

    # Elements have 4 boundaries: left, top, right, bottom in that order.
    # Here we want to set the right boundary.
    traction_group = traction_group.reshape(-1, 1) * np.array([[0, 0, 1, 0]])

    dirichlet_groups = {
        '1': (group_1, np.array([1, 0])),
        #'2': (group_2, np.array([0, 1]))
        '3': group_3,
    }

    traction_groups = {
        'A': traction_group,
    }

    print('Finished initializing geometry. Creating object.')
    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
        rigid_patches_boolean=None,
    ), construct_ctrl

