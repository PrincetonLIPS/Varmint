import jax.numpy as jnp
import numpy as np
import jax

from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel
from varmint.utils.geometry_utils import generate_constraints


def construct_beam(len_x, len_y, patch_ncp, spline_degree, quad_degree, material):
    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    x_controls = np.linspace(0, len_x, patch_ncp)
    y_controls = np.linspace(0, len_y, patch_ncp)

    all_ctrls = np.stack(np.meshgrid(x_controls, y_controls), axis=-1)
    all_ctrls = all_ctrls[None, ...]
    
    # Dirichlet labels
    group_1 = np.abs(all_ctrls[..., 0] - 0.0) < 1e-9
    group_2 = np.abs(all_ctrls[..., 0] - len_x) < 1e-9
    
    dirichlet_groups = {
        '1': group_1,
        '2': (group_2, np.array([0, 1])),
    }

    traction_groups = {
        # empty
    }

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(np.array([], dtype=np.int32), np.array([], dtype=np.int32)),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    ), all_ctrls
