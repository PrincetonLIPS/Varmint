from collections import defaultdict
from functools import partial
from typing import Callable, Tuple
from jax._src.api import vmap
import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp

from scipy.sparse import csr_matrix, csc_matrix, kron, save_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree

from varmintv2.geometry import elements
from varmintv2.geometry import bsplines
from varmintv2.geometry.geometry import SingleElementGeometry
from varmintv2.physics.constitutive import PhysicsModel
from varmintv2.utils.geometry_utils import generate_constraints


def su_corners(h1, h2, h3, l, t, t1):

    #squares corners
    all_ctrls = []

    # bottom row  
    all_ctrls.append(np.array([[0.5*l-t,0],[0.5*l-t,t],[0.5*l+t,t],[0.5*l+t,0]]))  # middle
    all_ctrls.append(np.array([[t,0],[t,t],[0.5*l-t,t],[0.5*l-t,0]]))  # left middle
    all_ctrls.append(np.array([[0,0],[0,t],[t,t],[t,0]]))  # left
    all_ctrls.append(np.array([[0.5*l+t,0],[0.5*l+t,t],[l-t,t],[l-t,0]]))  # right middle
    all_ctrls.append(np.array([[l-t,0],[l-t,t],[l,t],[l,0]]))  # right

    all_ctrls.append(np.array([[l-t,t],[l-t,h1-t1],[l,h1-t1],[l,t]]))
    all_ctrls.append(np.array([[l-t,h1-t1],[l-t,h1],[l,h1],[l,h1-t1]]))
    all_ctrls.append(np.array([[0,t],[0,h1-t1],[t,h1-t1],[t,t]]))
    all_ctrls.append(np.array([[0,h1-t1],[0,h1],[t,h1],[t,h1-t1]]))
    all_ctrls.append(np.array([[0.5*l-t,h2-t1],[0.5*l-t,h2],[0.5*l+t,h2],[0.5*l+t,h2-t1]]))
    all_ctrls.append(np.array([[0.5*l-t,h2],[0.5*l-t,h3-t],[0.5*l+t,h3-t],[0.5*l+t,h2]]))
    
    # top row
    all_ctrls.append(np.array([[0.5*l-t,h3-t],[0.5*l-t,h3],[0.5*l+t,h3],[0.5*l+t,h3-t]]))  # middle
    all_ctrls.append(np.array([[t,h3-t],[t,h3],[0.5*l-t,h3],[0.5*l-t,h3-t]]))  # left middle
    all_ctrls.append(np.array([[0,h3-t],[0,h3],[t,h3],[t,h3-t]]))  # left
    all_ctrls.append(np.array([[0.5*l+t,h3-t],[0.5*l+t,h3],[l-t,h3],[l-t,h3-t]]))  # right middle
    all_ctrls.append(np.array([[l-t,h3-t],[l-t,h3],[l,h3],[l,h3-t]]))  # right
        
    return np.stack(all_ctrls, axis=0)


def su_cosine_patches (h1, h2, l, t, t1, dx):
    
    #cosine curve
    curve_x = np.linspace(0,0.5*l-2*t,dx)
    curve_y = 0.5*(h2-h1)*(1+np.cos((np.pi*curve_x)/(0.5*l-2*t)))
    
    #right curved patch
    patch_rx = np.concatenate((np.array([curve_x + 0.5*l+t]).T,np.array([curve_x + 0.5*l+t]).T),axis=0)
    patch_ry = np.concatenate((np.array([curve_y + h1-t1]).T,np.array([curve_y +h1]).T),axis=0)
    patch_r = np.concatenate((patch_rx,patch_ry),axis=1)

    #left curved patch
    patch_lx = np.concatenate((np.array([-curve_x + 0.5*l-t]).T[::-1],np.array([-curve_x + 0.5*l-t]).T[::-1]),axis=0)
    patch_ly = np.concatenate((np.array([curve_y + h1-t1]).T[::-1],np.array([curve_y +h1]).T[::-1]),axis=0)
    patch_l = np.concatenate((patch_lx,patch_ly),axis=1)
        
    return (patch_l,patch_r)


def construct_metastable2D(patch_ncp, quad_degree, spline_degree,
                           material: PhysicsModel): # -> Tuple[SingleElementGeometry, Callable, int]:

    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    h1 = 0.5
    h2 = 0.7
    h3 = 1.0
    l = 2.0
    t = 0.1
    t1 = 0.05

    sq_patches_corners = su_corners(h1, h2, h3, l, t, t1)
    patch_cl, patch_cr = su_cosine_patches(h1, h2, l, t, t1, patch_ncp)

    ctrls = []

    for i in range(sq_patches_corners.shape[0]):
        l1 = np.linspace(sq_patches_corners[i, 1],
                        sq_patches_corners[i, 2], patch_ncp)
        l2 = np.linspace(sq_patches_corners[i, 0],
                        sq_patches_corners[i, 3], patch_ncp)
        l3 = np.linspace(l1, l2, patch_ncp)

        ctrls.append(l3)

    l3_r = np.linspace(patch_cr[patch_ncp:], patch_cr[0:patch_ncp], patch_ncp)
    ctrls.append(l3_r)

    l3_l = np.linspace(patch_cl[patch_ncp:], patch_cl[0:patch_ncp], patch_ncp)
    ctrls.append(l3_l)

    # Construct a single template for the material.
    template_ctrls = np.stack(ctrls, axis=0)
    template_ctrls = np.transpose(template_ctrls, (0, 2, 1, 3))
    template_ctrls = template_ctrls[:, :, ::-1, :]

    all_ctrls = []
    num_x = 3
    num_y = 3

    # Use the template to construct a a cellular structure with offsets.
    for i in range(num_x):
        for j in range(num_y):
            xy_offset = np.array([i*2, j])
            all_ctrls.append(template_ctrls.copy() + xy_offset)

    print('Constructing control points for Metastable2D.')
    all_ctrls = np.concatenate(all_ctrls, axis=0)
    flat_ctrls = all_ctrls.reshape((-1, 2))

    # Find all constraints with a KDTree. Should take O(n log n) time,
    # and much preferable to manually constructing constraints.
    print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-10)
    constraints = np.array(list(constraints))
    print('\tDone.')


    # Dirichlet labels
    group_1 = np.abs(all_ctrls[..., 1] - 0.0) < 1e-10
    group_2 = np.abs(all_ctrls[..., 1] - num_y) < 1e-10

    dirichlet_groups = {
        '1': group_1,
        '2': group_2
    }

    traction_groups = {
        # empty
    }

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    ), all_ctrls