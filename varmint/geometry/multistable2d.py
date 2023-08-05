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

from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel
from varmint.utils.geometry_utils import generate_constraints


def su_corners(h1, h2, h3, l, t, t1):

    #squares corners
    all_ctrls = []

    # bottom row  
    all_ctrls.append(jnp.array([[0.5*l-t,0],[0.5*l-t,t],[0.5*l+t,t],[0.5*l+t,0]]))  # middle
    all_ctrls.append(jnp.array([[t,0],[t,t],[0.5*l-t,t],[0.5*l-t,0]]))  # left middle
    all_ctrls.append(jnp.array([[0,0],[0,t],[t,t],[t,0]]))  # left
    all_ctrls.append(jnp.array([[0.5*l+t,0],[0.5*l+t,t],[l-t,t],[l-t,0]]))  # right middle
    all_ctrls.append(jnp.array([[l-t,0],[l-t,t],[l,t],[l,0]]))  # right

    all_ctrls.append(jnp.array([[l-t,t],[l-t,h1-t1],[l,h1-t1],[l,t]]))
    all_ctrls.append(jnp.array([[l-t,h1-t1],[l-t,h1],[l,h1],[l,h1-t1]]))
    all_ctrls.append(jnp.array([[0,t],[0,h1-t1],[t,h1-t1],[t,t]]))
    all_ctrls.append(jnp.array([[0,h1-t1],[0,h1],[t,h1],[t,h1-t1]]))
    all_ctrls.append(jnp.array([[0.5*l-t,h2-t1],[0.5*l-t,h2],[0.5*l+t,h2],[0.5*l+t,h2-t1]]))
    all_ctrls.append(jnp.array([[0.5*l-t,h2],[0.5*l-t,h3-t],[0.5*l+t,h3-t],[0.5*l+t,h2]]))
    
    # top row
    all_ctrls.append(jnp.array([[0.5*l-t,h3-t],[0.5*l-t,h3],[0.5*l+t,h3],[0.5*l+t,h3-t]]))  # middle
    #all_ctrls.append(np.array([[t,h3-t],[t,h3],[0.5*l-t,h3],[0.5*l-t,h3-t]]))  # left middle
    #all_ctrls.append(np.array([[0,h3-t],[0,h3],[t,h3],[t,h3-t]]))  # left
    #all_ctrls.append(np.array([[0.5*l+t,h3-t],[0.5*l+t,h3],[l-t,h3],[l-t,h3-t]]))  # right middle
    #all_ctrls.append(np.array([[l-t,h3-t],[l-t,h3],[l,h3],[l,h3-t]]))  # right
        
    return jnp.stack(all_ctrls, axis=0)


def su_cosine_patches (h1, h2, l, t, t1, dx):
    
    #cosine curve
    curve_x = jnp.linspace(0,0.5*l-2*t,dx)
    curve_y = 0.5*(h2-h1)*(1+jnp.cos((jnp.pi*curve_x)/(0.5*l-2*t)))
    
    #right curved patch
    patch_rx = jnp.concatenate((jnp.array([curve_x + 0.5*l+t]).T,jnp.array([curve_x + 0.5*l+t]).T),axis=0)
    patch_ry = jnp.concatenate((jnp.array([curve_y + h1-t1 - curve_y[-1]]).T,jnp.array([curve_y + h1 - curve_y[-1]]).T),axis=0)
    patch_r = jnp.concatenate((patch_rx,patch_ry),axis=1)

    #left curved patch
    patch_lx = jnp.concatenate((jnp.array([-curve_x + 0.5*l-t]).T[::-1],jnp.array([-curve_x + 0.5*l-t]).T[::-1]),axis=0)
    patch_ly = jnp.concatenate((jnp.array([curve_y + h1-t1 - curve_y[-1]]).T[::-1],jnp.array([curve_y +h1 - curve_y[-1]]).T[::-1]),axis=0)
    patch_l = jnp.concatenate((patch_lx,patch_ly),axis=1)
        
    return (patch_l,patch_r)


def construct_multistable2D(geo_params, numx, numy, patch_ncp, quad_degree, spline_degree,
                           material: PhysicsModel, multiplier=1.0): # -> Tuple[SingleElementGeometry, Callable, int]:

    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)
    
    num_x = numx
    num_y = numy
    
    def construct_ctrl(geo_params):
    
        l = geo_params[0] 
        t = geo_params[1]
        h1 = [geo_params[2:2+num_y]][0]
        h2 = [geo_params[2+num_y:2+2*num_y]][0]
        h3 = [geo_params[2+2*num_y:2+3*num_y]][0]
        t1 = [geo_params[2+3*num_y:2+4*num_y]][0]
        
               
        all_ctrls = []

        for j in range(num_y):

            sq_patches_corners = su_corners(h1[j], h2[j], h3[j], l, t, t1[j])
            patch_cl, patch_cr = su_cosine_patches(h1[j], h2[j], l, t, t1[j], 3*patch_ncp-2)

            ctrls = []

            for k in range(sq_patches_corners.shape[0]):
                l1 = jnp.linspace(sq_patches_corners[k, 1],
                            sq_patches_corners[k, 2], patch_ncp)
                l2 = jnp.linspace(sq_patches_corners[k, 0],
                            sq_patches_corners[k, 3], patch_ncp)
                l3 = jnp.linspace(l1, l2, patch_ncp)

                ctrls.append(l3)

            l3_r1 = jnp.linspace(patch_cr[3*patch_ncp-2:4*patch_ncp-2], patch_cr[0:patch_ncp], patch_ncp)
            ctrls.append(l3_r1)
            l3_r2 = jnp.linspace(patch_cr[4*patch_ncp-3:5*patch_ncp-3], patch_cr[patch_ncp-1:2*patch_ncp-1], patch_ncp)
            ctrls.append(l3_r2)
            l3_r3 = jnp.linspace(patch_cr[5*patch_ncp-4:], patch_cr[2*patch_ncp-2:3*patch_ncp-2], patch_ncp)
            ctrls.append(l3_r3)
            
            l3_l1 = jnp.linspace(patch_cl[3*patch_ncp-2:4*patch_ncp-2], patch_cl[0:patch_ncp], patch_ncp)
            ctrls.append(l3_l1)
            l3_l2 = jnp.linspace(patch_cl[4*patch_ncp-3:5*patch_ncp-3], patch_cl[patch_ncp-1:2*patch_ncp-1], patch_ncp)
            ctrls.append(l3_l2)
            l3_l3 = jnp.linspace(patch_cl[5*patch_ncp-4:], patch_cl[2*patch_ncp-2:3*patch_ncp-2], patch_ncp)
            ctrls.append(l3_l3)


            # Construct a single template for the material.
            template_ctrls = jnp.stack(ctrls, axis=0)
            template_ctrls = jnp.transpose(template_ctrls, (0, 2, 1, 3))
            template_ctrls = template_ctrls[:, :, ::-1, :]

            # Use the template to construct a a cellular structure with offsets.
            for i in range(num_x):

                 xy_offset = jnp.array([i*l, np.sum(h3[0:j])])
                 #all_ctrls.append(template_ctrls.copy() + xy_offset)
                 all_ctrls.append(jnp.array(template_ctrls, copy=True) + xy_offset)
                 

        #print('Constructing control points for Metastable2D.')
        all_ctrls = jnp.concatenate(all_ctrls, axis=0)
        
        return all_ctrls
    
    
    all_ctrls = construct_ctrl(geo_params)
    flat_ctrls = all_ctrls.reshape((-1, 2))

    # Find all constraints with a KDTree. Should take O(n log n) time,
    # and much preferable to manually constructing constraints.
    print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))
    print('\tDone.')

    l = geo_params[0]
    h3 = [geo_params[2+2*num_y:2+3*num_y]][0]
    
    
    # Dirichlet labels
    group_1 = np.abs(all_ctrls[..., 1] - 0.0) < 1e-14
    group_2 = np.abs(all_ctrls[..., 1] - np.sum(h3)) < 1e-14
    group_3 = np.abs(all_ctrls[..., 0] - 0.0) < 1e-14
    group_4 = np.abs(all_ctrls[..., 0] - num_x * l) < 1e-14

    dirichlet_groups = {
        '1': group_1,
        '2': group_2,
        '3': (group_3, np.array([1, 0])),
        '4': (group_4, np.array([1, 0])),
    }

    traction_groups = {
        # empty
    }

    #rigid_patches = (np.ones(12), np.zeros(6)) * num_x * num_y
    rigid_patches = (np.ones(5),np.ones(1),np.zeros(1),np.ones(1),np.zeros(3),np.ones(1),np.zeros(6)) * num_x * num_y
    rigid_patches = np.concatenate(rigid_patches).astype(np.bool)

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
        rigid_patches_boolean=rigid_patches,
    ), construct_ctrl
