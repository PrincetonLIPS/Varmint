import time
import os
import sys

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from varmintv2.geometry.elements import Patch2D
from varmintv2.geometry.geometry import SingleElementGeometry
from varmintv2.geometry import bsplines

from varmintv2.physics.constitutive import LinearElastic2D
from varmintv2.physics.materials import Material
from varmintv2.solver.optimization import SparseNewtonSolver

from scipy.spatial import KDTree

import numpy as np
import jax


def gen_cell(corners, num_ctrl, radius):
    center = np.mean(corners, axis=0)

    # Generate 4 patches manually
    ctrl = []

    # Left
    corner1 = corners[0]
    corner2 = corners[1]

    angles = np.linspace(-3*np.pi/4, -5*np.pi/4, num_ctrl)
    right_perim = np.linspace(corner1, corner2, num_ctrl)
    left_perim = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    ctrl.append(np.linspace(left_perim, right_perim, num_ctrl))

    # Top
    corner1 = corners[1]
    corner2 = corners[2]

    angles = np.linspace(-5*np.pi/4, -7*np.pi/4, num_ctrl)
    right_perim = np.linspace(corner1, corner2, num_ctrl)
    left_perim = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    ctrl.append(np.linspace(left_perim, right_perim, num_ctrl))

    # Right
    corner1 = corners[2]
    corner2 = corners[3]

    angles = np.linspace(-7*np.pi/4, -9*np.pi/4, num_ctrl)
    right_perim = np.linspace(corner1, corner2, num_ctrl)
    left_perim = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    ctrl.append(np.linspace(left_perim, right_perim, num_ctrl))

    # Bottom
    corner1 = corners[3]
    corner2 = corners[0]

    angles = np.linspace(-9*np.pi/4, -11*np.pi/4, num_ctrl)
    right_perim = np.linspace(corner1, corner2, num_ctrl)
    left_perim = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    ctrl.append(np.linspace(left_perim, right_perim, num_ctrl))

    return np.stack(ctrl, axis=0)


def solve_slab_with_hole(spline_degree, patch_ncp):
    l_x = 10.0
    l_y = 5.0
    radius = 1.0

    corners = np.array([
        [0.0, 0.0],
        [0.0, l_y],
        [l_x, l_y],
        [l_x, 0.0],
    ])

    ctrls = gen_cell(corners, patch_ncp, radius)

    class Steel(Material):
        _E = 200.0  # GPa
        _nu = 0.3
        _density = 7.85  # g / cm^3

    mat = LinearElastic2D(Steel)
    #mat = NeoHookean2D(Steel)

    flat_ctrls = ctrls.reshape((-1, 2))
    print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))
    print('\tDone.')

    group_1 = np.abs(ctrls[..., 0] - 0.0) < 1e-14
    group_2 = np.abs(ctrls[..., 0] - l_x) < 1e-10

    dirichlet_groups = {
        '1': group_1,
        '2': group_2,
    }

    traction_groups = {
        # empty
    }

    quad_degree = 10

    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = Patch2D(xknots, yknots, spline_degree, quad_degree)

    cell = SingleElementGeometry(
        element=element,
        material=mat,
        init_ctrl=ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    )

    ref_ctrl = ctrls
    potential_energy_fn = cell.get_potential_energy_fn(ref_ctrl)
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(ref_ctrl))

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    potential_energy_fn = jax.jit(potential_energy_fn)
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()
    curr_g_pos = l2g(ref_ctrl)
    ndof = curr_g_pos.size
    print(f"{ndof} global degrees of freedom.")

    d_x = 1.0

    fixed_displacements = {
        '1': np.array([0.0, 0.0]),
        '2': np.array([d_x, 0.0]),
    }

    tractions = {}

    fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
    tractions = cell.tractions_from_dict(tractions)

    print(f'Starting optimization.')
    opt_start = time.time()
    optimizer = SparseNewtonSolver(cell, potential_energy_fn, max_iter=1, step_size=1.0)
    new_x, success = optimizer.optimize(curr_g_pos, (fixed_locs, tractions))
    if not success:
        print(f'Optimization reached max iters.')
    else:
        print(f'Optimization succeeded')
    print(f'Took {time.time() - opt_start} seconds.')

    map_fn, stress_fn = cell.get_stress_field_fn()
    points = map_fn(ref_ctrl).reshape(-1, 2)
    stress = stress_fn(ref_ctrl, g2l(new_x, fixed_locs)).reshape(-1, 2, 2)

    s = stress - 1.0/3 * np.trace(stress, axis1=-2, axis2=-1)[:, None, None] * np.eye(2)
    von_Mises = np.sqrt(3./2 * np.einsum('ijk,ijk->i', s, s))

    def stress_at(point):
        ind =np.argmin(np.linalg.norm(points - point, axis=-1))
        return von_Mises[ind]

    def deformation_at(point):
        pind, qd = cell.point_to_patch_and_parent(point, ref_ctrl)
        return cell.patch_and_parent_to_point(pind, qd, g2l(new_x, fixed_locs)) - cell.patch_and_parent_to_point(pind, qd, ref_ctrl)

    def ref_at(point):
        pind, qd = cell.point_to_patch_and_parent(point, ref_ctrl)
        return cell.patch_and_parent_to_point(pind, qd, ref_ctrl)

    return (ref_ctrl, g2l(new_x, fixed_locs), element, cell, ndof), stress_at, deformation_at, ref_at