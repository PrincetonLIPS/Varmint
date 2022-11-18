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

import sys
import os
sys.path.append(os.path.dirname('/n/fs/mm-iga/Varmint/varmint'))
from varmint.geometry import elements
from varmint.geometry import bsplines
from varmint.geometry.geometry import SingleElementGeometry
from varmint.physics.constitutive import PhysicsModel

from mpi4py import MPI


class ShapeUnit2D(object):
    def get_ctrl_offset(self):
        return self.ctrl_offset

    def compute_internal_constraints(self):
        """Get internal constraints.

        Returns (row_inds, col_inds), where each element is an array of
        indices, shifted by ctrl_offset.
        """
        raise NotImplementedError()

    def get_side_indices(self, side):
        """Side can be left, top, right, bottom.

        Returns a 1-D array of indices, shifted by ctrl_offset.
        Order should be left->right or bottom->up.
        """
        raise NotImplementedError()

    def get_side_index_array(self, side, global_n_patches):
        """Returns a binary array that selects the indices of a side."""
        raise NotImplementedError()

    def get_side_orientation(self, side, global_n_patches):
        raise NotImplementedError()


class UnitCell2D(ShapeUnit2D):
    @staticmethod
    def _gen_random_radii(ncp, seed=None):
        npr.seed(seed)
        init_radii = npr.rand((ncp - 1) * 4) * 0.9 + 0.05
        return np.array(init_radii)

    @staticmethod
    def _gen_cell_edge(center, radial_ncp, corner1, corner2, radii, normalized=False):
        num_ctrl = len(radii)

        right_perim = jnp.linspace(corner1, corner2, num_ctrl)
        if not normalized:
            left_perim = radii[:, jnp.newaxis] * (right_perim - center) + center
        else:
            left_perim = radii[:, jnp.newaxis] * \
                (right_perim - center) / jnp.linalg.norm(right_perim - center, axis=-1)[:, jnp.newaxis] + center

        ctrl = jnp.linspace(left_perim, right_perim, radial_ncp)

        return ctrl

    @staticmethod
    def _gen_cell(corners, radial_ncp, radii, normalized=False):
        sides = corners.shape[0]
        num_ctrl = (len(radii) // sides) + 1
        centroid = jnp.mean(corners, axis=0)

        # Computes: left, top, right, bottom
        ctrl = []
        for ii in range(sides):
            corner1 = corners[ii, :]
            corner2 = corners[(ii+1) % sides]
            start = (num_ctrl - 1) * ii
            end = start + num_ctrl
            indices = jnp.arange(start, end)

            new_ctrl = UnitCell2D._gen_cell_edge(
                centroid,
                radial_ncp,
                corner1,
                corner2,
                jnp.take(radii, indices, mode='wrap'),
                normalized,
            )

            ctrl.append(new_ctrl)

        return jnp.stack(ctrl, axis=0)

    def __init__(self, corners, ncp, radial_ncp, patch_offset, side_labels=None, radii=None):
        if radii is None:
            radii = UnitCell2D._gen_random_radii(ncp)

        if side_labels is not None:
            # side_labels can handle boundary conditions by creating groups
            # upon which you can impose boundary conditions.
            # Generally sides that are shared should not have a label.
            self.side_labels = side_labels

        self.ncp = ncp
        self.radial_ncp = radial_ncp
        self.corners = corners
        self.ctrl = UnitCell2D._gen_cell(corners, radial_ncp, radii)
        n_all_ctrl = self.ctrl.size // self.ctrl.shape[-1]
        self.indices = np.arange(n_all_ctrl).reshape(self.ctrl.shape[:-1])
        self.ctrl_offset = patch_offset * ncp * radial_ncp
        self.patch_offset = patch_offset

        self.n_patches = 4

    def compute_internal_constraints(self):
        row_inds = []
        col_inds = []

        for i in range(4):
            ind1 = self.indices[(i+1) % 4, :, 0]
            ind2 = self.indices[i, :, -1]

            row_inds.extend([ind1, ind2])
            col_inds.extend([ind2, ind1])

        return (np.concatenate(row_inds) + self.ctrl_offset,
                np.concatenate(col_inds) + self.ctrl_offset)

    def get_side_indices(self, side):
        # bottom and right side indices must be flipped!
        # TODO: or is it the other way around?
        if side == 'top':
            return self.indices[1, -1, :] + self.ctrl_offset
        elif side == 'bottom':
            return np.flip(self.indices[3, -1, :]) + self.ctrl_offset
        elif side == 'left':
            return self.indices[0, -1, :] + self.ctrl_offset
        elif side == 'right':
            return np.flip(self.indices[2, -1, :]) + self.ctrl_offset
        else:
            raise ValueError(f'Invalid side {side}')

    def get_side_index_array(self, side, global_n_patches):
        ind_array = np.zeros((global_n_patches, self.radial_ncp, self.ncp))

        if side == 'top':
            ind_array[self.patch_offset + 1, -1, :] = 1
        elif side == 'bottom':
            ind_array[self.patch_offset + 3, -1, :] = 1
        elif side == 'left':
            ind_array[self.patch_offset + 0, -1, :] = 1
        elif side == 'right':
            ind_array[self.patch_offset + 2, -1, :] = 1
        else:
            raise ValueError(f'Invalid side {side}')

        return ind_array

    def get_side_orientation(self, side, global_n_patches):
        ind_array = np.zeros((global_n_patches, 4))

        # Always on the "right" side of patch
        if side == 'top':
            ind_array[self.patch_offset + 1, 2] = 1
        elif side == 'bottom':
            ind_array[self.patch_offset + 3, 2] = 1
        elif side == 'left':
            ind_array[self.patch_offset + 0, 2] = 1
        elif side == 'right':
            ind_array[self.patch_offset + 2, 2] = 1
        else:
            raise ValueError(f'Invalid side {side}')

        return ind_array


class UnitSquare2D(ShapeUnit2D):
    def __init__(self, corners, ncp, patch_offset, side_labels=None):
        if side_labels is not None:
            self.side_labels = side_labels

        l1 = np.linspace(corners[0], corners[1], ncp)
        l2 = np.linspace(corners[3], corners[2], ncp)

        self.ncp = ncp
        self.corners = corners
        self.ctrl = np.linspace(l1, l2, ncp)
        n_all_ctrl = self.ctrl.size // self.ctrl.shape[-1]
        self.indices = np.arange(n_all_ctrl).reshape(self.ctrl.shape[:-1])

        self.ctrl_offset = patch_offset * ncp * ncp
        self.patch_offset = patch_offset

        self.n_patches = 1

    def compute_internal_constraints(self):
        # Unit squares do not have internal constraints.
        return (np.array([], dtype=np.int), np.array([], dtype=np.int))

    def get_side_indices(self, side):
        if side == 'top':
            return self.indices[:, -1] + self.ctrl_offset
        elif side == 'bottom':
            return self.indices[:, 0] + self.ctrl_offset
        elif side == 'left':
            return self.indices[0, :] + self.ctrl_offset
        elif side == 'right':
            return self.indices[-1, :] + self.ctrl_offset
        else:
            raise ValueError(f'Invalid side {side}')

    def get_side_index_array(self, side, global_n_patches):
        ind_array = np.zeros((global_n_patches, self.ncp, self.ncp))

        if side == 'top':
            ind_array[self.patch_offset, :, -1] = 1
        elif side == 'bottom':
            ind_array[self.patch_offset, :, 0] = 1
        elif side == 'left':
            ind_array[self.patch_offset, 0, :] = 1
        elif side == 'right':
            ind_array[self.patch_offset, -1, :] = 1
        else:
            raise ValueError(f'Invalid side {side}')

        return ind_array

    def get_side_orientation(self, side, global_n_patches):
        ind_array = np.zeros((global_n_patches, 4))

        if side == 'top':
            ind_array[self.patch_offset, 1] = 1
        elif side == 'bottom':
            ind_array[self.patch_offset, 3] = 1
        elif side == 'left':
            ind_array[self.patch_offset, 0] = 1
        elif side == 'right':
            ind_array[self.patch_offset, 2] = 1
        else:
            raise ValueError(f'Invalid side {side}')

        return ind_array


def generate_pore_shapes_geometry(config, material):
    patch_ncp = config.ncp
    radial_ncp = config.radial_ncp
    quad_degree = config.quad_deg
    spline_degree = config.spline_deg

    xknots = bsplines.default_knots(spline_degree, radial_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    lines = [l.split(' ') for l in config.grid_str.strip().split('\n')[::-1]]
    # Grid will be specified as cells or solid squares (maybe add other shapes in the future)
    # Each location has two characters: First is shape type, second is boundary condition class.
    # S - square, C - cell, 0 - empty
    # 0 - no BC, 1...n - BC class
    # 00 00 00 00 00 00
    # S2 C0 C0 S0 S1 00
    # 00 S1 S1 S1 00 00

    # Let's transpose to be consistent with the way the code was written before
    cell_array = np.array(lines).T

    widths = config.cell_length * np.ones(cell_array.shape[0])
    heights = config.cell_length * np.ones(cell_array.shape[1])

    width_mesh = np.concatenate([np.array([0.0]), np.cumsum(widths)])
    height_mesh = np.concatenate([np.array([0.0]), np.cumsum(heights)])
    total_mesh = np.stack(np.meshgrid(width_mesh, height_mesh), axis=-1)[..., ::-1]

    # Map cell_array indices to a linear index. Create control point array.
    arr2lin = {}
    units = []
    ctrls = []
    fixed = []
    fixed_groups = defaultdict(list)
    traction_groups = defaultdict(list)

    npatches = 0
    n_cells = 0
    num_x = cell_array.shape[0]
    num_y = cell_array.shape[1]

    corner_width_indices = []
    corner_height_indices = []
    for i in range(num_x):
        for j in range(num_y):
            if cell_array[i, j][0] != '0':
                corners = np.array([
                    [width_mesh[i], height_mesh[j]],  # bottom left
                    [width_mesh[i], height_mesh[j+1]],  # top left
                    [width_mesh[i+1], height_mesh[j+1]],  # top right
                    [width_mesh[i+1], height_mesh[j]]  # bottom right
                ])

                if cell_array[i, j][0] == 'S':
                    # Construct a square and add to cells and ctrls.
                    unit = UnitSquare2D(corners, patch_ncp, npatches)
                elif cell_array[i, j][0] == 'C':
                    # Construct a cell.
                    unit = UnitCell2D(corners, patch_ncp, radial_ncp, npatches)
                    corner_width_indices.append([i, i, i+1, i+1])
                    corner_height_indices.append([j, j+1, j+1, j])
                    n_cells += 1
                else:
                    raise ValueError("Invalid shape.")

                npatches += unit.n_patches
                units.append(unit)
                ctrls.append(unit.ctrl.reshape(-1, radial_ncp, patch_ncp, 2))
                arr2lin[(i, j)] = len(units) - 1

                if cell_array[i, j][1] != '0':
                    group = cell_array[i, j][1]
                    fixed.append((len(units)-1, 'left'))
                    fixed_groups[group].append(
                        (len(units)-1, 'left'))

                if cell_array[i, j][2] != '0':
                    group = cell_array[i, j][2]
                    fixed.append((len(units)-1, 'top'))
                    fixed_groups[group].append(
                        (len(units)-1, 'top'))

                if cell_array[i, j][3] != '0':
                    group = cell_array[i, j][3]
                    fixed.append((len(units)-1, 'right'))
                    fixed_groups[group].append(
                        (len(units)-1, 'right'))

                if cell_array[i, j][4] != '0':
                    group = cell_array[i, j][4]
                    fixed.append((len(units)-1, 'bottom'))
                    fixed_groups[group].append(
                        (len(units)-1, 'bottom'))

    frame_ctrls = jnp.concatenate(ctrls, axis=0)

    dirichlet_labels = {}
    for group in fixed_groups:
        sides = fixed_groups[group]
        group_array = \
            sum(units[i].get_side_index_array(
                side, npatches) for (i, side) in sides)
        dirichlet_labels[group] = group_array

    x_coor = num_x * config.cell_length
    y_coor = num_y * config.cell_length


    corner_groups = {}
    corner_groups['99'] = np.sum(np.abs(frame_ctrls - np.array([x_coor, y_coor])), axis=-1) < 1e-14
    corner_groups['98'] = np.sum(np.abs(frame_ctrls - np.array([x_coor, 0.0])), axis=-1) < 1e-14
    corner_groups['97'] = np.sum(np.abs(frame_ctrls - np.array([0.0, y_coor])), axis=-1) < 1e-14
    corner_groups['96'] = np.sum(np.abs(frame_ctrls - np.array([0.0, 0.0])), axis=-1) < 1e-14

    # Construct the radii_to_ctrl function for initialization of control points.
    all_corners = []
    all_indices = []
    for u in units:
        if isinstance(u, UnitCell2D):
            all_corners.append(u.corners)
            all_indices.extend(
                list(range(u.patch_offset, u.patch_offset + 4)))

    all_dirichlet_labels = {**dirichlet_labels, **corner_groups}

    internal_corners = total_mesh[config.internal_corners[:, 0], config.internal_corners[:, 1]]
    init_central_radii = np.ones((patch_ncp-1)*config.internal_corners.shape[0]) * config.internal_radii
    internal_ctrls = UnitCell2D._gen_cell(
            internal_corners, radial_ncp, np.ones((patch_ncp-1)*config.internal_corners.shape[0]) * config.internal_radii, normalized=config.normalized_init)

    # Augment the dirichlet_labels with the new control points.
    for key in all_dirichlet_labels.keys():
        all_dirichlet_labels[key] = np.concatenate((all_dirichlet_labels[key], np.zeros(internal_ctrls.shape[:-1])), axis=0)

    # config.internal_corners is also the number of patches on the inside of the large pore.
    internal_ctrls_indices = np.arange(frame_ctrls.shape[0], frame_ctrls.shape[0] + config.internal_corners.shape[0])

    all_ctrls = jnp.concatenate((frame_ctrls, internal_ctrls), axis=0)

    def get_central_pore_points(ctrl):
        # Get last n patches
        n_central_patches = config.internal_corners.shape[0]
        last_ctrls = ctrl[-n_central_patches:]
        return last_ctrls[:, 0, :-1].reshape(-1, 2)

    all_corners = np.stack(all_corners, axis=0)
    all_indices = np.array(all_indices)

    corner_width_indices = np.stack(corner_width_indices, axis=0)
    corner_height_indices = np.stack(corner_height_indices, axis=0)
    all_corners = total_mesh[corner_width_indices, corner_height_indices]

    vmap_gencell = jax.vmap(UnitCell2D._gen_cell, in_axes=(0, None, 0))

    total_mesh = jnp.array(total_mesh)
    init_mesh_perturb = np.zeros_like(total_mesh[1:-1, 1:-1])
    def radii_to_ctrl(radii, central_radii, mesh_perturb=None):
        if mesh_perturb is not None:
            modified_total_mesh = total_mesh.at[1:-1, 1:-1].add(mesh_perturb)
        else:
            modified_total_mesh = total_mesh

        all_corners = modified_total_mesh[corner_width_indices, corner_height_indices]
        internal_corners = modified_total_mesh[config.internal_corners[:, 0], config.internal_corners[:, 1]]

        cell_ctrls = vmap_gencell(all_corners, radial_ncp, radii).reshape(
            (-1, radial_ncp, patch_ncp, 2))
        new_frame_ctrls = frame_ctrls.at[all_indices].set(cell_ctrls, indices_are_sorted=True)

        new_internal_ctrls = UnitCell2D._gen_cell(internal_corners, radial_ncp, central_radii, normalized=config.normalized_init)
        return jnp.concatenate((new_frame_ctrls, new_internal_ctrls), axis=0)

    flat_ctrls = all_ctrls.reshape((-1, 2))

    # Find all constraints with a KDTree. Should take O(n log n) time,
    # and much preferable to manually constructing constraints.
    #print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))
    #print('\tDone.')

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=all_dirichlet_labels,
        traction_labels={},
        comm=MPI.COMM_WORLD,
    ), radii_to_ctrl, n_cells, get_central_pore_points, init_central_radii, init_mesh_perturb


def generate_random_radii(shape, patch_ncp, seed=None):
    npr.seed(seed)
    init_radii = npr.rand(*shape, (patch_ncp-1)*4)*0.7 + 0.15
    return init_radii


def generate_rectangular_radii(shape, patch_ncp):
    init_radii = np.ones((*shape, (patch_ncp-1)*4)) * 0.5 * 0.9 + 0.05
    return init_radii


def generate_circular_radii(shape, patch_ncp):
    one_arc = 0.6 * \
        np.cos(np.linspace(-np.pi/4, np.pi/4, patch_ncp)[:-1])
    init_radii = np.broadcast_to(
        np.tile(one_arc, 4), (*shape, (patch_ncp-1)*4))
    return init_radii


def generate_bertoldi_radii(shape, patch_ncp, c1, c2, L0=5, phi0=0.5):
    # L0 is used in the original formula, but we want 0 to 1.
    r0 = np.sqrt(2 * phi0 / np.pi * (2 + c1**2 + c2**2))
    thetas = np.linspace(-np.pi/4, np.pi/4, patch_ncp)[:-1]
    r_theta = r0 * (1 + c1 * np.cos(4 * thetas) + c2 * np.cos(8 * thetas))
    xs_theta = np.cos(thetas) * r_theta
    init_radii = np.broadcast_to(
        np.tile(xs_theta, 4), (*shape, (patch_ncp-1)*4))
    return init_radii

