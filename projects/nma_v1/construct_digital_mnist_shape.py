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


def generate_digital_mnist_shape(config, material):
    spline_degree = config.spline_deg
    ncp = config.ncp
    quad_degree = config.quad_deg

    xknots = bsplines.default_knots(spline_degree, ncp)
    yknots = bsplines.default_knots(spline_degree, ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    cell_size = config.cell_size
    border_size = config.border_size


    # Define the mesh shell of the shape.
    x_corners = np.array([
        0.0,
        cell_size, 
        cell_size + border_size,
        2 * cell_size + border_size,
        3 * cell_size + border_size,
        3 * cell_size + 2 * border_size,
        4 * cell_size + 2 * border_size
    ])
    x_cell_indices = np.where(np.diff(x_corners) == cell_size)[0]
    x_all_indices = np.arange(x_corners.size - 1)

    y_corners = np.array([
        0.0,
        cell_size,
        cell_size + border_size,
        2 * cell_size + border_size,
        3 * cell_size + border_size,
        3 * cell_size + 2 * border_size,
        4 * cell_size + 2 * border_size,
        5 * cell_size + 2 * border_size,
        5 * cell_size + 3 * border_size,
        6 * cell_size + 3 * border_size
    ])
    y_cell_indices = np.where(np.diff(y_corners) == cell_size)[0]
    y_all_indices = np.arange(y_corners.size - 1)

    all_corners = np.stack(np.meshgrid(x_corners, y_corners), axis=-1)
    all_indices = np.stack(np.meshgrid(x_all_indices, y_all_indices), axis=-1).reshape(-1, 2)[:, ::-1]
    all_cell_indices = np.stack(np.meshgrid(x_cell_indices, y_cell_indices), axis=-1).reshape(-1, 2)[:, ::-1]

    # Hack to find all indices that are not cell indices.
    all_border_indices = np.stack([index for index in all_indices if not np.any(np.all(index == all_cell_indices, axis=-1))], axis=0)

    cell_bottom_left = all_corners[all_cell_indices[:, 0], all_cell_indices[:, 1]]
    border_bottom_left = all_corners[all_border_indices[:, 0], all_border_indices[:, 1]]


    # Create control points using the constructors.
    ctrls = []
    cells = []
    borders = []

    radii_to_ctrl_corners = []
    radii_to_ctrl_indices = []
    n_cells = 0
    for cell_x, cell_y in all_cell_indices:
        cell_corners = all_corners[
            [cell_x, cell_x, cell_x + 1, cell_x + 1],
            [cell_y, cell_y + 1, cell_y + 1, cell_y]
        ]
        cells.append(UnitCell2D(cell_corners, ncp, ncp, len(cells) * 4))
        ctrls.append(cells[-1].ctrl.reshape(-1, ncp, ncp, 2))
        radii_to_ctrl_corners.append(cells[-1].corners)
        radii_to_ctrl_indices.extend(list(range(cells[-1].patch_offset, cells[-1].patch_offset + 4)))
        n_cells += 1

    radii_to_ctrl_corners = np.stack(radii_to_ctrl_corners, axis=0)
    radii_to_ctrl_indices = np.array(radii_to_ctrl_indices)

    for border_x, border_y in all_border_indices:
        border_corners = all_corners[
            [border_x, border_x, border_x + 1, border_x + 1],
            [border_y, border_y + 1, border_y + 1, border_y]
        ]
        borders.append(UnitSquare2D(border_corners, ncp, ncp, len(cells) * 4 + len(borders)))
        ctrls.append(borders[-1].ctrl.reshape(1, ncp, ncp, 2))

    all_ctrls = np.concatenate(ctrls, axis=0)

    vmap_gencell = jax.vmap(UnitCell2D._gen_cell, in_axes=(0, None, 0))
    vmap_gencell = partial(vmap_gencell, radii_to_ctrl_corners, ncp)

    all_ctrls = jnp.array(all_ctrls)
    def radii_to_ctrl(radii):
        cell_ctrls = vmap_gencell(radii).reshape(
                (-1, ncp, ncp, 2))
        return all_ctrls.at[radii_to_ctrl_indices].set(cell_ctrls, indices_are_sorted=True)

    x_max = all_ctrls[..., 0].max()
    y_max = all_ctrls[..., 1].max()

    # Define the Dirichlet boundary condition groups.
    corner_groups = {}
    corner_groups['99'] = np.sum(np.abs(all_ctrls - np.array([x_max, y_max])), axis=-1) < 1e-14
    corner_groups['98'] = np.sum(np.abs(all_ctrls - np.array([x_max, 0.0])), axis=-1) < 1e-14
    corner_groups['97'] = np.sum(np.abs(all_ctrls - np.array([0.0, y_max])), axis=-1) < 1e-14
    corner_groups['96'] = np.sum(np.abs(all_ctrls - np.array([0.0, 0.0])), axis=-1) < 1e-14

    # g1: x = 0, cell_size + border_size < y < 3 * cell_size + border_size
    # g2: x = 0, 3 * cell_size + 2 * border_size < y < 5 * cell_size + 2 * border_size
    # g3: cell_size + border_size < x < 3 * cell_size + border_size, y = y_max
    # g4: x = x_max, 3 * cell_size + 2 * border_size < y < 5 * cell_size + 2 * border_size
    # g5: x = x_max, cell_size + border_size < y < 3 * cell_size + border_size
    # g6: cell_size + border_size < x < 3 * cell_size + border_size, y = 0

    dirichlet_g1 = \
        (np.abs(all_ctrls[..., 0] - 0.0) < 1e-14) & \
        (all_ctrls[..., 1] > 1 * cell_size + 1 * border_size - 1e-14) & \
        (all_ctrls[..., 1] < 3 * cell_size + 1 * border_size + 1e-14)

    dirichlet_g2 = \
        (np.abs(all_ctrls[..., 0] - 0.0) < 1e-14) & \
        (all_ctrls[..., 1] > 3 * cell_size + 2 * border_size - 1e-14) & \
        (all_ctrls[..., 1] < 5 * cell_size + 2 * border_size + 1e-14)

    dirichlet_g3 = \
        (np.abs(all_ctrls[..., 1] - y_max) < 1e-14) & \
        (all_ctrls[..., 0] > 1 * cell_size + 1 * border_size - 1e-14) & \
        (all_ctrls[..., 0] < 3 * cell_size + 1 * border_size + 1e-14)

    dirichlet_g4 = \
        (np.abs(all_ctrls[..., 0] - x_max) < 1e-14) & \
        (all_ctrls[..., 1] > 3 * cell_size + 2 * border_size - 1e-14) & \
        (all_ctrls[..., 1] < 5 * cell_size + 2 * border_size + 1e-14)

    dirichlet_g5 = \
        (np.abs(all_ctrls[..., 0] - x_max) < 1e-14) & \
        (all_ctrls[..., 1] > 1 * cell_size + 1 * border_size - 1e-14) & \
        (all_ctrls[..., 1] < 3 * cell_size + 1 * border_size + 1e-14)

    dirichlet_g6 = \
        (np.abs(all_ctrls[..., 1] - 0.0) < 1e-14) & \
        (all_ctrls[..., 0] > 1 * cell_size + 1 * border_size - 1e-14) & \
        (all_ctrls[..., 0] < 3 * cell_size + 1 * border_size + 1e-14)

    all_dirichlet_groups = {
        '1': dirichlet_g1,
        '2': dirichlet_g2,
        '3': dirichlet_g3,
        '4': dirichlet_g4,
        '5': dirichlet_g5,
        '6': dirichlet_g6,
        **corner_groups,
    }

    flat_ctrls = all_ctrls.reshape((-1, 2))

    # Find all constraints with a KDTree. Should take O(n log n) time,
    # and much preferable to manually constructing constraints.
    #print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=all_ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=all_dirichlet_groups,
        traction_labels={},
        comm=MPI.COMM_WORLD,
    ), radii_to_ctrl, n_cells


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

