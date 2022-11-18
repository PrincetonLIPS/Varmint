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
    def _gen_cell_edge(center, corner1, corner2, radii):
        num_ctrl = len(radii)

        right_perim = jnp.linspace(corner1, corner2, num_ctrl)
        left_perim = radii[:, jnp.newaxis] * (right_perim - center) + center

        ctrl = jnp.linspace(left_perim, right_perim, num_ctrl)

        return ctrl

    @staticmethod
    def _gen_cell(corners, radii):
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
                corner1,
                corner2,
                jnp.take(radii, indices, mode='wrap'),
            )

            ctrl.append(new_ctrl)

        return jnp.stack(ctrl, axis=0)

    def __init__(self, corners, ncp, patch_offset, side_labels=None, radii=None):
        if radii is None:
            radii = UnitCell2D._gen_random_radii(ncp)

        if side_labels is not None:
            # side_labels can handle boundary conditions by creating groups
            # upon which you can impose boundary conditions.
            # Generally sides that are shared should not have a label.
            self.side_labels = side_labels

        self.ncp = ncp
        self.corners = corners
        self.ctrl = UnitCell2D._gen_cell(corners, radii)
        n_all_ctrl = self.ctrl.size // self.ctrl.shape[-1]
        self.indices = np.arange(n_all_ctrl).reshape(self.ctrl.shape[:-1])
        self.ctrl_offset = patch_offset * ncp * ncp
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
        ind_array = np.zeros((global_n_patches, self.ncp, self.ncp))

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


def get_connectivity_matrix(num_x, num_y, arr2lin, units, global_ctrl):
    n_cp = global_ctrl.size // global_ctrl.shape[-1]
    unflat_indices = np.arange(n_cp).reshape(global_ctrl.shape[:-1])

    row_inds = []
    col_inds = []

    # Handle internal constraints for each unit
    for u in units:
        new_row, new_col = u.compute_internal_constraints()
        row_inds.append(new_row)
        col_inds.append(new_col)

    # Handle the constraints between cells
    for x in range(num_x):
        for y in range(num_y):
            if (x, y) not in arr2lin:
                continue

            this_unit = units[arr2lin[(x, y)]]
            if x > 0 and (x-1, y) in arr2lin:
                # Handle constraint with left
                that_unit = units[arr2lin[(x-1, y)]]
                side1 = this_unit.get_side_indices('left')
                side2 = that_unit.get_side_indices('right')

                row_inds.append(side1)
                col_inds.append(side2)

            if y > 0 and (x, y-1) in arr2lin:
                # Handle constraint with bottom
                that_unit = units[arr2lin[(x, y-1)]]
                side1 = this_unit.get_side_indices('bottom')
                side2 = that_unit.get_side_indices('top')

                row_inds.append(side1)
                col_inds.append(side2)

            if x < num_x - 1 and (x+1, y) in arr2lin:
                # Handle constraint with right
                that_unit = units[arr2lin[(x+1, y)]]
                side1 = this_unit.get_side_indices('right')
                side2 = that_unit.get_side_indices('left')

                row_inds.append(side1)
                col_inds.append(side2)

            if y < num_y - 1 and (x, y+1) in arr2lin:
                # Handle constraint with top
                that_unit = units[arr2lin[(x, y+1)]]
                side1 = this_unit.get_side_indices('top')
                side2 = that_unit.get_side_indices('bottom')

                row_inds.append(side1)
                col_inds.append(side2)

    all_rows = np.concatenate(row_inds)
    all_cols = np.concatenate(col_inds)

    return unflat_indices, (all_rows, all_cols)


def construct_cell2D(input_str, patch_ncp, quad_degree, spline_degree,
                     material: PhysicsModel) -> Tuple[SingleElementGeometry, Callable, int]:
    cell_length = 5  # TODO(doktay): This is arbitrary.

    xknots = bsplines.default_knots(spline_degree, patch_ncp)
    yknots = bsplines.default_knots(spline_degree, patch_ncp)

    element = elements.Patch2D(xknots, yknots, spline_degree, quad_degree)

    lines = [l.split(' ') for l in input_str.strip().split('\n')[::-1]]
    # Grid will be specified as cells or solid squares (maybe add other shapes in the future)
    # Each location has two characters: First is shape type, second is boundary condition class.
    # S - square, C - cell, 0 - empty
    # 0 - no BC, 1...n - BC class
    # 00 00 00 00 00 00
    # S2 C0 C0 S0 S1 00
    # 00 S1 S1 S1 00 00

    # Let's transpose to be consistent with the way the code was written before
    cell_array = np.array(lines).T

    widths = cell_length * np.ones(cell_array.shape[0])
    heights = cell_length * np.ones(cell_array.shape[1])

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
                    unit = UnitCell2D(corners, patch_ncp, npatches)
                    corner_width_indices.append([i, i, i+1, i+1])
                    corner_height_indices.append([j, j+1, j+1, j])
                    n_cells += 1
                else:
                    raise ValueError("Invalid shape.")

                npatches += unit.n_patches
                units.append(unit)
                ctrls.append(unit.ctrl.reshape(-1, patch_ncp, patch_ncp, 2))
                arr2lin[(i, j)] = len(units) - 1

                if cell_array[i, j][1] != '0':
                    group = cell_array[i, j][1]
                    if group.isdigit():
                        fixed.append((len(units)-1, 'left'))
                        fixed_groups[group].append(
                            (len(units)-1, 'left'))
                    else:
                        traction_groups[group].append(
                            (len(units)-1, 'left'))

                if cell_array[i, j][2] != '0':
                    group = cell_array[i, j][2]
                    if group.isdigit():
                        fixed.append((len(units)-1, 'top'))
                        fixed_groups[group].append(
                            (len(units)-1, 'top'))
                    else:
                        traction_groups[group].append(
                            (len(units)-1, 'top'))

                if cell_array[i, j][3] != '0':
                    group = cell_array[i, j][3]
                    if group.isdigit():
                        fixed.append((len(units)-1, 'right'))
                        fixed_groups[group].append(
                            (len(units)-1, 'right'))
                    else:
                        traction_groups[group].append(
                            (len(units)-1, 'right'))

                if cell_array[i, j][4] != '0':
                    group = cell_array[i, j][4]
                    if group.isdigit():
                        fixed.append((len(units)-1, 'bottom'))
                        fixed_groups[group].append(
                            (len(units)-1, 'bottom'))
                    else:
                        traction_groups[group].append(
                            (len(units)-1, 'bottom'))

    ctrls = jnp.concatenate(ctrls, axis=0)
    #flat_ctrls = ctrls.reshape((-1, 2))
    #kdtree = KDTree(flat_ctrls)
    #constraints = kdtree.query_pairs(1e-10)
    #constraints = np.array(list(constraints))
    #constraints = (constraints[:, 0], constraints[:, 1])

    # Now create index array from matching control points.
    unflat_indices, constraints = get_connectivity_matrix(
        num_x, num_y, arr2lin, units, ctrls)

    dirichlet_labels = {}
    for group in fixed_groups:
        sides = fixed_groups[group]
        group_array = \
            sum(units[i].get_side_index_array(
                side, npatches) for (i, side) in sides)
        dirichlet_labels[group] = group_array


    traction_labels = {}
    for group in traction_groups:
        sides = traction_groups[group]
        traction_labels[group] = \
            sum(units[i].get_side_orientation(side, npatches)
                for (i, side) in sides)

    corner_groups = {}
    #for g in range(1, num_y):
    #    y_coor = cell_length * g
    #    group_selection = \
    #        np.sum(np.abs(ctrls - np.array([0.0, y_coor])), axis=-1) < 1e-14
    #    corner_groups[str(g + 100)] = group_selection

    x_coor = num_x * cell_length
    y_coor = num_y * cell_length
    #print(x_coor)
    #print(y_coor)
    corner_groups['99'] = np.sum(np.abs(ctrls - np.array([x_coor, y_coor])), axis=-1) < 1e-14
    corner_groups['98'] = np.sum(np.abs(ctrls - np.array([x_coor, 0.0])), axis=-1) < 1e-14
    corner_groups['97'] = np.sum(np.abs(ctrls - np.array([0.0, y_coor])), axis=-1) < 1e-14
    corner_groups['96'] = np.sum(np.abs(ctrls - np.array([0.0, 0.0])), axis=-1) < 1e-14

    # Construct the radii_to_ctrl function for initialization of control points.
    all_corners = []
    all_indices = []
    for u in units:
        if isinstance(u, UnitCell2D):
            all_corners.append(u.corners)
            all_indices.extend(
                list(range(u.patch_offset, u.patch_offset + 4)))

    # Case when we have no cells.
    if len(all_corners) == 0:
        return SingleElementGeometry(
            element=element,
            material=material,
            init_ctrl=ctrls,
            constraints=constraints,
            dirichlet_labels={**dirichlet_labels, **corner_groups},
            traction_labels=traction_labels
        ), lambda _: ctrls, 0

    all_corners = np.stack(all_corners, axis=0)
    all_indices = np.array(all_indices)

    corner_width_indices = np.stack(corner_width_indices, axis=0)
    corner_height_indices = np.stack(corner_height_indices, axis=0)
    all_corners = total_mesh[corner_width_indices, corner_height_indices]

    vmap_gencell = jax.vmap(UnitCell2D._gen_cell)

    total_mesh = jnp.array(total_mesh)
    init_mesh_perturb = np.zeros_like(total_mesh[1:-1, 1:-1])
    def radii_to_ctrl(radii, mesh_perturb=None):
        if mesh_perturb is not None:
            modified_total_mesh = total_mesh.at[1:-1, 1:-1].add(mesh_perturb)
        else:
            modified_total_mesh = total_mesh

        all_corners = modified_total_mesh[corner_width_indices, corner_height_indices]
        cell_ctrls = vmap_gencell(all_corners, radii).reshape(
            (-1, patch_ncp, patch_ncp, 2))
        return ctrls.at[all_indices].set(cell_ctrls, indices_are_sorted=True)
        #return jax.ops.index_update(ctrls, all_indices, cell_ctrls,
        #                            indices_are_sorted=True)

    return SingleElementGeometry(
        element=element,
        material=material,
        init_ctrl=ctrls,
        constraints=constraints,
        dirichlet_labels={**dirichlet_labels, **corner_groups},
        traction_labels=traction_labels
    ), radii_to_ctrl, n_cells, init_mesh_perturb


# Some helper functions to generate radii
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
