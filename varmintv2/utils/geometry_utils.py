"""Various utilities for testing geometries."""

import jax
import jax.numpy as jnp
import numpy as onp

from scipy.sparse.coo import coo_matrix
from scipy.spatial.distance import pdist, squareform


def verify_constraints(ctrl, constraints):
    """Verify whether a set of SingleElementGeometry constraints
    are satisfied by given control points.
    """

    all_rows, all_cols = constraints
    flat_ctrl = ctrl.reshape(-1, ctrl.shape[-1])
    return onp.allclose(flat_ctrl[all_rows, :], flat_ctrl[all_cols, :],
                        rtol=0.0, atol=1e-14)


def generate_constraints(ctrl):
    """Generate constraints satisfied by SingleElementGeometry control points.
    
    This should only really be used for testing, since it will be inefficient.
    """

    n_cp = ctrl.size // ctrl.shape[-1]
    local_indices = onp.arange(n_cp).reshape(ctrl.shape[:-1])
    flat_ctrl = ctrl.reshape(-1, ctrl.shape[-1])

    # Compute inter-control point distances.
    dists = squareform(pdist(flat_ctrl))
    dists = dists < 1e-10
    adjacency = coo_matrix(dists - onp.diag(onp.ones(n_cp)))

    return (adjacency.row, adjacency.col)


def get_patch_side_indices(ctrl, patch_num, side, ctrl_offset=0):
    n_cp = ctrl.size // ctrl.shape[-1]
    local_indices = onp.arange(n_cp).reshape(ctrl.shape[:-1])

    if side == 'top':
        return local_indices[:, -1] + ctrl_offset
    elif side == 'bottom':
        return local_indices[:, 0] + ctrl_offset
    elif side == 'left':
        return local_indices[0, :] + ctrl_offset
    elif side == 'right':
        return local_indices[-1, :] + ctrl_offset
    else:
        raise ValueError(f'Invalid side {side}')


def get_patch_side_index_array(ctrl, patch_num, side):
    """Get the index array representing the side of a bspline patch."""

    ind_array = onp.zeros(ctrl.shape[:-1])

    if side == 'top':
        ind_array[patch_num, :, -1] = 1
    elif side == 'bottom':
        ind_array[patch_num, :, 0] = 1
    elif side == 'left':
        ind_array[patch_num, 0, :] = 1
    elif side == 'right':
        ind_array[patch_num, -1, :] = 1
    else:
        raise ValueError(f'Invalid side {side}')

    return ind_array


def constrain_ctrl(ctrl, constraints):
    """Arbitrarily force constraints to be true."""

    flat_ctrl = ctrl.reshape(-1, ctrl.shape[-1])

    # lol
    flat_ctrl[constraints[0]] = flat_ctrl[constraints[1]] \
                              = flat_ctrl[constraints[0]] + flat_ctrl[constraints[1]]

    return flat_ctrl.reshape(*ctrl.shape)