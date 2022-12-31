import jax
import jax.numpy as jnp
import numpy as onp

from scipy import ndimage

import matplotlib.pyplot as plt


def base_domain(len_x, len_y, fidelity):
    """Generate a dense rectangular grid of cells with the given fidelity."""

    nx, ny = fidelity, fidelity
    xcoords = onp.linspace(0.0, len_x, nx + 1)
    ycoords = onp.linspace(0.0, len_y, ny + 1)

    coords = onp.asarray(onp.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    i, j = onp.meshgrid(onp.arange(nx), onp.arange(ny))

    cells = [
        i * (ny + 1) + j,
        i * (ny + 1) + j + 1,
        (i + 1) * (ny + 1) + j,
        (i + 1) * (ny + 1) + j + 1,
    ]
    cells = onp.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
    return coords, cells


def pixelize_implicit(domain_oracle, params, len_x, len_y, fidelity, negative=True):
    """Given domain oracle, size of rectangular domain, and fidelity, discretize the domain.
    
    `domain_oracle` should take two inputs: `params`, and a single 2-D coordinate.
    It should output a scalar. It will internally be vectorized wrt the second input.

    If negative=True, the material exists where `domain_oracle < 0`. The discretization
    is meant to overestimate; if `domain_oracle < 0` on any cell corner, it will be included.
    """

    coords, cells = base_domain(len_x, len_y, fidelity)
    v_oracle = jax.vmap(domain_oracle, in_axes=(None, 0))

    if negative:
        node_occupancy = v_oracle(params, coords) < 0.0
    else:
        node_occupancy = v_oracle(params, coords) > 0.0

    # bool array in the shape of cells.shape[0] 
    cell_occupancy = onp.any(node_occupancy[cells], axis=-1)
    filtered_cells = cells[cell_occupancy, :]
    inv_cell_ids = onp.ones(cells.shape[0], dtype=onp.int32) * -1
    inv_cell_ids[cell_occupancy] = onp.arange(filtered_cells.shape[0])
    inv_cell_ids = jnp.array(inv_cell_ids)
    cells = filtered_cells

    # Indexing magic to reindex cell ids and coords.
    inv_ids = onp.zeros_like(coords[:, 0], dtype=onp.int32)
    ids = onp.unique(cells)
    inv_ids[ids] = onp.arange(ids.shape[0], dtype=onp.int32)
    cells = inv_ids[cells]
    coords = coords[ids]

    # Define a function to transform a point in physical space to 
    # a cell index and coordinate within the cell in [0, 1] x [0, 1].
    def find_patch(point):
        # Find cell index
        large_point = point * fidelity / jnp.array([len_x, len_y]) 
        i = jnp.trunc(large_point[0]).astype(onp.int32)
        j = jnp.trunc(large_point[1]).astype(onp.int32)

        cell_index = i * fidelity + j
        new_cell_index = inv_cell_ids[cell_index]

        return new_cell_index, large_point % 1.0

    return coords, cells, find_patch
