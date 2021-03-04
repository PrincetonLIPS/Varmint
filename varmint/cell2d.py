import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import string

from varmint.patch2d      import Patch2D
from varmint.shape2d      import Shape2D
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D
from varmint.bsplines     import default_knots
from varmint.lagrangian   import generate_patch_lagrangian
from varmint.discretize   import get_hamiltonian_stepper
from varmint.levmar       import get_lmfunc
from varmint.cellular2d   import index_array_from_ctrl, generate_quad_lattice

import experiment_utils as exputils

import json
import logging
import random
import argparse
import time
import os

from collections import namedtuple


CellShape = namedtuple('CellShape', [
  'num_x', 'num_y', 'num_cp', 'quad_degree', 'spline_degree'
])


class Cell2D:
  def __init__(self, cell_shape, fixed_side, material):
    """
    Initialize the cell class.

    cell_shape (CellShape): Parameters for shape and representation of cell.
    init_radii (np.array): Initial radii to use. Shape (num_x, num_y, (num_cp - 1)*4)
    fixed_side (choice): 'left' or 'bottom' supported.
    """

    self.cs = cell_shape

    xknots = default_knots(self.cs.spline_degree, self.cs.num_cp)
    yknots = default_knots(self.cs.spline_degree, self.cs.num_cp)

    # TODO(doktay): The only thing we use ref_ctrl here for is to determine boundary conditions.
    # So in the current setup we are forced to create a random init_radii just to identify
    # boundaries even though we never use it. Figure out a better way to represent boundaries.
    init_radii = self.generate_random_radii()

    ref_ctrl = self.radii_to_ctrl(init_radii)
    self.n_components, self.index_arr = \
        index_array_from_ctrl(self.cs.num_x, self.cs.num_y, ref_ctrl)

    self.fixed_side_str = fixed_side
    if fixed_side == 'left':
      fixed_side  = onp.array(ref_ctrl[:,:,:,0] == 0.0)
      nonfixed_side  = onp.array(ref_ctrl[:,:,:,0] != 0.0)
    elif fixed_side == 'bottom':
      fixed_side  = onp.array(ref_ctrl[:,:,:,1] == 0.0)
      nonfixed_side  = onp.array(ref_ctrl[:,:,:,1] != 0.0)
    else:
      raise ValueError(f'Unsupported side {fixed_side}')

    self.fixed_labels = np.unique(self.index_arr[fixed_side])
    self.nonfixed_labels = np.unique(self.index_arr[nonfixed_side])

    self.patch = Patch2D(
      xknots,
      yknots,
      self.cs.spline_degree,
      material,
      self.cs.quad_degree,
      None, #labels[ii,:,:],
      self.fixed_labels, # <-- Labels not locations
    )

  def generate_random_radii(self, seed=None):
    npr.seed(seed)
    init_radii = npr.rand(self.cs.num_x, self.cs.num_y, (self.cs.num_cp-1)*4)*0.9 + 0.05
    return np.array(init_radii)

  def radii_to_ctrl(self, radii):
    widths  = 5 * np.ones(self.cs.num_x)
    heights = 5 * np.ones(self.cs.num_y)
    return generate_quad_lattice(widths, heights, radii)

  def get_dynamics_flatten_unflatten(self):
    def flatten(unflat_pos, unflat_vel):
      kZeros = np.zeros((self.n_components, 2))

      flat_pos = jax.ops.index_update(kZeros, self.index_arr, unflat_pos)
      flat_vel = jax.ops.index_update(kZeros, self.index_arr, unflat_vel)

      flat_pos = np.take(flat_pos, self.nonfixed_labels, axis=0)
      flat_vel = np.take(flat_vel, self.nonfixed_labels, axis=0)

      return flat_pos.flatten(), flat_vel.flatten()

    def unflatten(flat_pos, flat_vel, fixed_locs):
      kZeros = np.zeros((self.n_components, 2))

      fixed_locs = jax.ops.index_update(kZeros, self.index_arr, fixed_locs)
      fixed_locs = np.take(fixed_locs, self.fixed_labels, axis=0)

      unflat_pos  = kZeros
      unflat_vel  = kZeros

      flat_pos = flat_pos.reshape((-1, 2))
      flat_vel = flat_vel.reshape((-1, 2))

      unflat_pos = jax.ops.index_update(unflat_pos, self.nonfixed_labels, flat_pos)
      unflat_vel = jax.ops.index_update(unflat_vel, self.nonfixed_labels, flat_vel)

      fixed_pos = jax.ops.index_update(unflat_pos, self.fixed_labels,
                                                   fixed_locs)
      fixed_vel = jax.ops.index_update(unflat_vel, self.fixed_labels,
                                                   np.zeros_like(fixed_locs))

      return np.take(fixed_pos, self.index_arr, axis=0), \
             np.take(fixed_vel, self.index_arr, axis=0)

    return flatten, unflatten

  def unflatten_dynamics_sequence(self, QQ, fixed_locs):
    _, unflatten = self.get_dynamics_flatten_unflatten()
    ctrl_seq = [unflatten(q, np.zeros_like(q), fixed_locs)[0] for q in QQ]

    return ctrl_seq

  def get_statics_flatten_unflatten(self):
    def flatten(unflat_pos):
      kZeros = np.zeros((self.n_components, 2))

      # flatten to connected components, only taking a single (arbitrary one of each).
      flat_pos = jax.ops.index_update(kZeros, self.index_arr, unflat_pos)

      # Remove fixed locations
      flat_pos = np.take(flat_pos, self.nonfixed_labels, axis=0)

      return flat_pos.flatten()

    def unflatten(flat_pos, fixed_locs):
      kZeros = np.zeros((self.n_components, 2))

      # create the array of fixed locations
      fixed_locs = jax.ops.index_update(kZeros, self.index_arr, fixed_locs)
      fixed_locs = np.take(fixed_locs, self.fixed_labels, axis=0)

      unflat_pos  = kZeros  # with space for fixed locations
      flat_pos = flat_pos.reshape((-1, 2))  # without space for fixed locations

      # pick an arbitrary element from each connected component
      unflat_pos = jax.ops.index_update(unflat_pos, self.nonfixed_labels, flat_pos)

      # fill in the fixed locations
      unflat_pos = jax.ops.index_update(unflat_pos, self.fixed_labels,
                                                    fixed_locs)

      return np.take(unflat_pos, self.index_arr, axis=0)

    return flatten, unflatten

  def get_lagrangian_fun(self):
    p_lagrangian = generate_patch_lagrangian(self.patch)
    flatten, unflatten = self.get_dynamics_flatten_unflatten()

    def full_lagrangian(q, qdot, ref_ctrl, displacement):
      def_ctrl, def_vels = unflatten(q, qdot, displacement)
      return np.sum(jax.vmap(p_lagrangian)(def_ctrl, def_vels, ref_ctrl))

    return full_lagrangian
