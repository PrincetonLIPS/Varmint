import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr

from varmint.patch2d      import Patch2D
from varmint.bsplines     import default_knots
from varmint.lagrangian   import generate_patch_lagrangian
from varmint.statics      import generate_patch_free_energy

import constructive_shape

from collections import defaultdict, namedtuple


CellShape = namedtuple('CellShape', [
  'num_cp', 'quad_degree', 'spline_degree'
])


def register_dirichlet_bc(group, cell):
  def inner(fn):
    vel_fn = jax.jacfwd(fn)  # Differentiate position to get velocity.
    def decorated(t):
      return fn(t) * cell.group_labels[group][..., np.newaxis]
    
    def decorated_vel(t):
      return vel_fn(t) * cell.group_labels[group][..., np.newaxis]
    
    cell.bc_movements[group] = (decorated, decorated_vel)
    return decorated
  return inner


def register_traction_bc(group, cell):
  def inner(fn):
    def decorated(t):
      return fn(t) * cell.traction_groups[group][..., np.newaxis]
    
    cell.traction_fns[group] = decorated
    return decorated
  return inner


class Cell2D:
  def __init__(self, cell_shape, material, instr):
    """
    Initialize the cell class.

    cell_shape (CellShape): Parameters for shape and representation of cell.
    init_radii (np.array): Initial radii to use. Shape (num_x, num_y, (num_cp - 1)*4)
    fixed_side (choice): 'left' or 'bottom' supported.
    """

    self.cs = cell_shape

    xknots = default_knots(self.cs.spline_degree, self.cs.num_cp)
    yknots = default_knots(self.cs.spline_degree, self.cs.num_cp)

    material_grid = constructive_shape.MaterialGrid(instr, self.cs.num_cp)

    self.n_components, self.index_arr = \
        material_grid.n_components, material_grid.labels

    self.fixed_labels = material_grid.fixed_labels
    self.nonfixed_labels = material_grid.nonfixed_labels
    self.group_labels = material_grid.group_labels
    self.bc_movements = dict()
    self.traction_fns = dict()
    self.radii_to_ctrl_fn = material_grid.get_radii_to_ctrl_fn()
    self.n_cells = material_grid.n_cells

    self.orientations = material_grid.all_orientations
    self.traction_groups = material_grid.traction_group_labels

    # Sparsity things
    self.sparse_jvps_mat = material_grid.jvpmat
    self.sparse_reconstruct = material_grid.reconstruct

    self.init_ctrls = material_grid.ctrls
    self.init_ctrls = self.init_ctrls.reshape(-1, self.cs.num_cp, self.cs.num_cp, 2)
    self.index_arr = self.index_arr.reshape(-1, self.cs.num_cp, self.cs.num_cp)

    self.patch = Patch2D(
      xknots,
      yknots,
      self.cs.spline_degree,
      material,
      self.cs.quad_degree,
      None, #labels[ii,:,:],
      self.fixed_labels, # <-- Labels not locations
    )

  ### Some helper functions to generate radii
  def generate_random_radii(self, shape, seed=None):
    npr.seed(seed)
    init_radii = npr.rand(*shape, (self.cs.num_cp-1)*4)*0.7 + 0.15
    return init_radii
  
  def generate_rectangular_radii(self, shape):
    init_radii = np.ones((*shape, (self.cs.num_cp-1)*4)) * 0.5 * 0.9 + 0.05
    return init_radii
  
  def generate_circular_radii(self, shape):
    one_arc = 0.6 * np.cos(np.linspace(-np.pi/4, np.pi/4, self.cs.num_cp)[:-1])
    init_radii = np.broadcast_to(np.tile(one_arc, 4), (*shape, (self.cs.num_cp-1)*4))
    return init_radii
  
  def generate_bertoldi_radii(self, shape, c1, c2, L0=5, phi0=0.5):
    # L0 is used in the original formula, but we want 0 to 1.
    r0 = np.sqrt(2 * phi0 / np.pi * (2 + c1**2 + c2**2))
    thetas = np.linspace(-np.pi/4, np.pi/4, self.cs.num_cp)[:-1]
    r_theta = r0 * (1 + c1 * np.cos(4 * thetas) + c2 * np.cos(8 * thetas))
    xs_theta = np.cos(thetas) * r_theta
    init_radii = np.broadcast_to(np.tile(xs_theta, 4), (*shape, (self.cs.num_cp-1)*4))
    return init_radii

  def get_dynamics_flatten_unflatten(self):
    def flatten(unflat_pos, unflat_vel):
      kZeros = np.zeros((self.n_components, 2))

      flat_pos = jax.ops.index_update(kZeros, self.index_arr, unflat_pos)
      flat_vel = jax.ops.index_update(kZeros, self.index_arr, unflat_vel)

      flat_pos = np.take(flat_pos, self.nonfixed_labels, axis=0)
      flat_vel = np.take(flat_vel, self.nonfixed_labels, axis=0)

      return flat_pos.flatten(), flat_vel.flatten()

    def unflatten(flat_pos, flat_vel, fixed_locs, fixed_vels):
      kZeros = np.zeros((self.n_components, 2))

      fixed_locs = jax.ops.index_update(kZeros, self.index_arr, fixed_locs)
      fixed_locs = np.take(fixed_locs, self.fixed_labels, axis=0)

      fixed_vels = jax.ops.index_update(kZeros, self.index_arr, fixed_vels)
      fixed_vels = np.take(fixed_vels, self.fixed_labels, axis=0)

      unflat_pos  = kZeros
      unflat_vel  = kZeros

      flat_pos = flat_pos.reshape((-1, 2))
      flat_vel = flat_vel.reshape((-1, 2))

      unflat_pos = jax.ops.index_update(unflat_pos, self.nonfixed_labels, flat_pos)
      unflat_vel = jax.ops.index_update(unflat_vel, self.nonfixed_labels, flat_vel)

      fixed_pos = jax.ops.index_update(unflat_pos, self.fixed_labels,
                                                   fixed_locs)
      fixed_vel = jax.ops.index_update(unflat_vel, self.fixed_labels,
                                                   fixed_vels)

      return np.take(fixed_pos, self.index_arr, axis=0), \
             np.take(fixed_vel, self.index_arr, axis=0)

    return flatten, unflatten

  def unflatten_dynamics_sequence(self, QQ, PP, fixed_locs, fixed_vels):
    _, unflatten = self.get_dynamics_flatten_unflatten()
    unflat_pos, unflat_vel = zip(*[unflatten(q, p, f, v) for q, p, f, v in zip(QQ, PP, fixed_locs, fixed_vels)])

    return unflat_pos, unflat_vel

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

  def get_statics_flatten_add(self):
    def flatten_add(unflat_pos):
      kZeros = np.zeros((self.n_components, 2))

      # flatten to connected components, accumulating shared points.
      flat_pos = jax.ops.index_add(kZeros, self.index_arr, unflat_pos)

      # Remove fixed locations
      flat_pos = np.take(flat_pos, self.nonfixed_labels, axis=0)

      return flat_pos.flatten()

    return flatten_add

  def get_fixed_locs_fn(self, ref_ctrl):
    pos_fns = []
    vel_fns = []
    for group in self.group_labels:
      pos, vel = self.bc_movements.get(group, (None, None))

      if pos is not None:
        pos_fns.append(pos)
        vel_fns.append(vel)
    
    def fixed_locs_fn(t):
      return ref_ctrl + sum(fn(t) for fn in pos_fns)
    
    def fixed_vels_fn(t):
      return np.zeros_like(ref_ctrl) + sum(fn(t) for fn in vel_fns)
    
    return fixed_locs_fn, fixed_vels_fn
  
  def get_traction_fn(self):
    fns = []
    for group in self.traction_groups:
      fns.append(self.traction_fns.get(group, lambda _: 0.0))
    
    def traction_fn(t):
      return np.zeros((self.index_arr.shape[0], 4, 2)) + sum(fn(t) for fn in fns)
    
    return traction_fn

  def get_lagrangian_fun(self, patchwise=False):
    p_lagrangian = generate_patch_lagrangian(self.patch)

    if patchwise:
      return p_lagrangian
    flatten, unflatten = self.get_dynamics_flatten_unflatten()

    def full_lagrangian(q, qdot, ref_ctrl, displacement, velocity, traction):
      def_ctrl, def_vels = unflatten(q, qdot, displacement, velocity)

      # Map instead of vmap over each patch to save memory. Otherwise
      # vmap will be over all patches, elements, and quad points.
      def fn_for_map(x):
        return p_lagrangian(*x)
      
      #return np.sum(jax.lax.map(fn_for_map, (def_ctrl, def_vels, ref_ctrl,
      #                                       self.orientations, traction)))

      return np.sum(jax.vmap(p_lagrangian)(def_ctrl, def_vels, ref_ctrl,
                                           self.orientations, traction))

    return full_lagrangian

  def get_free_energy_fun(self, patchwise=False):
    p_free_energy = generate_patch_free_energy(self.patch)

    if patchwise:
      return p_free_energy
    
    flatten, unflatten = self.get_statics_flatten_unflatten()

    def full_free_energy(q, ref_ctrl):
      def_ctrl = unflatten(q, ref_ctrl)
      return np.sum(jax.vmap(p_free_energy)(def_ctrl, ref_ctrl))
    return full_free_energy
