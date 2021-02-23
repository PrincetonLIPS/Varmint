import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import numpy            as onp
import logging

from .vmap_utils import *


def generate_patch_free_energy(patch):
  """Generates a function that computes the free energy for a single patch.

  Assumes homogeneous patches.
  """

  jacobian_u_fn  = patch.get_cached_jacobian_u_fn()
  energy_fn      = patch.get_energy_fn()
  quad_fn        = patch.get_quad_fn()
  deformation_fn = patch.get_cached_deformation_fn()
  vmap_energy_fn = jax.vmap(energy_fn, in_axes=(0,))
  jac_dets_fn    = jax.vmap(npla.det, in_axes=(0,))

  defgrads_fn = jax.vmap(
    lambda A, B: npla.solve(B.T, A.T).T,
    in_axes=(0,0),
  )

  mat_density = patch.material.density
  gravity = 981.0 # cm/s^2

  def free_energy(def_ctrl, ref_ctrl):
    # Jacobian of reference config wrt parent config.
    def_jacs = jacobian_u_fn(def_ctrl)
    ref_jacs = jacobian_u_fn(ref_ctrl)

    # Deformation gradients. def_jacs @ ref_jacs_inv computed via a linear solve.
    defgrads = defgrads_fn(def_jacs, ref_jacs)

    # Jacobian determinants of reference config wrt parent.
    ref_jac_dets = jac_dets_fn(ref_jacs)

    # Strain energy density wrt to parent config.
    strain_energy_density = vmap_energy_fn(defgrads) * np.abs(ref_jac_dets)

    # Total potential energy via integrating over parent config.
    strain_potential = 1e3 * np.sum(quad_fn(strain_energy_density))

    # Mass density in parent config.
    mass_density = mat_density * np.abs(ref_jac_dets)

    # Positions in deformed config.
    positions = deformation_fn(def_ctrl)

    # Work density done by gravity.
    grav_energy_density = positions[:,1] * gravity * mass_density

    # Total work done by gravity integrated over parent config.
    gravity_potential = 1e-7 * np.sum(quad_fn(grav_energy_density))

    # Returning total energy here.
    return strain_potential + gravity_potential

  return free_energy
