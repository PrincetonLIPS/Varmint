import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import numpy            as onp
import logging

from .vmap_utils import *

def generate_free_energy_structured(shape):
  ''' For when all the patches have the same control point shapes. '''

  shape_unflatten = shape.get_unflatten_fn()
  def unflatten(q, qdot, displacement):
    ctrl, vels = shape_unflatten(q, qdot, displacement)
    return np.array(ctrl), np.array(vels)

  # FIXME: Sketchy to just use one patch.
  jacobian_u_fn  = shape.patches[0].get_jacobian_u_fn()
  energy_fn      = shape.patches[0].get_energy_fn()
  quad_fn        = shape.patches[0].get_quad_fn()
  deformation_fn = shape.patches[0].get_deformation_fn()
  jacobian_ctrl_fn = shape.patches[0].get_jacobian_ctrl_fn()

  #grad_energy_fn = jax.jit(jax.grad(energy_fn))
  grad_energy_fn = jax.grad(energy_fn)

  mat_densities = np.array([p.material.density for p in shape.patches])

  gravity = 981.0 # cm/s^2

  #@jax.jit
  def free_energy(q, qdot, ref_ctrl, displacement):
    def_ctrl, def_vels = unflatten(q, qdot, displacement)

    # Jacobian of reference config wrt parent config.
    ref_jacs = jax.vmap(
      jacobian_u_fn,
      in_axes=(0,)
    )(
      ref_ctrl
    )

    # Jacobian of deformed config wrt parent config.
    def_jacs = jax.vmap(
      jacobian_u_fn,
      in_axes=(0,)
    )(
      def_ctrl
    )

    #defgrads = jax.vmap(jax.vmap(np.dot, in_axes=(0,0,)), in_axes=(0,0,))(
    #  def_jacs, ref_jac_invs,
    #)

    # Deformation gradients. def_jacs @ ref_jacs_inv computed via a linear solve.
    defgrads = jax.vmap(
      jax.vmap(
        lambda A, B: npla.solve(B.T, A.T).T,
        in_axes=(0,0),
      ),
      in_axes=(0,0),
    )(
      def_jacs,
      ref_jacs,
    )

    # Jacobian determinants of reference config wrt parent.
    ref_jac_dets = jax.vmap(
      jax.vmap(
        npla.det,
        in_axes=(0,)
      ), in_axes=(0,)
    )(
      ref_jacs,
    )

    # Strain energy density wrt to parent config.
    strain_energy_density = jax.vmap(
      jax.vmap(
        energy_fn,
        in_axes=(0,)
      ),
      in_axes=(0,)
    )(
      defgrads
    ) * np.abs(ref_jac_dets)

    # Total potential energy via integrating over parent config.
    strain_potential = 1e3 * np.sum(
      jax.vmap(
        quad_fn,
        in_axes=(0,)
      )(
      strain_energy_density
      )
    )

    # Mass density in parent config.
    mass_density = mat_densities[:,np.newaxis] * np.abs(ref_jac_dets)

    # Positions in deformed config.
    positions = jax.vmap(deformation_fn, in_axes=(0,))(def_ctrl)

    # Work density done by gravity.
    grav_energy_density = positions[:,:,1] * gravity * mass_density

    # Total work done by gravity integrated over parent config.
    gravity_potential = 1e-7 * np.sum(
      jax.vmap(
        quad_fn,
        in_axes=(0,)
      )(
        grav_energy_density
      )
    )

    # Returning total energy here.
    fenergy = strain_potential + gravity_potential

    return fenergy

  return free_energy
