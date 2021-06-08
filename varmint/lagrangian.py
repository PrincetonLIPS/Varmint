import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import numpy            as onp
import logging

from .vmap_utils import *

def generate_lagrangian(shape, ref_ctrl):

  # Precompute useful per-patch objects that never change.
  P = {}
  for ii, patch in enumerate(shape.patches):

    # TODO: do all this via cached_property
    P[patch] = {

      # Precompute useful functions.
      'deformation_fn':   patch.get_deformation_fn(),
      'jacobian_u_fn' :   patch.get_jacobian_u_fn(),
      'jacobian_ctrl_fn': patch.get_jacobian_ctrl_fn(),
      'energy_fn':        patch.get_energy_fn(),
      'quad_fn':          patch.get_quad_fn(),
    }

    # Precompute useful properties of the reference configuration.
    P[patch]['ref_jacs']     = P[patch]['jacobian_u_fn'](ref_ctrl[ii])
    P[patch]['ref_jac_dets'] = vmap_det(P[patch]['ref_jacs'])
    P[patch]['ref_jac_invs'] = vmap_inv(P[patch]['ref_jacs'])

  unflatten = shape.get_unflatten_fn()

  #@jax.jit
  def lagrangian(q, qdot):
    ''' Compute the Lagrangian in generalized coordinates.

    Here generalized coordinates are the locations of the control points, after
    accounting for simple coincidence and fixed-location constraints.

    Parameters:
    -----------
     - q: one-dimensional ndarray of generalized coordinates

     - qdot: one-dimensional ndarray of generalized velocities

    Returns:
    --------
    Returns the one-dimensional value of the Lagrangian, i.e., the kinetic
    energy minus the potential eneergy.
    '''

    # Use the shape object to unflatten both the coordinates and the velocities.
    # This gives a list of control points and a list of velocities.
    def_ctrl, def_vels = unflatten(q, qdot)

    gravity = 981.0 # cm/s^2

    # It'd be nice to have a cleverer way to do this with map and friends,
    # but because the patches can have different numbers of control points,
    # we just have to live with doing it in a Python loop and live with the
    # annoying unroll. :-(
    tot_strain  = 0.0
    tot_gravity = 0.0
    tot_kinetic = 0.0
    for ii, patch in enumerate(shape.patches):

      # Get the Jacobians for the deformed configuration.
      def_jacs = P[patch]['jacobian_u_fn'](def_ctrl[ii])

      # Compute the deformation gradients.
      defgrads = vmap_dot(def_jacs, P[patch]['ref_jac_invs'])

      # Get the strain energy density.
      # Units are GPa = 10^9 J / m^3 in the reference configuration.
      # Convert to J / cm^3 by multiplying by 10^3.
      strain_energy_density = P[patch]['energy_fn'](defgrads) \
        * np.abs(P[patch]['ref_jac_dets'])

      # Sum over the quadrature points.
      strain_potential = P[patch]['quad_fn'](strain_energy_density)

      # I'm going to assume each patch is uniform in the reference
      # configuration. Each patch might have a different density because it's
      # a diff material. Densities are g / cm^3.
      mass_density = patch.material.density * np.abs(P[patch]['ref_jac_dets'])

      # Get the actual locations. These are in cm.
      positions = P[patch]['deformation_fn'](def_ctrl[ii])

      # Compute a gravitational energy density, in J / cm^3.
      # position = cm
      # gravity = cm / s^2
      # mass_density = g / cm^3
      # result = cm * (cm / s^2) * g / cm^3 = cm * (g * cm / s^2) / cm^3
      # This is ergs per cubic centimeter, i.e., 10^-7 J / cm ^3, so we need to
      # divide by 10^7 to put it into the same units as strain potential.
      # Should we do this before or after the quadrature?
      grav_energy_density = positions[:,1] * gravity * mass_density
      gravitational_potential = P[patch]['quad_fn'](grav_energy_density)

      # Jacobian of the deformed configuraton wrt control points.
      ctrl_jacs = P[patch]['jacobian_ctrl_fn'](def_ctrl[ii])

      # Mass matrix in control point space.
      # Control points are in cm and density is in g/cm^3 so this should work
      # out to get a mass matrix with units of grams.
      ctrl_jacTjac  = vmap_tensordot(ctrl_jacs, ctrl_jacs, (0,0))
      mass_matrices = (ctrl_jacTjac.T * mass_density.T).T
      mass_matrix   = P[patch]['quad_fn'](mass_matrices)

      # Compute the inertia with this matrix.
      # This should be g * cm/s * cm/s = g * cm^2 / s^2 = erg.
      # So also divide by 10^7 to get Joules.
      kinetic_energy = 0.5*np.tensordot(
        np.tensordot(
          mass_matrix,
          def_vels[ii],
          ((3,4,5), (0,1,2)),
        ),
        def_vels[ii],
        ((0,1,2), (0,1,2)),
      ) # * 1e-7

      tot_strain  = tot_strain + strain_potential * 1e3
      tot_gravity = tot_gravity + gravitational_potential * 1e-7
      tot_kinetic = tot_kinetic + kinetic_energy * 1e-7

    return tot_kinetic - tot_strain - tot_gravity

  return lagrangian

def generate_lagrangian_structured(shape):
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
  def lagrangian(q, qdot, ref_ctrl, displacement):
    def_ctrl, def_vels = unflatten(q, qdot, displacement)

    ref_jacs = jax.vmap(
      jacobian_u_fn,
      in_axes=(0,)
    )(
      ref_ctrl
    )

    def_jacs = jax.vmap(
      jacobian_u_fn,
      in_axes=(0,)
    )(
      def_ctrl
    )

    #defgrads = jax.vmap(jax.vmap(np.dot, in_axes=(0,0,)), in_axes=(0,0,))(
    #  def_jacs, ref_jac_invs,
    #)
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

    ref_jac_dets = jax.vmap(
      jax.vmap(
        npla.det,
        in_axes=(0,)
      ), in_axes=(0,)
    )(
      ref_jacs,
    )

    strain_energy_density = jax.vmap(
      jax.vmap(
        energy_fn,
        in_axes=(0,)
      ),
      in_axes=(0,)
    )(
      defgrads
    ) * np.abs(ref_jac_dets)

    strain_potential = 1e3 * np.sum(
      jax.vmap(
        quad_fn,
        in_axes=(0,)
      )(
      strain_energy_density
      )
    )

    mass_density = mat_densities[:,np.newaxis] * np.abs(ref_jac_dets)

    positions = jax.vmap(deformation_fn, in_axes=(0,))(def_ctrl)

    grav_energy_density = positions[:,:,1] * gravity * mass_density

    gravity_potential = 1e-7 * np.sum(
      jax.vmap(
        quad_fn,
        in_axes=(0,)
      )(
        grav_energy_density
      )
    )

    ctrl_jacs = jax.vmap(
      jacobian_ctrl_fn,
      in_axes=(0,)
    )(
      def_ctrl
    )

    ctrl_jacTjac = jax.vmap(
      jax.vmap(
        np.tensordot,
        in_axes=(0,0,None)
      ),
      in_axes=(0,0,None)
    )(
      ctrl_jacs,
      ctrl_jacs,
      (0,0,)
    )

    mass_matrices = (ctrl_jacTjac.T * mass_density.T).T

    mass_matrix = jax.vmap(
      quad_fn,
      in_axes=(0,)
    )(
      mass_matrices
    )

    kinetic_energy = 1e-7 * np.sum(
      jax.vmap(
        lambda mm, vv: 0.5 * np.tensordot(
          np.tensordot(
            mm,
            vv,
            ((3,4,5), (0,1,2))
          ),
          vv,
          ((0,1,2), (0,1,2))
        ),
        in_axes=(0,0,)
      )(
        mass_matrix,
        def_vels
      )
    )

    lag = kinetic_energy - gravity_potential - strain_potential

    return lag

  return lagrangian

def generate_patch_lagrangian(patch):
  ''' For when all the patches have the same control point shapes. '''

  jacobian_u_fn    = patch.get_jacobian_u_fn()
  energy_fn        = patch.get_energy_fn()
  quad_fn          = patch.get_quad_fn()
  deformation_fn   = patch.get_deformation_fn()
  jacobian_ctrl_fn = patch.get_jacobian_ctrl_fn()
  vmap_energy_fn   = jax.vmap(energy_fn, in_axes=(0,))
  jac_dets_fn      = jax.vmap(npla.det, in_axes=(0,))

  defgrads_fn = jax.vmap(
    lambda A, B: npla.solve(B.T, A.T).T,
    in_axes=(0,0),
  )

  kinetic_energy_fn = \
    lambda mm, vv: 0.5 * np.tensordot(
      np.tensordot(mm, vv, ((3,4,5), (0,1,2))), vv, ((0,1,2), (0,1,2))
    )

  mat_density = patch.material.density
  gravity = 981.0 # cm/s^2

  def lagrangian(def_ctrl, def_vels, ref_ctrl):
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

    ctrl_jacs = jacobian_ctrl_fn(def_ctrl)
    ctrl_jacTjac = jax.vmap(np.tensordot, in_axes=(0,0,None))(ctrl_jacs, ctrl_jacs, (0,0,))

    mass_matrices = (ctrl_jacTjac.T * mass_density.T).T
    mass_matrix = quad_fn(mass_matrices)

    kinetic_energy = 1e-7 * np.sum(kinetic_energy_fn(mass_matrix, def_vels))

    # Compute and return lagrangian.
    return kinetic_energy - gravity_potential - strain_potential

  return lagrangian
