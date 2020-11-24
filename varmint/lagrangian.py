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
    def_ctrl, def_vels = shape.unflatten(q, qdot)

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
      strain_energy_density = 1000 * P[patch]['energy_fn'](defgrads) \
        * np.abs(P[patch]['ref_jac_dets'])

      # Sum over the quadrature points.
      strain_potential = P[patch]['quad_fn'](strain_energy_density)

      # I'm going to assume each patch is uniform in the reference
      # configuration. Each patch might have a different density because it's
      # a diff material. Densities are g / cm^3.
      mass_density = patch.material.density() * np.abs(P[patch]['ref_jac_dets'])

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
      gravitational_potential = P[patch]['quad_fn'](grav_energy_density) / 10**7

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
      ) / 10**7

      tot_strain  += strain_potential
      tot_gravity += gravitational_potential
      tot_kinetic += kinetic_energy

    return tot_kinetic - tot_strain - tot_gravity

  return lagrangian
