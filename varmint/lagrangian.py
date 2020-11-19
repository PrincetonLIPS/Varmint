import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import logging

from vmap_utils import *

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
      defgrads = vmap_dot(def_jacs, ref_jac_invs[ii])

      # Get the strain energy density.
      strain_energy_density = P[patch]['energy_fn'](defgrads) \
        * np.abs(ref_jac_dets[ii])

      # Sum over the quadrature points.
      strain_potential = P[patch]['quad_fn'](strain_energy_density)

      # I'm going to assume each patch is uniform in the reference configuration.
      # Each patch might have a different density because it's a diff material.
      # This doesn't actually multiply because the ref_jac_dets are a list.
      mass_density = patch.material.density * np.abs(ref_jac_dets[ii])

      # Get the actual locations.
      positions = P[patch]['deformation_fn'](def_ctrl)

      # Compute a gravitational energy density.
      # This won't work like this because they're arrays.
      # We're doing this patchwise.
      grav_energy_density = positions[:,1] * gravity * mass_density
      gravitational_potential = P[patch]['quad_fn'](grav_energy_density)

      # Jacobian of the deformed configuraton wrt control points.
      ctrl_jacs = P[patch]['jacobian_ctrl_fn'](def_ctrl[ii])

      # Inertia matrix in control point space.
      #ctrl_jacTjac = vmap_tensordot...
      #mass_matrices = ...

      #mass_matrix = P[patch]['quad_fn'](mass_matrices)

      # big tensordot to get kinetic.

if __name__ == '__main__':
  import shape2d
  import materials

  from constitutive import NeoHookean
  from quadrature import Quad3D_TensorGaussLegendre

  mat   = NeoHookean(materials.NinjaFlex)
  quad  = Quad3D_TensorGaussLegendre(10)
  shape = shape3d.test_shape1()

  generate_lagrangian(quad, shape, mat)
