import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import numpy            as onp
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

    gravity = 9.81

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
      strain_energy_density = P[patch]['energy_fn'](defgrads) \
        * np.abs(P[patch]['ref_jac_dets'])

      # Sum over the quadrature points.
      strain_potential = P[patch]['quad_fn'](strain_energy_density)

      # I'm going to assume each patch is uniform in the reference configuration.
      # Each patch might have a different density because it's a diff material.
      # This doesn't actually multiply because the ref_jac_dets are a list.
      mass_density = patch.material.density() * np.abs(P[patch]['ref_jac_dets'])

      # Get the actual locations.
      positions = P[patch]['deformation_fn'](def_ctrl[ii])

      # Compute a gravitational energy density.
      # This won't work like this because they're arrays.
      # We're doing this patchwise.
      grav_energy_density = positions[:,1] * gravity * mass_density
      gravitational_potential = P[patch]['quad_fn'](grav_energy_density)

      # Jacobian of the deformed configuraton wrt control points.
      ctrl_jacs = P[patch]['jacobian_ctrl_fn'](def_ctrl[ii])

      # Mass matrix in control point space.
      ctrl_jacTjac  = vmap_tensordot(ctrl_jacs, ctrl_jacs, (0,0))
      mass_matrices = (ctrl_jacTjac.T * mass_density.T).T
      mass_matrix   = P[patch]['quad_fn'](mass_matrices)

      # Compute the inertia with this matrix.
      kinetic_energy = 0.5*np.tensordot(
        np.tensordot(
          mass_matrix,
          def_vels[ii],
          ((3,4,5), (0,1,2)),
        ),
        def_vels[ii],
        ((0,1,2), (0,1,2)),
      )

      print(kinetic_energy, strain_potential, gravitational_potential)
      tot_strain  += strain_potential
      tot_gravity += gravitational_potential
      tot_kinetic += kinetic_energy

    return tot_kinetic - tot_strain - tot_gravity

  return lagrangian

if __name__ == '__main__':
  import patch2d
  import shape2d
  import materials
  import bsplines

  from constitutive import NeoHookean
  from quadrature import Quad3D_TensorGaussLegendre

  mat   = NeoHookean(materials.NinjaFlex, dims=2)

  # Do this in mm?
  length    = 25
  height    = 5
  num_xctrl = 10
  num_yctrl = 5
  ctrl = bsplines.mesh(np.linspace(0, length, num_xctrl),
                       np.linspace(0, height, num_yctrl))

  # Make the patch.
  spline_deg = 3
  quad_deg = 10
  xknots = bsplines.default_knots(spline_deg, num_xctrl)
  yknots = bsplines.default_knots(spline_deg, num_yctrl)
  labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')
  labels[0,:] = ['A', 'B', 'C', 'D', 'E']
  fixed = {
    'A': ctrl[0,0,:],
    'B': ctrl[0,1,:],
    'C': ctrl[0,2,:],
    'D': ctrl[0,3,:],
    'E': ctrl[0,4,:],
  }
  patch = patch2d.Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    labels=labels,
    fixed=fixed
  )
  shape = shape2d.Shape2D(patch)
  ref_ctrl = [ctrl]

  def_ctrl = [ctrl.copy()]
  def_vels = [np.zeros_like(ctrl)]
  q, qdot  = shape.flatten(def_ctrl, def_vels)

  lagrangian = generate_lagrangian(shape, ref_ctrl)
