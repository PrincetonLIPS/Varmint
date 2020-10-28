import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import logging

from vmap_utils import *

def generate_lagrangian(quad, ref_shape, mat):
  ''' Generate a Lagrangian function for a deformation problem.

  Returns a function that uses quadrature to evaluate the Lagrangian associated
  with a particular deformation and generalized velocity, with the given
  reference configuration and b-spline parameters.

  I'm taking D \in {2,3} to be the number of spatial dimensions.

  Parameters:
  -----------
   - quad : An object that implements the appropriately-dimensioned Quadrature
            object.

   - ref_shape: A representation of the reference object as a Shape2D or Shape3D
                object.

   - mat: A Material object.

  Returns:
  --------
   FIXME: update for the Shape setup...

   Returns a function with the signature:
    (ndarray[J x K x L x 3], ndarray[J x K x L x 3]) -> float
   This function takes a set of deformation control points as generalized
   coordinates, as well as the associated generalized velocities.  It computes
   the associated Lagrangian as computed by quadrature and returns its value.
  '''

  uu = quad.locs

  # Precompute useful reference quantities.
  ref_jacs     = ref_shape.jacobian(uu)
  ref_jac_invs = map(vmap_inv, ref_jacs)
  ref_jac_dets = map(vmap_det, ref_jacs)

  def lagrangian(q, qdot):

    # Maybe should be doing fancier flattening things here.
    def_jacs = q
    def_vels = qdot

    # Compute the deformation gradients.
    defgrads = map(vmap_dot, def_jacs, ref_jacs)

    # Determinant of the reference configuration.
    ref_jac_dets = map(vmap_det, def_jacs)

    # Compute the energy density.
    # FIXME: does map work with methods?
    energy_density = map(
      np.multiply,
      map(
        mat.energy, defgrads),
      np.abs(ref_jac_dets)
    )

    strain_energy = reduce(np.add, map(quad.compute, energy_density))

    # Compute the mass density assuming uniform reference density.
    mass_density = map(lambda rjd: rjd * material.density, np.abs(ref_jac_dets))

    # TODO: compute gravitational potential

    # TODO: compute mass matrix

    # TODO: compute kinetic energy

    # return lagrangian


if __name__ == '__main__':
  import shape3d
  import materials

  from constitutive import NeoHookean
  from quadrature import Quad3D_TensorGaussLegendre

  mat   = NeoHookean(materials.NinjaFlex)
  quad  = Quad3D_TensorGaussLegendre(10)
  shape = shape3d.test_shape1()

  generate_lagrangian(quad, shape, mat)
