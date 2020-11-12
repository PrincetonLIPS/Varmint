import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import logging

from vmap_utils import *

# How to do this in a sensible functional way?
#
# q is generalized coordinates, which in this case means the post-constraint
# coordinates.
#
# qdot has the same dimensions as q.
#
# This function computes three energetic quantities: strain energy,
# gravitational energy, and kinetic energy.
#
# The strain energy needs to get the deformation gradient at each quadrature
# point in all patches.  This needs to be done by computing the Jacobian of
# the deformation for each patch with respect to the parent domain. How do
# we do this in a functional way?
#
# One approach: Shape2D interface gives you back a function that you can call
# to get deformations and associated functions, rather than doing it itself.
#

def generate_lagrangian(quad, shape, mat):

  uu = quad.locs

  # What all do we need?
  # - ref jacobian wrt u
  # - ref jac det
  # - ref jac inv
  # - def positions
  # - def jacobian wrt u
  # - def jacobian wrt q

  # compute energies

  # approaches:
  # 1) get lists of all these quantities for each patch, at common parent pts.
  # 2) generalize this by having each have its own quadrature pts
  # 3) generalize this further by solving each with adaptive quadrature.
  #    the downside of this is that the ref stuff may need to be recomputed

  # Since the reference and deformed shape need to be homologous with respect
  # to patches, etc., I think maybe there should be a single global "shape"
  # object.  I don't think it makes sense for them to have different knots, etc.
  # Probably each patch could also have its own quadrature object.
  # It might be useful to think of ways to interface with quadrature in which
  # you hand the quadrature object a function handle and it does its best to
  # compute the result....

  # Is this the right way to think about getting the flat rep?
  # It seems like maybe should uncouple the shape rep from the actual control
  # point values.  But then we also need this to represent constraints?
  # Maybe better like: shape_flatten(ref_shape.init_ctrl) ?
  # or ref_shape.init_flat() ?
  ref_ctrl_flat = ref_shape.flattened()

  # Hand back a non-side-effect-having function that takes a flattened ctrl
  # vector, and quadrature locations, and returns the deformations at each quad
  # point.  This could be a list of lists, or a large-ish tensor because every
  # patch gets its own set of quadrature points.
  ref_def_fn = ref_shape.get_ref_def_fn()

  # Get the associated Jacobian function wrt to u.
  ref_jac_u_fn = ref_shape.get_ref_jac_u_fn()

  # Precompute useful reference quantities.
  ref_jacs     = ref_jac_u_fn(ref_ctrl_flat, uu)
  ref_jac_invs = map(vmap_inv, ref_jacs)
  ref_jac_dets = map(vmap_det, ref_jacs)

  def lagrangian(q, qdot):

    # Compute the jacobian of the deformed config.
    # This function needs to come from somewhere.
    def_jacs = def_jac_u_fn(def_ctrl_flat, uu)

    # Now get the deformation gradients.
    # Probably need to map.
    defgrads = vmap_dot(def_jacs, ref_jac_invs)

    # Evaluate energy densities.
    # This should also be of the "created function" variety.
    energy_densities = mat.energy(defgrads) * np.abs(ref_jac_dets)

    # Now integrate these energy densities.
    # Could loop over patches here and hand off functions to quadrature.
    # I think right now it's better not to overthink the quadrature.
    # Can explore adaptive quadrature down the road.



if __name__ == '__main__':
  import shape3d
  import materials

  from constitutive import NeoHookean
  from quadrature import Quad3D_TensorGaussLegendre

  mat   = NeoHookean(materials.NinjaFlex)
  quad  = Quad3D_TensorGaussLegendre(10)
  shape = shape3d.test_shape1()

  generate_lagrangian(quad, shape, mat)
