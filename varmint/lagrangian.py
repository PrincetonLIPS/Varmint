import jax
import jax.numpy as np
import logging

from exceptions import DimensionError

def generate_lagrangian(quad, ref_ctrl, degree, knots=None):
  ''' Generate a Lagrangian function for a deformation problem.

  Returns a function that uses quadrature to evaluate the Lagrangian associated
  with a particular deformation and generalized velocity, with the given
  reference configuration and b-spline parameters.

  Parameters:
  -----------
   - quad : An object that implements the Quadrature3D interface for the 3D cube
            [0,1]^3.

   - ref_ctrl: An ndarray of size J x K x L x 3 that contains the control points
               of the reference configuration.

   - degree: A non-negative integer indicating the degree of the b-spline.

   - knots: A three-tuple of knot ndarrays for x, y, and z dimensions. The
            lengths of these vectors needs to make sense relative to the
            provided degree and number of control points.  Each vector must be
            non-decreasing and is assumed to go from 0 to 1 inclusive. If it is
            not provided, the default is to generate uniformly distributed with
            open knots at the endpoints.

  Returns:
  --------
   Returns a function with the signature:
    (ndarray[J x K x L x 3], ndarray[J x K x L x 3]) -> float
   This function takes a set of deformation control points as generalized
   coordinates, as well as the associated generalized velocities.  It computes
   the associated Lagrangian as computed by quadrature and returns its value.
  '''

  # Get the shape of the control points.
  xdim, ydim, zdim, _ = ref_ctrl.shape
  logging.debug('xdim=%d ydim=%d zdim=%d' % (xdim, ydim, zdim))

  # Compute the knots, if necessary.
  if knots is None:
    knots = []
    for dd in range(3):
      num_knots = ref_ctrl.shape[dd] + degree + 1
      knots.append(np.hstack([np.zeros(degree),
                              np.linspace(0, 1, num_knots - 2*degree + 1),
                              np.ones(degree)]))
    knots = tuple(knots)
    logging.info('Using default knots.')
  else:
    logging.debug('Using user-provided knots.')
  if xdim != knots[0].shape[0]:
    raise DimensionError(
      'Unexpected number of x-dim knots vs. control points.'
    )
  if ydim != knots[1].shape[0]:
    raise DimensionError(
      'Unexpected number of y-dim knots vs. control points.'
    )
  if zdim != knots[2].shape[0]:
    raise DimensionError(
      'Unexpected number of z-dim knots vs. control points.'
    )

  # Get the quadrature points.
  quadlocs = quad.locs
  if len(quadlocs.shape) != 2:
    raise DimensionError(
      'Quadrature locations should be a 2-dimensional array.'
    )
  if quadlocs.shape[1] != 3:
    raise DimensionError(
      'Quadrature locations should be in 3D.'
    )

  # Generate the bspline basis functions.
