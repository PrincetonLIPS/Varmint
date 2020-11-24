import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt
import quadpy

from .exceptions import LabelError
from .bsplines import (
  mesh,
  bspline2d,
  bspline2d_derivs,
  bspline2d_derivs_ctrl,
)

# TODO: this will own its quadrature points.
# TODO: will also have its own material.

class Patch2D:
  ''' Class for individual patches in two dimensions.

  A patch corresponds to a two-dimensional bspline with tensor product basis
  functions.  They are individual maps from [0,1]^2 to R^2, parameterized
  by control points.
  '''

  def __init__(
      self,
      xknots,
      yknots,
      spline_deg,
      material,
      quad_deg,
      labels=None,
      fixed=None,
  ):
    ''' Constructor for two dimensional patch.

    Parameters:
    -----------
     - xknots: A length-M one-dimensional array of bspline knots for the x
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - yknots: A length-N one-dimensional array of bspline knots for the y
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - spline_deg: The degree of the bspline.

    # FIXME

     - labels: An optional M x N array of strings that allow constraints to be
               specified across patches. Labels are used to specify coincidence
               constraints and to specify fixed locations in space.  Note that
               they will get dimension-specific extensions appended to them.

     - fixed: An optional dictionary that maps labels to 2d locations, as
              appropriate.
    '''
    self.xknots     = xknots
    self.yknots     = yknots
    self.spline_deg = spline_deg
    self.fixed      = fixed
    self.material   = material
    self.quad_deg   = quad_deg

    # Determine the number of control points.
    num_xknots = self.xknots.shape[0]
    num_yknots = self.yknots.shape[0]

    self.num_xctrl  = num_xknots - self.spline_deg - 1
    self.num_yctrl  = num_yknots - self.spline_deg - 1

    if labels is None:
      # Generate an empty label matrix.
      self.pretty_labels = onp.zeros((self.num_xctrl, self.num_yctrl),
                                     dtype='<U256')
      self.labels = onp.zeros((self.num_xctrl, self.num_yctrl,2),
                              dtype='<U256')

    else:
      # Expand the given label matrix to include dimensions.
      if labels.shape != (self.num_xctrl,self.num_yctrl):
        raise DimensionError('The labels must have shape %d x %d.' \
                             % (self.num_xctrl, self.num_yctrl))
      self.pretty_labels = labels
      self.labels = onp.tile(labels[:,:,np.newaxis], (1, 1, 2,))
      rows, cols = onp.nonzero(labels)
      self.labels[rows,cols,:] = onp.core.defchararray.add(
        self.labels[rows,cols,:],
        onp.array(['_x', '_y']),
      )

    # Expand the fixed labels with dimensions.
    self.fixed = {}
    if fixed:
      for label, value in fixed.items():
        for ii, dim in enumerate(['_x', '_y']):
          newlabel = label + dim
          if not self.has_label(newlabel):
            raise LabelError('Label %s not found' % (newlabel))
          self.fixed[label + dim] = value[ii]
      self.pretty_fixed = fixed
    else:
      self.pretty_fixed = {}

    self.compute_quad_points()

  def compute_quad_points(self):

    # Each knot span has its own quadrature.
    uniq_xknots = onp.unique(self.xknots)
    uniq_yknots = onp.unique(self.yknots)

    xwidths = onp.diff(uniq_xknots)
    ywidths = onp.diff(uniq_yknots)

    # We need the span volumes for performing integration later.
    self.span_volumes = xwidths[:,np.newaxis] * ywidths[np.newaxis,:]

    # Ask quadpy for a quadrature scheme.
    scheme = quadpy.c2.get_good_scheme(self.quad_deg)

    # Change the domain from (-1,1)^2 to (0,1)^2
    points = scheme.points.T/2 + 0.5

    # Repeat the quadrature points for each knot span, scaled appropriately.
    offset_mesh = mesh(uniq_xknots[:-1], uniq_yknots[:-1])
    width_mesh  = mesh(xwidths, ywidths)

    self.points = np.reshape(points[np.newaxis,np.newaxis,:,:] \
                             * width_mesh[:,:,np.newaxis,:] \
                             + offset_mesh[:,:,np.newaxis,:],
                             (-1, 2))

    # FIXME: Why don't I have to divide this by 4 to accommodate the change in
    # interval?
    self.weights = np.reshape(scheme.weights, (1, 1, -1))

  def num_quad_pts(self):
    return self.points.shape[0]

  def get_deformation_fn(self):
    ''' Get a function that produces deformations

    Takes in control points and returns a deformation for each quad point.

    This is assumed to be in cm.
    '''
    def deformation_fn(ctrl):
      return bspline2d(
        self.points,
        ctrl,
        self.xknots,
        self.yknots,
        self.spline_deg
      )
    return deformation_fn

  def get_jacobian_u_fn(self):
    ''' Take control points, return 2x2 Jacobians wrt quad points. '''
    def jacobian_u_fn(ctrl):
      return bspline2d_derivs(
        self.points,
        ctrl,
        self.xknots,
        self.yknots,
        self.spline_deg
      )
    return jacobian_u_fn

  def get_jacobian_ctrl_fn(self):
    ''' Take control points, return Jacobian wrt control points. '''
    def jacobian_ctrl_fn(ctrl):
      return bspline2d_derivs_ctrl(
        self.points,
        ctrl,
        self.xknots,
        self.yknots,
        self.spline_deg,
      )
    return jacobian_ctrl_fn

  def get_energy_fn(self):
    ''' Get the energy density function associated with the material model.

    The various material properties are in GPa, and Pa = N/m^3 so GPa is
    billons of Newtons per cubic meter = GN/m^3.  To get a sense of how this
    varies, it is roughly quadratic in the log of the scale of deformation
    gradient.
    '''
    return self.material.get_energy_fn()

  def get_quad_fn(self):
    def quad_fn(ordinates):

      # Need to get into kind of a fancy shape to both broadcast correctly
      # and to be able to sum in two stages with quadrature weights.
      ords = np.reshape(ordinates,
                        (*self.span_volumes.shape, -1, *ordinates.shape[1:]))

      # The transpose makes it possible to sum over additional dimensions.
      return np.sum(np.sum(self.weights.T * ords.T, axis=-3) \
                    * self.span_volumes.T, axis=(-1,-2)).T

    return quad_fn

  def get_ctrl_shape(self):
    return self.num_xctrl, self.num_yctrl, 2

  def has_label(self, label):
    ''' Predicate for verifying that one of the control points has a label.

    Parameters:
    -----------
     - label: A string representing the label.

    Returns:
    --------
     True if one of the labels matches the string, otherwise False.
    '''
    return onp.any(self.labels == label)

  def label2idx(self, label):
    ''' Identify the control point associated with the label, throwing an error
        if the label is not present.

    Params:
    -------
     - label: A string representing the label.

    Returns:
    --------
     An index tuple indicating which control point has the label.

    Raises:
    -------
     Throws a LabelError exception if the label is not present or more than one
     of the labels is present.
    '''
    rows, cols, third = onp.nonzero(self.labels == label)
    if rows.shape[0] > 1 or cols.shape[0] > 1 or third.shape[0] > 1:
      raise LabelError('More than one control point has label %s.' % (label))
    elif rows.shape[0] == 0 or cols.shape[0] == 0 or third.shape[0] == 0:
      raise LabelError('No control points have label %s.' % (label))
    return rows[0], cols[0], third[0]

  def get_labels(self):
    ''' Get all the labels at once, along with their indices.

    Returns:
    --------
     A list of (label, indices) tuples. Could be length zero.
    '''
    indices = onp.nonzero(self.labels != '')
    labels  = self.labels[indices]
    return list(zip(labels, onp.column_stack(indices)))

  def get_fixed(self):
    ''' Get all fixed control points. '''
    return self.fixed
