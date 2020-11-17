import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from exceptions import LabelError

import bsplines

# TODO: this will own its quadrature points.
# TODO: will also have its own material.

class Patch2D:
  ''' Class for individual patches in two dimensions.

  A patch corresponds to a two-dimensional bspline with tensor product basis
  functions.  They are individual maps from [0,1]^2 to R^2, parameterized
  by control points.
  '''

  def __init__(self, xknots, yknots, deg, labels=None, fixed=None):
    ''' Constructor for two dimensional patch.

    Parameters:
    -----------
     - xknots: A length-M one-dimensional array of bspline knots for the x
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - yknots: A length-N one-dimensional array of bspline knots for the y
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - deg: The degree of the bspline.

     - labels: An optional M x N array of strings that allow constraints to be
               specified across patches. Labels are used to specify coincidence
               constraints and to specify fixed locations in spae.

     - fixed: A dictionary that maps labels to 2d locations, as appropriate.
    '''
    self.xknots = xknots
    self.yknots = yknots
    self.deg    = deg
    self.fixed  = fixed

    # Determine the number of control points.
    num_xknots = self.xknots.shape[0]
    num_yknots = self.yknots.shape[0]

    self.num_xctrl  = num_xknots - self.deg - 1
    self.num_yctrl  = num_yknots - self.deg - 1

    if labels is None:
      # Generate an empty label matrix.
      self.labels = onp.zeros((self.num_xctrl,self.num_yctrl,2), dtype='<U256')

    else:
      # Expand the given label matrix to include dimensions.
      if labels.shape != (self.num_xctrl,self.num_yctrl):
        raise DimensionError('The labels must have shape %d x %d.' \
                             % (self.num_xctrl, self.num_yctrl))
      self.labels = onp.tile(labels[:,:,np.newaxis], (1, 1, 2,))
      rows, cols = onp.nonzero(labels)
      self.labels[rows,cols,:] = onp.core.defchararray.add(
        self.labels[rows,cols,:],
        onp.array(['_x', '_y']),
      )

    # TODO: sanity check fixed dictionary.

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
    ''' Get the labels of all fixed control points. '''
    return self.fixed.keys()
