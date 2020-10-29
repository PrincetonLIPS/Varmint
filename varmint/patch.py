import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from exceptions import LabelError

import bsplines

class Patch2D:
  ''' Class for individual patches in two dimensions.

  A patch corresponds to a two-dimensional bspline with tensor product basis
  functions.  They are individual maps from [0,1]^2 to R^2.
  '''

  def __init__(self, ctrl, xknots, yknots, deg, labels=None):
    ''' Constructor for two dimensional patch.

    Parameters:
    -----------
     - ctrl: A M x N x 2 ndarray of control points.

     - xknots: A length-M one-dimensional array of bspline knots for the x
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - yknots: A length-N one-dimensional array of bspline knots for the y
               dimension. These are assumed to be in non-decreasing order from
               0.0 to 1.0.

     - deg: The degree of the bspline.

     - labels: An optional M x N array of strings that allow constraints to be
               specified across patches.
    '''
    self.ctrl   = ctrl
    self.xknots = xknots
    self.yknots = yknots
    self.deg    = deg

    if labels is None:
      self.labels = onp.zeros((ctrl.shape[0], ctrl.shape[1]), dtype='<U256')
    else:
      if labels.shape != ctrl.shape[:-1]:
        raise DimensionError('The labels must have shape %d x %d.' % (
          ctrl.shape[0], ctrl.shape[1]))

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
     A tuple (row, col) indicating which control point has the label.

    Raises:
    -------
     Throws a LabelError exception if the label is not present or more than one
     of the labels is present.
    '''
    rows, cols = onp.nonzero(self.labels == label)
    if rows.shape[0] > 1 or cols.shape[0] > 1:
      raise LabelError('More than one control point has label %s.' % (label))
    elif rows.shape[0] == 0 or cols.shape[0] == 0:
      raise LabelError('No control points have label %s.' % (label))
    return rows[0], cols[0]

  def label2ctrl(self, label):
    ''' Retrieve the location of the control point based on the label.

    Params:
    -------
     - label: A string representing the label.

    Returns:
    --------
     A length-two one-dimensional ndarray with the x-y location of the labeled
     control point.

    Raises:
    -------
     Throws a LabelError exception if the label is not present or more than one
     of the labels is present.
    '''
    row, col = self.label2idx(label)
    return self.ctrl[row,col,:]
