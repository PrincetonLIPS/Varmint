import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt

from exceptions import LabelError

import bsplines

# TODO: how to fix control points in space?

class Patch2D:
  ''' Class for individual patches in two dimensions.

  A patch corresponds to a two-dimensional bspline with tensor product basis
  functions.  They are individual maps from [0,1]^2 to R^2, parameterized
  by control points.
  '''

  def __init__(self, xknots, yknots, deg, labels=None):
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
               specified across patches. In general, labels are used to specify
               coincidence constraints.  The exception is the special label
               'FIXED' which is only used for fixing a control point in space.
    '''
    self.xknots = xknots
    self.yknots = yknots
    self.deg    = deg

    M = self.xknots.shape[0]
    N = self.yknots.shape[1]
    
    if labels is None:
      self.labels = onp.zeros((M,N), dtype='<U256')
    else:
      if labels.shape != (M,N):
        raise DimensionError('The labels must have shape %d x %d.' % (M,N))

  def has_label(self, label):
    ''' Predicate for verifying that one of the control points has a label.

    Parameters:
    -----------
     - label: A string representing the label.

    Returns:
    --------
     True if one of the labels matches the string, otherwise False.
    '''
    if label == 'FIXED':
      raise LabelError('The FIXED label is reserved for "fixed" constraints.')
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
    if label == 'FIXED':
      raise LabelError('The FIXED label is reserved for "fixed" constraints.')
    rows, cols = onp.nonzero(self.labels == label)
    if rows.shape[0] > 1 or cols.shape[0] > 1:
      raise LabelError('More than one control point has label %s.' % (label))
    elif rows.shape[0] == 0 or cols.shape[0] == 0:
      raise LabelError('No control points have label %s.' % (label))
    return rows[0], cols[0]

  def get_labels(self):
    ''' Get all the (non-FIXED) labels at once, along with their indices.

    Returns:
    --------
     (labels, rows, cols) where they are each one-dimensional arrays with length
     corresponding to the total number of labels.  So they could be length zero.
     These can be usefully turned into a dictionary via:

     >> zip(labels, zip(rows, cols))
    '''
    rows, cols = onp.nonzero(onp.logical_and(self.labels != 'FIXED',
                                             self.labels != ''))
    labels = self.labels[rows, cols]
    return labels, rows, cols

  def is_fixed(self, row, col):
    ''' A predicate to determine whether a particular control point, specified
    by row and column index is a fixed location.
    '''
    return self.labels[row,col] == 'FIXED'

  def get_fixed(self):
    ''' Get all rows and columns of fixed control points. '''
    return onp.nonzero(self.labels == 'FIXED')
