import jax
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest
import unittest      as ut

import bsplines

from patch      import Patch2D
from exceptions import LabelError

class Test_Patch2D_NoLabels(ut.TestCase):
  ''' Test basic functionality with no labels. '''

  # TODO: Need to add back in reasonable tests.

  def setUp(self):
    deg    = 4
    xknots = bsplines.default_knots(deg, 10)
    yknots = bsplines.default_knots(deg, 5)

    self.patch = Patch2D(xknots, yknots, deg)

  def test_deg(self):
    self.assertEqual(self.patch.deg, 4)

  def test_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_no_labels(self):
    self.assertTrue(np.all(self.patch.labels == ''))
    self.assertFalse(self.patch.has_label('foo'))

  def test_label_errors(self):
    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

  def test_get_labels(self):
    labels, rows, cols = self.patch.get_labels()
    self.assertEqual(len(labels), 0)
    self.assertEqual(len(rows), 0)
    self.assertEqual(len(cols), 0)

class Test_Patch2D_Labels(ut.TestCase):
  ''' Test basic label functionality. '''
  # TODO: Had to remove all these when I changed the interface. :-(
  pass


class Test_Patch2D_Fixed(ut.TestCase):
  ''' Test special FIXED label functionality. '''

  # TODO: Had to remove all these when I changed the interface. :-(
  pass
