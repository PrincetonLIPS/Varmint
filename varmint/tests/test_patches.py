import jax
import jax.numpy     as np
import numpy         as onp
import numpy.random  as npr
import numpy.testing as nptest
import unittest      as ut

import bsplines

from patch      import Patch2D
from exceptions import LabelError

class Test_Patch2D_NoLabels(ut.TestCase):
  ''' Test basic functionality with no labels. '''

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
    labels = self.patch.get_labels()
    self.assertEqual(len(labels), 0)

class Test_Patch2D_Labels(ut.TestCase):
  ''' Test basic label functionality. '''

  def setUp(self):
    deg    = 4
    xknots = bsplines.default_knots(deg, 10)
    yknots = bsplines.default_knots(deg, 5)
    labels = onp.zeros((10,5), dtype='<U256')
    labels[0,0] = 'alfa'
    labels[0,1] = 'bravo'
    labels[1,3] = 'charlie'

    self.patch = Patch2D(xknots, yknots, deg, labels)

  def test_deg(self):
    self.assertEqual(self.patch.deg, 4)

  def test_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_missing_label(self):
    self.assertFalse(self.patch.has_label('foo'))

  def test_label_errors(self):
    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

  def test_get_labels(self):
    labels = dict(self.patch.get_labels())
    self.assertEqual(len(labels), 6)
    nptest.assert_array_equal(labels['alfa_x'], [0, 0, 0])
    nptest.assert_array_equal(labels['alfa_y'], [0, 0, 1])
    nptest.assert_array_equal(labels['bravo_x'], [0, 1, 0])
    nptest.assert_array_equal(labels['bravo_y'], [0, 1, 1])
    nptest.assert_array_equal(labels['charlie_x'], [1, 3, 0])
    nptest.assert_array_equal(labels['charlie_y'], [1, 3, 1])

  def test_has_label(self):
    self.assertTrue(self.patch.has_label('alfa_x'))
    self.assertTrue(self.patch.has_label('alfa_y'))
    self.assertTrue(self.patch.has_label('bravo_x'))
    self.assertTrue(self.patch.has_label('bravo_y'))
    self.assertTrue(self.patch.has_label('charlie_x'))
    self.assertTrue(self.patch.has_label('charlie_y'))

  def test_label_indices(self):
    nptest.assert_array_equal(self.patch.label2idx('alfa_x'), [0, 0, 0])
    nptest.assert_array_equal(self.patch.label2idx('alfa_y'), [0, 0, 1])
    nptest.assert_array_equal(self.patch.label2idx('bravo_x'), [0, 1, 0])
    nptest.assert_array_equal(self.patch.label2idx('bravo_y'), [0, 1, 1])
    nptest.assert_array_equal(self.patch.label2idx('charlie_x'), [1, 3, 0])
    nptest.assert_array_equal(self.patch.label2idx('charlie_y'), [1, 3, 1])

class Test_Patch2D_Fixed(ut.TestCase):
  ''' Test special FIXED label functionality. '''

  # TODO: Had to remove all these when I changed the interface. :-(
  pass
