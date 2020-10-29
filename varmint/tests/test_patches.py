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

  def setUp(self):
    deg    = 4
    ctrl   = bsplines.mesh(np.arange(10), np.arange(5))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])

    self.patch = Patch2D(ctrl, xknots, yknots, deg)

  def test_patch_shape(self):
    self.assertEqual(self.patch.ctrl.shape, (10,5,2))

  def test_patch_values(self):
    self.assertEqual(self.patch.ctrl[0,0,0], 0)
    self.assertEqual(self.patch.ctrl[-1,-1,1], 4)

  def test_patch_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_patch_deg(self):
    self.assertEqual(self.patch.deg, 4)

  def test_patch_no_labels(self):
    self.assertTrue(np.all(self.patch.labels == ''))
    self.assertFalse(self.patch.has_label('foo'))

  def test_patch_label_errors(self):
    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('foo')

  def test_patch_fixed_error(self):
    with self.assertRaises(LabelError):
      self.patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('FIXED')

class Test_Patch2D_Labels(ut.TestCase):
  ''' Test basic label functionality. '''

  def setUp(self):
    deg    = 2
    ctrl   = bsplines.mesh(np.arange(10), np.arange(4))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])

    self.patch  = Patch2D(ctrl, xknots, yknots, deg)

    self.patch.labels[0,0] = '00'
    self.patch.labels[-1,:] = ['A', 'B', 'C', 'D']

  def test_patch_shape(self):
    self.assertEqual(self.patch.ctrl.shape, (10,4,2))

  def test_patch_values(self):
    self.assertEqual(self.patch.ctrl[0,0,0], 0)
    self.assertEqual(self.patch.ctrl[-1,-1,1], 3)

  def test_patch_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_patch_deg(self):
    self.assertEqual(self.patch.deg, 2)

  def test_patch_absent_labels(self):
    self.assertFalse(self.patch.has_label('foo'))

    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('foo')

  def test_patch_has_labels(self):
    self.assertFalse(np.all(self.patch.labels == ''))

  def test_patch_present_label(self):
    self.assertTrue(self.patch.has_label('D'))
    self.assertEqual(self.patch.labels[-1,0], 'A')
    self.assertEqual(self.patch.label2idx('C'), (9,2))

    nptest.assert_array_equal(self.patch.label2ctrl('B'),
                              np.array([9.0, 1.0]))

  def test_patch_fixed_error(self):
    with self.assertRaises(LabelError):
      self.patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('FIXED')
