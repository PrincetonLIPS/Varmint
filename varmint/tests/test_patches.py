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
    ctrl   = bsplines.mesh(np.arange(10.), np.arange(5.))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])

    self.patch = Patch2D(ctrl, xknots, yknots, deg)

  def test_shape(self):
    self.assertEqual(self.patch.ctrl.shape, (10,5,2))

  def test_values(self):
    self.assertEqual(self.patch.ctrl[0,0,0], 0)
    self.assertEqual(self.patch.ctrl[-1,-1,1], 4)

  def test_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_deg(self):
    self.assertEqual(self.patch.deg, 4)

  def test_no_labels(self):
    self.assertTrue(np.all(self.patch.labels == ''))
    self.assertFalse(self.patch.has_label('foo'))

  def test_label_errors(self):
    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('foo')

  def test_fixed_error(self):
    with self.assertRaises(LabelError):
      self.patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('FIXED')

  def test_get_labels(self):
    labels, rows, cols = self.patch.get_labels()
    self.assertEqual(len(labels), 0)
    self.assertEqual(len(rows), 0)
    self.assertEqual(len(cols), 0)

  def test_flatten(self):
    values, labels = self.patch.flatten()
    self.assertEqual(values.shape[0], 50)
    self.assertEqual(values.shape[1], 2)
    self.assertEqual(labels.shape[0], 50)

    # Test x-major reshaping.
    nptest.assert_array_equal(values[5,:], np.array([1.0, 0.0]))

  def test_unflatten_basic(self):
    old_ctrl = np.array(self.patch.ctrl)

    ctrl, labels = self.patch.flatten()
    self.patch.unflatten(ctrl)

    nptest.assert_array_equal(old_ctrl, self.patch.ctrl)

  def test_unflatten_modify(self):
    old_ctrl = np.array(self.patch.ctrl)

    npr.seed(1)

    ctrl, _  = self.patch.flatten()
    new_ctrl = np.array(npr.randn(*ctrl.shape), dtype=np.float32)

    self.patch.unflatten(new_ctrl)

    test_ctrl, _ = self.patch.flatten()

    nptest.assert_array_equal(new_ctrl, test_ctrl)

class Test_Patch2D_Labels(ut.TestCase):
  ''' Test basic label functionality. '''

  def setUp(self):
    deg    = 2
    ctrl   = bsplines.mesh(np.arange(10.), np.arange(4.))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])

    self.patch = Patch2D(ctrl, xknots, yknots, deg)

    self.patch.labels[0,0] = '00'
    self.patch.labels[-1,:] = ['A', 'B', 'C', 'D']

  def test_shape(self):
    self.assertEqual(self.patch.ctrl.shape, (10,4,2))

  def test_values(self):
    self.assertEqual(self.patch.ctrl[0,0,0], 0)
    self.assertEqual(self.patch.ctrl[-1,-1,1], 3)

  def test_knots(self):
    self.assertEqual(self.patch.xknots[0], 0)
    self.assertEqual(self.patch.xknots[-1], 1)
    self.assertEqual(self.patch.yknots[0], 0)
    self.assertEqual(self.patch.yknots[-1], 1)

  def test_deg(self):
    self.assertEqual(self.patch.deg, 2)

  def test_absent_labels(self):
    self.assertFalse(self.patch.has_label('foo'))

    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('foo')

  def test_has_labels(self):
    self.assertFalse(np.all(self.patch.labels == ''))

  def test_present_label(self):
    self.assertTrue(self.patch.has_label('D'))
    self.assertEqual(self.patch.labels[-1,0], 'A')
    self.assertEqual(self.patch.label2idx('C'), (9,2))

    nptest.assert_array_equal(self.patch.label2ctrl('B'),
                              np.array([9.0, 1.0]))

  def test_fixed_error(self):
    with self.assertRaises(LabelError):
      self.patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('FIXED')

  def test_get_labels(self):
    labels, rows, cols = self.patch.get_labels()
    self.assertEqual(len(labels), 5)
    self.assertEqual(len(rows), 5)
    self.assertEqual(len(cols), 5)

    labdict = dict(zip(labels, zip(rows, cols)))
    self.assertEqual(labdict['00'], (0, 0))
    self.assertEqual(labdict['A'], (9, 0))
    self.assertEqual(labdict['B'], (9, 1))
    self.assertEqual(labdict['C'], (9, 2))
    self.assertEqual(labdict['D'], (9, 3))

  def test_flatten(self):
    values, labels = self.patch.flatten()
    self.assertEqual(values.shape[0], 40)
    self.assertEqual(values.shape[1], 2)
    self.assertEqual(labels.shape[0], 40)

    # Test x-major reshaping.
    nptest.assert_array_equal(values[4,:], np.array([1.0, 0.0]))

  def test_unflatten_modify(self):
    old_ctrl = np.array(self.patch.ctrl)

    npr.seed(1)

    ctrl, _  = self.patch.flatten()
    new_ctrl = np.array(npr.randn(*ctrl.shape), dtype=np.float32)

    self.patch.unflatten(new_ctrl)

    test_ctrl, _ = self.patch.flatten()

    nptest.assert_array_equal(new_ctrl, test_ctrl)

class Test_Patch2D_Fixed(ut.TestCase):
  ''' Test special FIXED label functionality. '''

  def setUp(self):
    deg    = 2
    ctrl   = bsplines.mesh(np.arange(10.), np.arange(4.))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])

    self.patch = Patch2D(ctrl, xknots, yknots, deg)

    self.patch.labels[0,0] = 'FIXED'
    self.patch.labels[1,2] = 'FIXED'
    self.patch.labels[2,3] = 'FIXED'

  def test_label_errors(self):
    with self.assertRaises(LabelError):
      self.patch.label2idx('foo')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('foo')

  def test_fixed_error(self):
    with self.assertRaises(LabelError):
      self.patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      self.patch.label2ctrl('FIXED')

  def test_is_fixed(self):
    self.assertTrue(self.patch.is_fixed(0,0))
    self.assertFalse(self.patch.is_fixed(1,1))

  def test_fixed(self):
    rows, cols = self.patch.get_fixed()
    self.assertEqual(len(rows), 3)
    self.assertEqual(len(cols), 3)
    nptest.assert_array_equal(rows, np.array([0, 1, 2]))
    nptest.assert_array_equal(cols, np.array([0, 2, 3]))

  def test_get_labels(self):
    labels, rows, cols = self.patch.get_labels()
    self.assertEqual(len(labels), 0)
    self.assertEqual(len(rows), 0)
    self.assertEqual(len(cols), 0)

  def test_flatten(self):
    values, labels = self.patch.flatten()
    self.assertEqual(values.shape[0], 37)
    self.assertEqual(values.shape[1], 2)
    self.assertEqual(labels.shape[0], 37)

    # Test x-major reshaping.
    nptest.assert_array_equal(values[4,:], np.array([1.0, 1.0]))

  def test_unflatten_modify(self):
    old_ctrl = np.array(self.patch.ctrl)

    npr.seed(1)

    ctrl, _  = self.patch.flatten()
    new_ctrl = np.array(npr.randn(*ctrl.shape), dtype=np.float32)

    self.patch.unflatten(new_ctrl)

    test_ctrl, _ = self.patch.flatten()

    nptest.assert_array_equal(new_ctrl, test_ctrl)
