import jax
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest
import unittest      as ut

import bsplines

from patch      import Patch2D
from exceptions import LabelError

class TestPatch2D(ut.TestCase):

  def test_1(self):
    ''' Test basic functionality with no labels. '''

    deg    = 4
    ctrl   = bsplines.mesh(np.arange(10), np.arange(5))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])
    patch  = Patch2D(ctrl, xknots, yknots, deg)

    self.assertEqual(patch.ctrl.shape, (10,5,2))
    self.assertEqual(patch.ctrl[0,0,0], 0)
    self.assertEqual(patch.ctrl[-1,-1,1], 4)
    self.assertEqual(patch.xknots[0], 0)
    self.assertEqual(patch.xknots[-1], 1)
    self.assertEqual(patch.yknots[0], 0)
    self.assertEqual(patch.yknots[-1], 1)
    self.assertEqual(patch.deg, 4)
    self.assertFalse(patch.has_label('foo'))
    self.assertTrue(np.all(patch.labels == ''))

    with self.assertRaises(LabelError):
      patch.label2idx('foo')

    with self.assertRaises(LabelError):
      patch.label2ctrl('foo')

    with self.assertRaises(LabelError):
      patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      patch.label2ctrl('FIXED')

  def test_2(self):
    ''' Test basic label functionality. '''

    deg    = 2
    ctrl   = bsplines.mesh(np.arange(10), np.arange(4))
    xknots = bsplines.default_knots(deg, ctrl.shape[0])
    yknots = bsplines.default_knots(deg, ctrl.shape[1])
    patch  = Patch2D(ctrl, xknots, yknots, deg)
    patch.labels[0,0] = '00'
    patch.labels[-1,:] = ['A', 'B', 'C', 'D']

    self.assertEqual(patch.ctrl.shape, (10,4,2))
    self.assertEqual(patch.ctrl[0,0,0], 0)
    self.assertEqual(patch.ctrl[-1,-1,1], 3)
    self.assertEqual(patch.xknots[0], 0)
    self.assertEqual(patch.xknots[-1], 1)
    self.assertEqual(patch.yknots[0], 0)
    self.assertEqual(patch.yknots[-1], 1)
    self.assertEqual(patch.deg, 2)
    self.assertFalse(patch.has_label('foo'))
    self.assertTrue(patch.has_label('D'))
    self.assertFalse(np.all(patch.labels == ''))
    self.assertEqual(patch.labels[-1,0], 'A')
    self.assertEqual(patch.label2idx('C'), (9,2))

    nptest.assert_array_equal(patch.label2ctrl('B'),
                              np.array([9.0, 1.0]))

    with self.assertRaises(LabelError):
      patch.label2idx('foo')

    with self.assertRaises(LabelError):
      patch.label2ctrl('foo')

    with self.assertRaises(LabelError):
      patch.has_label('FIXED')

    with self.assertRaises(LabelError):
      patch.label2idx('FIXED')

    with self.assertRaises(LabelError):
      patch.label2ctrl('FIXED')
