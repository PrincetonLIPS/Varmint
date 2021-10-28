import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import numpy.testing as nptest
import unittest as ut

from varmint.bsplines import *
from varmint.constitutive import NeoHookean2D
from varmint.materials import NinjaFlex
from varmint.patch2d import Patch2D
from varmint.exceptions import LabelError


class Test_Patch2D_NoLabels(ut.TestCase):
    ''' Test basic functionality with no labels. '''

    def setUp(self):
        spline_deg = 4
        xknots = default_knots(spline_deg, 10)
        yknots = default_knots(spline_deg, 5)
        material = NeoHookean2D(NinjaFlex)
        quad_deg = 10

        self.patch = Patch2D(xknots, yknots, spline_deg, material, quad_deg)

    def test_spline_deg(self):
        self.assertEqual(self.patch.spline_deg, 4)

    def test_quad_deg(self):
        self.assertEqual(self.patch.quad_deg, 10)

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

    def test_get_fixed(self):
        fixed = self.patch.get_fixed()
        self.assertEqual(len(fixed), 0)

    def test_deformation_fn(self):
        deformation_fn = self.patch.get_deformation_fn()
        ctrl = mesh(np.arange(10), np.arange(5))
        deformation = deformation_fn(ctrl)
        num_points = self.patch.num_quad_pts()

        self.assertEqual(deformation.shape, (num_points, 2))

    def test_jacobian_u_fn(self):
        jac_u_fn = self.patch.get_jacobian_u_fn()
        ctrl = mesh(np.arange(10), np.arange(5))
        jac_u = jac_u_fn(ctrl)
        num_points = self.patch.num_quad_pts()

        self.assertEqual(jac_u.shape, (num_points, 2, 2))

    def test_jacobian_ctrl_fn(self):
        jac_ctrl_fn = self.patch.get_jacobian_ctrl_fn()
        ctrl = mesh(np.arange(10.), np.arange(5.))
        jac_ctrl = jac_ctrl_fn(ctrl)
        num_points = self.patch.num_quad_pts()

        self.assertEqual(jac_ctrl.shape, (num_points, 2, 10, 5, 2))

    def test_energy_fn(self):
        energy_fn = self.patch.get_energy_fn()
        defgrad = np.eye(2)

        self.assertEqual(energy_fn(defgrad), 0.0)

    def test_quad_fn(self):
        quad_fn = self.patch.get_quad_fn()
        num_points = self.patch.num_quad_pts()
        ordinates = 2*np.ones(num_points)

        self.assertAlmostEqual(quad_fn(ordinates), 2.0)


"""
class Test_Patch2D_Labels(ut.TestCase):
  ''' Test basic label functionality. '''

  def setUp(self):
    deg    = 4
    xknots = default_knots(deg, 10)
    yknots = default_knots(deg, 5)
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
"""
