from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import numpy.testing as nptest
import unittest as ut

from varmintv2.geometry import bsplines
from varmintv2.geometry import elements


class Test_Patch2D(ut.TestCase):
    ''' Test basic functionality with no labels. '''

    def setUp(self):
        spline_deg = 4
        xknots = bsplines.default_knots(spline_deg, 10)
        yknots = bsplines.default_knots(spline_deg, 5)
        quad_deg = 10

        self.patch = elements.Patch2D(xknots, yknots, spline_deg, quad_deg)

    def test_spline_deg(self):
        self.assertEqual(self.patch.spline_deg, 4)

    def test_quad_deg(self):
        self.assertEqual(self.patch.quad_deg, 10)

    def test_knots(self):
        self.assertEqual(self.patch.xknots[0], 0)
        self.assertEqual(self.patch.xknots[-1], 1)
        self.assertEqual(self.patch.yknots[0], 0)
        self.assertEqual(self.patch.yknots[-1], 1)

    def test_map_fn(self):
        deformation_fn = self.patch.get_map_fn()
        ctrl = bsplines.mesh(np.arange(10), np.arange(5))
        deformation = deformation_fn(ctrl)
        num_points = self.patch.num_quad_pts

        self.assertEqual(deformation.shape, (num_points, 2))

    def test_map_boundary_fn(self):
        ctrl = bsplines.mesh(np.arange(10), np.arange(5))

        for i in range(self.patch.num_boundaries):
            boundary_deformation_fn = self.patch.get_map_boundary_fn(i)
            deformation = boundary_deformation_fn(ctrl)
            num_points = self.patch.num_boundary_quad_pts(i)

            self.assertEqual(deformation.shape, (num_points, 2))

    def test_map_jac_fn(self):
        jac_u_fn = self.patch.get_map_jac_fn()
        ctrl = bsplines.mesh(np.arange(10), np.arange(5))
        jac_u = jac_u_fn(ctrl)
        num_points = self.patch.num_quad_pts

        self.assertEqual(jac_u.shape, (num_points, 2, 2))

    def test_map_jac_1d_spline_fn(self):
        # For degree 1 splines, the mapping will be linear,
        # so we can directly compute Jacobians.
        spline_deg = 1
        xknots = bsplines.default_knots(spline_deg, 10)
        yknots = bsplines.default_knots(spline_deg, 5)
        quad_deg = 10

        patch = elements.Patch2D(xknots, yknots, spline_deg, quad_deg)

        jac_u_fn = patch.get_map_jac_fn()
        ctrl = bsplines.mesh(np.arange(10), np.arange(5))
        jac_u = jac_u_fn(ctrl)
        num_points = patch.num_quad_pts

        nptest.assert_allclose(jac_u, np.tile(np.diag(np.array([9., 4.])),
                                              (num_points, 1, 1)), atol=1e-7)

    def test_map_boundary_jac_fn(self):
        ctrl = bsplines.mesh(np.arange(10), np.arange(5))

        for i in range(self.patch.num_boundaries):
            boundary_jac_fn = self.patch.get_map_boundary_jac_fn(i)
            jacs = boundary_jac_fn(ctrl)
            num_points = self.patch.num_boundary_quad_pts(i)

            self.assertEqual(jacs.shape, (num_points, 2))

    def test_ctrl_jacobian_fn(self):
        jac_ctrl_fn = self.patch.get_ctrl_jacobian_fn()
        ctrl = bsplines.mesh(np.arange(10.), np.arange(5.))
        jac_ctrl = jac_ctrl_fn(ctrl)
        num_points = self.patch.num_quad_pts

        self.assertEqual(jac_ctrl.shape, (num_points, 2, 10, 5, 2))

    def test_quad_fn(self):
        quad_fn = self.patch.get_quad_fn()
        num_points = self.patch.num_quad_pts
        ordinates = 2*np.ones(num_points)

        self.assertAlmostEqual(quad_fn(ordinates), 2.0)

    def test_boundary_quad_fn(self):
        for i in range(self.patch.num_boundaries):
            quad_fn = self.patch.get_boundary_quad_fn(i)
            num_points = self.patch.num_boundary_quad_pts(i)
            ordinates = 2*np.ones(num_points)

            self.assertAlmostEqual(quad_fn(ordinates), 2.0, places=4)

    def test_affine_map(self):
        random_mat = np.random.randn(2, 2)
        pos_random_mat = random_mat.T @ random_mat

        map_fn = self.patch.get_map_fn()
        map_jac_fn = self.patch.get_map_jac_fn()

        ctrl_before = bsplines.mesh(np.arange(10.), np.arange(5.))
        ctrl_after = ctrl_before @ pos_random_mat

        map_before = map_fn(ctrl_before)
        map_after = map_fn(ctrl_after)

        nptest.assert_allclose(map_before @ pos_random_mat, map_after)

        jac_before = map_jac_fn(ctrl_before)
        jac_after = map_jac_fn(ctrl_after)

        # Matmul goes in the other direction, so just use einsum.
        contracted = np.einsum('ijk,jl->ilk', jac_before, pos_random_mat)
        nptest.assert_allclose(contracted, jac_after)

    def test_affine_map_boundary(self):
        random_mat = np.random.randn(2, 2)
        pos_random_mat = random_mat.T @ random_mat

        ctrl_before = bsplines.mesh(np.arange(10.), np.arange(5.))
        ctrl_after = ctrl_before @ pos_random_mat

        for i in range(self.patch.num_boundaries):
            map_fn = self.patch.get_map_boundary_fn(i)
            map_jac_fn = self.patch.get_map_boundary_jac_fn(i)

            map_before = map_fn(ctrl_before)
            map_after = map_fn(ctrl_after)

            nptest.assert_allclose(map_before @ pos_random_mat, map_after)

            jac_before = map_jac_fn(ctrl_before)
            jac_after = map_jac_fn(ctrl_after)

            # Don't need einsum here because boundary Jacobians have
            # scalar inputs which are squeezed out..
            nptest.assert_allclose(jac_before @ pos_random_mat, jac_after)
