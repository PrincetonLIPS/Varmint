from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import numpy.testing as nptest
import unittest as ut

from varmintv2.geometry.geometry import SingleElementGeometry
from varmintv2.geometry import elements
from varmintv2.geometry import bsplines
from varmintv2.utils import geometry_utils
from varmintv2.physics.materials import WigglyMat
from varmintv2.physics.constitutive import NeoHookean2D


class Test_SEGeometry(ut.TestCase):
    """ Test basic functionality with no labels. """

    def setUp(self):
        spline_deg = 4
        xknots = bsplines.default_knots(spline_deg, 10)
        yknots = bsplines.default_knots(spline_deg, 10)
        quad_deg = 10

        self.patch = elements.Patch2D(xknots, yknots, spline_deg, quad_deg)

    def test_two_patch_geometry(self):
        m1 = bsplines.mesh(np.arange(10), np.arange(10))
        m2 = bsplines.mesh(np.arange(10) + 9, np.arange(10))
        m = np.stack((m1, m2), axis=0)

        n_cp = m.size // m.shape[-1]
        local_indices = np.arange(n_cp).reshape(m.shape[:-1])

        constraints = geometry_utils.generate_constraints(m)
        dirichlet_labels = {}
        dirichlet_labels['1'] = \
            geometry_utils.get_patch_side_index_array(m, 0, 'left')
        traction_labels = {}

        mat = NeoHookean2D(WigglyMat)
        seg = SingleElementGeometry(self.patch, mat,
                                    m, constraints,
                                    dirichlet_labels, traction_labels)
        
        self.assertEqual(seg.n_components, 2 * 10 * 10 - 10)

        l2g, g2l = seg.get_global_local_maps()

        ctrl_pos = npr.randn(*m.shape)
        ctrl_pos = geometry_utils.constrain_ctrl(ctrl_pos, constraints)
        ctrl_pos_fixed = ctrl_pos# * dirichlet_labels['1'][..., np.newaxis]

        ctrl_vels = npr.randn(*m.shape)
        ctrl_vels = geometry_utils.constrain_ctrl(ctrl_vels, constraints)
        ctrl_vels_fixed = ctrl_vels# * dirichlet_labels['1'][..., np.newaxis]

        g_pos = l2g(ctrl_pos)
        g_vel = l2g(ctrl_vels)
        self.assertEqual(g_pos.shape, (2 * (2 * 10 * 10 - 20),))
        self.assertEqual(g_vel.shape, (2 * (2 * 10 * 10 - 20),))

        l_pos = g2l(g_pos, ctrl_pos_fixed)
        l_vel = g2l(g_vel, ctrl_vels_fixed)

        nptest.assert_equal(l_pos, ctrl_pos)
        nptest.assert_equal(l_vel, ctrl_vels)

        self.assertTrue(np.all(seg.active_traction_boundaries == 0))
        self.assertEqual(local_indices[seg.all_dirichlet_indices > 0].size, 10)

    def test_dirichlet_completion(self):
        m1 = bsplines.mesh(np.arange(10), np.arange(10))
        m2 = bsplines.mesh(np.arange(10) + 9, np.arange(10))
        m = np.stack((m1, m2), axis=0)

        n_cp = m.size // m.shape[-1]
        local_indices = np.arange(n_cp).reshape(m.shape[:-1])

        constraints = geometry_utils.generate_constraints(m)
        dirichlet_labels = {}
        dirichlet_labels['1'] = \
            geometry_utils.get_patch_side_index_array(m, 0, 'right')
        traction_labels = {}

        mat = NeoHookean2D(WigglyMat)
        seg = SingleElementGeometry(self.patch, mat,
                                    m, constraints,
                                    dirichlet_labels, traction_labels)

        self.assertEqual(local_indices[seg.all_dirichlet_indices > 0].size, 20)