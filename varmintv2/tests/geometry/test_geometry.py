from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import numpy.testing as nptest
import unittest as ut

from varmintv2.geometry.geometry import SEGlobalLocalMap, SEBoundaryConditions, SingleElementGeometry
from varmintv2.geometry import elements
from varmintv2.geometry import bsplines


class Test_SEGeometry(ut.TestCase):
    """ Test basic functionality with no labels. """

    def setUp(self):
        spline_deg = 4
        xknots = bsplines.default_knots(spline_deg, 10)
        yknots = bsplines.default_knots(spline_deg, 10)
        quad_deg = 10

        self.patch = elements.Patch2D(xknots, yknots, spline_deg, quad_deg)
        self.ctrl = bsplines.mesh(np.arange(10), np.arange(10))

    def test_global_local_map(self):
        # Construct array with arbitrary shared indices.
        index_array = np.arange(24)
        index_array[20:24] = index_array[16:20]
        index_array = index_array.reshape(4, 6)
        n_components = 20

        fixed_labels = np.array([0, 1, 2])
        nonfixed_labels = np.arange(3, 20)

        glmap = SEGlobalLocalMap(n_elements=1,
                                 element=self.patch,
                                 index_array=index_array,
                                 fixed_labels=fixed_labels,
                                 nonfixed_labels=nonfixed_labels,
                                 n_components=n_components)

        l2g, g2l = glmap.get_global_local_maps()

        ctrl_pos = npr.randn(24, 2)
        fixed_pos = ctrl_pos.copy()
        fixed_pos[3:, :] = 0

        ctrl_pos[20:24, :] = ctrl_pos[16:20, :]
        ctrl_pos = ctrl_pos.reshape(4, 6, 2)
        fixed_pos = fixed_pos.reshape(4, 6, 2)

        ctrl_vel = npr.randn(24, 2)
        fixed_vel = ctrl_vel.copy()
        fixed_vel[3:, :] = 0

        ctrl_vel[20:24, :] = ctrl_vel[16:20, :]
        ctrl_vel = ctrl_vel.reshape(4, 6, 2)
        fixed_vel = fixed_vel.reshape(4, 6, 2)

        g_pos, g_vel = l2g(ctrl_pos, ctrl_vel)
        self.assertEqual(g_pos.shape, (17*2,))
        self.assertEqual(g_vel.shape, (17*2,))

        l_pos, l_vel = g2l(g_pos, g_vel, fixed_pos, fixed_vel)
        nptest.assert_array_equal(l_pos, ctrl_pos)
        nptest.assert_array_equal(l_vel, ctrl_vel)

    def test_boundary_conditions(self):
        pass