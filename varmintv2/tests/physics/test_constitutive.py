import unittest as ut
import jax.numpy as np
import numpy.random as npr

from varmintv2.physics.constitutive import *
from varmintv2.physics.materials import *
from varmintv2.utils.exceptions import *


class Test_NeoHookean2D(ut.TestCase):

    def test_zero_energy(self):
        shear = 1.0
        bulk = 1.0
        defgrad = np.eye(2)

        self.assertEqual(neohookean_energy2d(shear, bulk, defgrad), 0.0)

    def test_rigid_energy(self):
        shear = 1.0
        bulk = 1.0

        npr.seed(1)
        for ii in range(100):
            theta = npr.rand() * 2 * np.pi

            defgrad = np.array([[np.cos(theta), -np.sin(theta), ],
                                [np.sin(theta), np.cos(theta)]])

            self.assertAlmostEqual(neohookean_energy2d(shear, bulk, defgrad), 0.0,
                                   places=6)


class Test_NeoHookean2D_log(ut.TestCase):

    def test_zero_energy(self):
        shear = 1.0
        bulk = 1.0
        defgrad = np.eye(2)

        self.assertEqual(neohookean_energy2d_log(shear, bulk, defgrad), 0.0)

    def test_rigid_energy(self):
        shear = 1.0
        bulk = 1.0

        npr.seed(1)
        for ii in range(100):
            theta = npr.rand() * 2 * np.pi

            defgrad = np.array([[np.cos(theta), -np.sin(theta), ],
                                [np.sin(theta), np.cos(theta)]])

            self.assertAlmostEqual(neohookean_energy2d_log(shear, bulk, defgrad), 0.0,
                                   places=6)


class Test_NeoHookean3D(ut.TestCase):

    def test_zero_energy(self):
        shear = 1.0
        bulk = 1.0
        defgrad = np.eye(3)

        self.assertEqual(neohookean_energy3d_log(shear, bulk, defgrad), 0.0)

    def test_rigid_energy(self):
        shear = 1.0
        bulk = 1.0

        npr.seed(1)
        for ii in range(100):
            alpha = npr.rand() * 2 * np.pi
            beta = npr.rand() * 2 * np.pi
            gamma = npr.rand() * 2 * np.pi

            yaw = np.array([
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha),  0],
                [0, 0, 1],
            ])
            pitch = np.array([
                [np.cos(beta), 0, -np.sin(beta)],
                [0, 1, 0],
                [np.sin(beta), 0, np.cos(beta)],
            ])
            roll = np.array([
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)],
            ])

            defgrad = yaw @ pitch @ roll

            self.assertAlmostEqual(neohookean_energy3d_log(shear, bulk, defgrad), 0.0,
                                   places=6)
