import unittest  as ut
import jax.numpy as np

from constitutive import *
from materials    import *
from exceptions   import *

class Test_NeoHookean2D(ut.TestCase):

  def test_zero_energy(self):
    shear   = 1.0
    bulk    = 1.0
    defgrad = np.eye(2)

    self.assertEqual(neohookean_energy2d(defgrad, shear, bulk), 0.0)

class Test_NeoHookean2D_log(ut.TestCase):

  def test_zero_energy(self):
    shear   = 1.0
    bulk    = 1.0
    defgrad = np.eye(2)

    self.assertEqual(neohookean_energy2d_log(defgrad, shear, bulk), 0.0)

class Test_NeoHookean3D(ut.TestCase):

  def test_zero_energy(self):
    shear   = 1.0
    bulk    = 1.0
    defgrad = np.eye(3)

    self.assertEqual(neohookean_energy3d_log(defgrad, shear, bulk), 0.0)
