import unittest      as ut
import jax
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest

from varmint.discretize import *

class Test_Discretize_Hamiltonian(ut.TestCase):

  def test_stepping_pendulum_rest_1(self):

    def lagrangian(q, qdot):
      # Simple pendulum with a mass and length.
      # q is angle from hanging down vertically.
      gravity = 9.81 # m/s^2
      mass    = 1.0   # kg
      length  = 2.0   # m

      # Needs scalars.
      ke = np.sum(0.5 * mass * length**2 * qdot**2)
      pe = -np.sum(mass*gravity*length*np.cos(q))

      return ke - pe

    stepper = get_hamiltonian_stepper(lagrangian)

    dt = 0.01

    # Start at rest.
    q = np.zeros(1)
    p = np.zeros(1)

    new_q, new_p = stepper(q, p, dt)

    self.assertAlmostEqual(new_q, 0.0)
    self.assertAlmostEqual(new_p, 0.0)


  def test_stepping_pendulum_rest_2(self):

    def lagrangian(q, qdot):
      # Simple pendulum with a mass and length.
      # q is angle from hanging down vertically.
      gravity = 9.81 # m/s^2
      mass    = 1.0   # kg
      length  = 2.0   # m

      # Needs scalars.
      ke = np.sum(0.5 * mass * length**2 * qdot**2)
      pe = -np.sum(mass*gravity*length*np.cos(q))

      return ke - pe

    stepper = get_hamiltonian_stepper(lagrangian)

    dt = 0.01

    # Start at rest.
    q = np.zeros(1)
    p = np.zeros(1)

    for ii in range(10):
      q, p = stepper(q, p, dt)

    self.assertAlmostEqual(q, 0.0)
    self.assertAlmostEqual(p, 0.0)
