import unittest      as ut
import jax
import jax.numpy     as np
import numpy.random  as npr
import numpy.testing as nptest

from varmint.discretize import *

class Test_Discretize_Hamiltonian(ut.TestCase):

  def test_stepping_pendulum_rest_1(self):

    def lagrangian(q, qdot, *args):
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

    def lagrangian(q, qdot, *args):
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


  def test_stepping_pendulum_swing(self):

    def lagrangian(q, qdot, *args):
      # Simple pendulum with a mass and length.
      # q is angle from hanging down vertically.
      gravity = 9.81 # m/s^2
      mass    = 2.0   # kg
      length  = 1.0   # m

      # Needs scalars.
      ke = np.sum(0.5 * mass * length**2 * qdot**2)
      pe = -np.sum(mass*gravity*length*np.cos(q))

      return ke - pe

    stepper = get_hamiltonian_stepper(lagrangian)

    dt = 0.01

    q = np.ones(1) * np.pi * 10.0 / 180.0
    p = np.zeros(1)

    Q = []
    P = []

    # Roll forward two seconds. Very close to one period.
    for ii in range(200):
      q, p = stepper(q, p, dt)
      Q.append(q)
      P.append(p)

    self.assertAlmostEqual(Q[0], Q[-1])


  def test_stepping_pendulum_swing_grad(self):

    @jax.jit
    def two_second_theta(params):
      # params is an ndarray

      def lagrangian(q, qdot, params):
        mass = params[0]
        length = params[1]

        gravity = 9.81
        ke = np.sum(0.5 * mass * length**2 * qdot**2)
        pe = -np.sum(mass*gravity*length*np.cos(q))
        return ke - pe

      stepper = get_hamiltonian_stepper(lagrangian)

      dt = 0.01

      q = np.ones(1) * np.pi * 10.0 / 180.0
      p = np.zeros(1)

      Q = []
      P = []

      # Roll forward two seconds. Very close to one period.
      for ii in range(200):
        q, p = stepper(q, p, dt, params)
        Q.append(q)
        P.append(p)

      return np.sum((Q[-1]-Q[0])**2)

    p = np.array([2.0, 1.5])
    # print(two_second_theta(p))

    vg_2s_theta = jax.jit(jax.value_and_grad(two_second_theta))

    # Should not fail.
    vg_2s_theta(p)

    #step = 1e-1
    #for ii in range(1000):
    #  v, g = vg_2s_theta(p)
    #  print(ii, v, p)
    #  p = p - step*g
