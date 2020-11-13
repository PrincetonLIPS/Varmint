import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import logging

from vmap_utils import *

def strain_potential(ctrl):
  pass

def gravitational_potential(ctrl):
  pass

def kinetic_energy(ctrl, ctrl_vel):
  pass

def generate_lagrangian():

  def lagrangian(q, qdot):


if __name__ == '__main__':
  import shape3d
  import materials

  from constitutive import NeoHookean
  from quadrature import Quad3D_TensorGaussLegendre

  mat   = NeoHookean(materials.NinjaFlex)
  quad  = Quad3D_TensorGaussLegendre(10)
  shape = shape3d.test_shape1()

  generate_lagrangian(quad, shape, mat)
