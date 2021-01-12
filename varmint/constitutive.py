import jax
import jax.numpy        as np
import jax.numpy.linalg as npla

from functools import partial

#@jax.jit
def neohookean_energy2d_log(shear, bulk, F):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  return (shear/2) * (I1 - 2 - 2*np.log(J)) + (bulk/2)*np.log(J)**2
#vmap_neohookean_energy2d_log = jax.jit(
#  jax.vmap(
#    neohookean_energy2d_log,
#    in_axes=(0, None, None),
#  ),
#)

#@jax.jit
def neohookean_energy3d_log(shear, bulk, F):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  return (shear/2) * (I1 - 3 - 2*np.log(J)) + (bulk/2)*(J-1)**2
#vmap_neohookean_energy3d_log = jax.jit(
#  jax.vmap(
#    neohookean_energy3d_log,
#    in_axes=(0, None, None),
#  ),
#)

#@jax.jit
def neohookean_energy2d(shear, bulk, F):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  J23 = J**(-2/3)
  return (shear/2) * (J23 * (I1+1) - 3) + (bulk/2)*(J-1)**2
#vmap_neohookean_energy2d = jax.jit(
#  jax.vmap(
#    neohookean_energy2d,
#    in_axes=(0, None, None),
#  ),
#)

class NeoHookean2D:

  def __init__(self, material, log=True, thickness=1):
    ''' thickness in cm '''
    self.material  = material
    self.log       = log
    self.thickness = thickness
    self.shear     = self.material.shear
    self.bulk      = self.material.bulk
    self.density   = self.material.density

  def get_energy_fn(self):
    if self.log:
      return partial(neohookean_energy2d_log, self.shear, self.bulk)

    return partial(neohookean_energy2d, self.shear, self.bulk)

  def density(self):
    # TODO: How should this interact with third dimension?
    return self.density

class NeoHookean3D:

  def __init__(self, material):
    self.material = material
    self.shear    = self.material.shear
    self.bulk     = self.material.bulk
    self.density  = self.material.density

  def get_energy_fn(self):
    return partial(neohookean_energy3d_log, self.shear, self.bulk)

  def density(self):
    return self.density
