import jax
import jax.numpy        as np
import jax.numpy.linalg as npla

@jax.jit
def neohookean_energy2d_log(F, shear, bulk):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  return (shear/2) * (I1 - 2 - 2*np.log(J)) + (bulk/2)*np.log(J)**2
vmap_neohookean_energy2d_log = jax.jit(
  jax.vmap(
    neohookean_energy2d_log,
    in_axes=(0, None, None),
  ),
)

@jax.jit
def neohookean_energy3d_log(F, shear, bulk):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  return (shear/2) * (I1 - 2 - 2*np.log(J)) + (bulk/2)*np.log(J)**2
vmap_neohookean_energy3d_log = jax.jit(
  jax.vmap(
    neohookean_energy3d_log,
    in_axes=(0, None, None),
  ),
)

@jax.jit
def neohookean_energy2d(F, shear, bulk):
  I1 = np.trace(F.T @ F)
  J  = npla.det(F)
  J23 = J**(-2/3)
  return (shear/2) * (J23 * (I1+1) - 3) + (bulk/2)*(J-1)**2
vmap_neohookean_energy2d = jax.jit(
  jax.vmap(
    neohookean_energy2d,
    in_axes=(0, None, None),
  ),
)

# TODO: implement Tianju's 3d w/o logs.

class NeoHookean:

  def __init__(self, material, dims=3, log=False):
    self.material = material
    self.dims     = dims
    self.log      = log

  def energy(self, defgrad):
    if self.dims == 2:
      if self.log:
        func = vmap_neohookean_energy2d_log
      else:
        func = vmap_neohookean_energy2d
    else:
      if self.log:
        func = vmap_neohookean_energy3d_log
      else:
        func = vmap_neohookean_energy3d

    return func(defgrad, self.material.shear(), self.material.bulk())

  def density(self):
    return self.material.density
