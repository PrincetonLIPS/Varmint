import jax
import jax.numpy as np

from numpy.polynomial.legendre import leggauss

class Quad1D_GaussLegendre:

  def __init__(self, degree, interval=(-1,1)):
    self.locs, self.weights = leggauss(degree)
    self.interval = interval
    self.weights = self.weights * (interval[1]-interval[0]) / 2
    self.locs = self.locs * (interval[1]-interval[0]) / 2 \
      + (interval[0]+interval[1])/2

  def compute(self, funcs):
    # Assume the first dimension is nodes.
    return self.weights @ funcs

class Quad2D_TensorGaussLegendre:

  def __init__(self, degree, bounds=((-1,1),(-1,1))):
    self.quadx = Quad1D_GaussLegendre(degree, bounds[0])
    self.quady = Quad1D_GaussLegendre(degree, bounds[1])

  @property
  def locs(self):
    qx, qy = np.meshgrid(self.quadx.locs, self.quady.locs)
    return np.vstack([qx.ravel(), qy.ravel()]).T

  @property
  def weights(self):
    return np.outer(self.quadx.weights, self.quady.weights).ravel()

  def compute(self, funcs, truncate=0):
    # Assume the first dimension of funcs is nodes.
    weights = self.weights

    return np.tensordot(weights, funcs, axes=(0,0))

class Quad3D_TensorGaussLegendre:

  def __init__(self, degree, bounds=((-1,1),(-1,1),(-1,1))):
    self.quadx = Quad1D_GaussLegendre(degree, bounds[0])
    self.quady = Quad1D_GaussLegendre(degree, bounds[1])
    self.quadz = Quad1D_GaussLegendre(degree, bounds[2])

  @property
  def locs(self):
    qx, qy, qz= np.meshgrid(self.quadx.locs, self.quady.locs, self.quadz.locs)
    return np.vstack([qx.ravel(), qy.ravel(), qz.ravel()]).T

  @property
  def weights(self):
    return (self.quadx.weights[:,np.newaxis,np.newaxis]
            * self.quady.weights[np.newaxis,:,np.newaxis]
            * self.quadz.weights[np.newaxis,np.newaxis,:]).ravel()

  def compute(self, funcs):
    return np.tensordot(self.weights, funcs, axes=(0,0))
