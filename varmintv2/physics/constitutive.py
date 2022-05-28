from abc import ABC, abstractmethod
from typing import Callable
import jax
import jax.numpy as np
import jax.numpy.linalg as npla

from functools import partial


def neohookean_energy2d_log(F, E, nu):
    # Convert E and nu to shear and bulk
    shear = E / (2*(1+nu))
    bulk = E / (3*(1-2*nu))

    I1 = np.trace(F.T @ F)
    J = npla.det(F)
    return (shear/2) * (I1 - 2 - 2*np.log(J)) + (bulk/2)*np.log(J)**2


def linear_elastic_energy2d(F, lmbda, mu):
    strain = 0.5 * (F + F.T - 2 * np.eye(2))
    return 0.5 * lmbda * (np.trace(strain) ** 2) + \
        mu * np.tensordot(strain, strain, axes=([0, 1], [0, 1]))


def neohookean_energy3d_log(F, shear, bulk):
    I1 = np.trace(F.T @ F)
    J = npla.det(F)
    return (shear/2) * (I1 - 3 - 2*np.log(J)) + (bulk/2)*(J-1)**2


def neohookean_energy2d(F, E, nu):
    # Convert E and nu to shear and bulk
    shear = E / (2*(1+nu))
    bulk = E / (3*(1-2*nu))

    I1 = np.trace(F.T @ F)
    J = npla.det(F)
    J23 = J**(-2/3)
    return (shear/2) * (J23 * (I1+1) - 3) + (bulk/2)*(J-1)**2


class PhysicsModel(ABC):
    @abstractmethod
    def get_energy_fn(self) -> Callable:
        pass

    @property
    @abstractmethod
    def density(self) -> float:
        pass


class LinearElastic2D(PhysicsModel):
    def __init__(self, material, thickness=1):
        """ thickness in cm """
        self.material = material
        self.thickness = thickness
        self.lmbda = self.material.lmbda
        self.mu = self.material.mu
        self._density = self.material.density

    def get_energy_fn(self):
        return linear_elastic_energy2d
        #return partial(linear_elastic_energy2d, lmbda=self.lmbda, mu=self.mu)

    @property
    def density(self):
        # TODO: How should this interact with third dimension?
        return self._density


class NeoHookean2D(PhysicsModel):
    def __init__(self, material, log=True, thickness=1):
        """ thickness in cm """
        self.material = material
        self.log = log
        self.thickness = thickness
        self.shear = self.material.shear
        self.bulk = self.material.bulk
        self._density = self.material.density

    def get_energy_fn(self):

        if self.log:
            return neohookean_energy2d_log
            #return partial(neohookean_energy2d_log, shear=self.shear, bulk=self.bulk)

        return neohookean_energy2d
        #return partial(neohookean_energy2d, shear=self.shear, bulk=self.bulk)

    @property
    def density(self):
        # TODO: How should this interact with third dimension?
        return self._density


class NeoHookean3D(PhysicsModel):
    def __init__(self, material):
        self.material = material
        self.shear = self.material.shear
        self.bulk = self.material.bulk
        self._density = self.material.density

    def get_energy_fn(self):
        return neohookean_energy3d_log
        #return partial(neohookean_energy3d_log, shear=self.shear, bulk=self.bulk)

    @property
    def density(self):
        return self._density
