
from .exceptions import *

class classproperty(object):
  ''' Make it so that there can be class-level properties. '''
  def __init__(self, f):
    self.f = f
  def __get__(self, obj, owner):
    return self.f(owner)

class Material:
  ''' Base class to represent material properties.

  The core functionality of this class is to implement relationships between
  commonly used material parameters, specifically:

  Young's modulus (GPa)

  Poisson's ratio (unitless, in (0, 0.5))

  Lame's first parameter: (GPa)

  Shear modulus: (GPa)

  Bulk modulus: (GPa)

  Materials can be specified either with a Young's/Poisson's pair or with a
  Lame-first/shear pair.  These are _E/_nu and _lmbda/_mu, respectively.
  This class also requires density to be specified in grams per cubic cm.

  '''

  def __init__(self):
    self.check_values()
    self.check_pairs()

  def check_values(self):
    ''' Run through some simple range checks for sensible parameters. '''
    if '_E' in dir(self):
      self.verify_youngs()
    if '_nu' in dir(self):
      self.verify_poissons()
    if '_lmbda' in dir(self):
      self.verify_lame1()
    if '_mu' in dir(self):
      self.verify_shear()
    if '_K' in dir(self):
      self.verify_bulk()

    self.verify_density()

  def verify_youngs(self):
    ''' Basic checks for reasonable Young's moduli '''
    if self._E > 1000:
      raise UnitsError('''
      Young's modulus of %f seems too large. This should be in GPa.
      ''' % (self._E))
    elif self._E < 0:
      raise MaterialError('''
      Young's modulus must be positive. Got %f.
      ''' % (self._E))

  def verify_poissons(self):
    ''' Basic checks for reasonable Poisson's ratio '''
    if self._nu < 0 or self._nu > 0.5:
      raise MaterialError('''
      Poisson's ratio must be between 0 and 0.5. Got %f.
      ''' % (self.nu))

  def verify_lame1(self):
    ''' Basic checks for reasonble Lame's first parameter '''
    if self._lmbda > 1000:
      raise UnitsError('''
      Lame's first parameter of %f seems too large. This should be in GPa.
      ''' % (self._lmbda))
    elif self._lmbda < 0:
      raise MaterialError('''
      Lame's first parameter must be positive. Got %f.
      ''' % (self._lmbda))

  def verify_shear(self):
    ''' Basic checks for reasonble shear modulus '''
    if self._mu > 1000:
      raise UnitsError('''
      Shear modulus of %f seems too large. This should be in GPa.
      ''' % (self._mu))
    elif self._mu < 0:
      raise MaterialError('''
      Shear modulus must be positive. Got %f.
      ''' % (self._mu))

  def verify_bulk(self):
    ''' Basic checks for reasonble bulk modulus '''
    if self._K > 1000:
      raise UnitsError('''
      Bulk modulus of %f seems too large. This should be in GPa.
      ''' % (self._K))
    elif self._K < 0:
      raise MaterialError('''
      Bulk modulus must be positive. Got %f.
      ''' % (self._K))

  def verify_density(self):
    ''' Basic checks for reasonable densities. '''
    if self._density <= 0:
      raise MaterialError('''
      Density must be a positive number. Got %f.
      ''' % (self._density))
    elif self._density < 0.05:
      raise UnitsError('''
      Surprisingly small density of %f (less than styrofoam).
      This quantity should be in g/cm^3.
      ''' % (self._density))
    elif self._density > 8:
      raise UnitsError('''
      Surprisingly high density of %f (greater than steel).
      This quantity should be in g/cm^3.
      ''' % (self._density))

  def check_pairs(self):
    ''' Since we don't have all conversions implemented, verify that we can at
    least accommodate the major pairs. '''
    if '_E' in dir(self) and '_nu' in dir(self):
      return
    elif '_lmbda' in dir(self) and '_mu' in dir(self):
      return
    else:
      raise MaterialError('''
      Need to specify either E/nu or lmbda/mu
      ''')

  @classproperty
  def E(cls):
    ''' Young's modulus: GPa = 10^9 kg / (m * s**2) '''
    if '_E' in dir(cls):
      return cls._E

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._E = cls._mu * (3 * cls._lmbda + 2 * cls._mu) / (cls._lmbda + cls._mu)
      return cls._E

  @classproperty
  def youngs(cls):
    return cls.E

  @classproperty
  def nu(cls):
    ''' Poisson's ratio: unitless '''
    if '_nu' in dir(cls):
      return cls._nu

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._nu = cls._lmbda / (2*(cls._lmbda + cls._mu))
      return cls._nu

  @classproperty
  def poissons(cls):
    return cls.nu

  @classproperty
  def lmbda(cls):
    ''' Lame's first parameter '''
    if '_lmbda' in dir(cls):
      return cls._lmbda

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._lmbda = cls._nu * cls._E / ((1+cls._nu)*(1-2*cls._nu))
      return cls._lmbda

  @classproperty
  def lame1(cls):
    return cls.lmbda

  @classproperty
  def mu(cls):
    ''' Shear modulus: GPa = 10^9 kg / (m * s**2) '''
    if '_mu' in dir(cls):
      return cls._mu

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._mu = cls._E / (2*(1+cls._nu))
      return cls._mu

  @classproperty
  def shear(cls):
    return cls.mu

  @classproperty
  def K(cls):
    ''' Bulk modulus: GPa = 10^9 kg / (m * s**2)  '''
    if '_K' in dir(cls):
      return cls._K

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._K = cls._E / (3*(1-2*cls._nu))
      return cls._K

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._K = cls._lmbda + 2 * cls._mu / 3
      return cls._K

  @classproperty
  def bulk(cls):
    return cls.K

  @classproperty
  def density(cls):
    return cls._density

class NinjaFlex(Material):
  # https://arxiv.org/pdf/1704.00943.pdf
  _E       = 0.11 # GPa
  _nu      = 0.34
  _density = 1.040 # g / cm^3

class SiliconeRubber(Material):
  _E       = 0.05 # GPa
  _nu      = 0.48
  _density = 1.2

class PLA(Material):
  _E       = 3.5 # GPa
  _nu      = 0.35
  _density = 1.25

class ABS(Material):
  _E       = 1.1
  _nu      = 0.35 # ?
  _density = 1.2

class CollapsingMat(Material):
  _E = 0.00005
  _nu = 0.48
  _density = 1.0

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0
