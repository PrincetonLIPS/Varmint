
# TODO: throw errors when bits are missing
# TODO: compute all pairs

class Material:

  @classmethod
  def E(cls):
    ''' Young's modulus '''
    if '_E' in dir(cls):
      return cls._E

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._E = 2 * cls._lmbda * (1 + cls._mu)
      return cls._E
  def youngs(cls):
    return cls.E()

  @classmethod
  def nu(cls):
    ''' Poisson's ratio '''
    if '_nu' in dir(cls):
      return cls._nu

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._nu = cls._lmbda / (2*(cls._lmbda + cls._mu))
      return cls._nu
  def poissons(cls):
    return cls.nu()

  @classmethod
  def lmbda(cls):
    ''' Lame's first parameter '''
    if '_lmbda' in dir(cls):
      return cls._lmbda

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._lmbda = cls._nu * cls._E / ((1+cls._nu)*(1-2*cls._nu))
      return cls._lmbda
  def lame1(cls):
    return cls.lmbda()

  @classmethod
  def mu(cls):
    ''' Shear modulus '''
    if '_mu' in dir(cls):
      return cls._mu

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._mu = cls._E / (2*(1+cls._nu))
      return cls._mu
  def shear(cls):
    return cls.mu()

  @classmethod
  def K(cls):
    ''' Bulk modulus '''
    if '_K' in dir(cls):
      return cls._K

    elif '_E' in dir(cls) and '_nu' in dir(cls):
      cls._K = cls._E / (3*(1-2*cls._nu))
      return cls._K

    elif '_lmbda' in dir(cls) and '_mu' in dir(cls):
      cls._K = cls._lmbda + 2 * cls._mu / 3
      return cls._K
  def bulk(cls):
    return cls.K()

class NinjaFlex(Material):
  _E      = 12000000 # 12MPa -- consider changing these units.
  _nu     = 0.48
  density = 1.20 # g/cm^3
