import unittest as ut

from varmint.materials import *
from varmint.exceptions import *

class Test_Material(ut.TestCase):

  def test_props_1(self):
    class NewMat(Material):
      _E       = 1.0
      _nu      = 0.25
      _density = 1.0

    mat = NewMat()

    self.assertEqual(mat.E, 1.0)
    self.assertEqual(mat.youngs, 1.0)

    self.assertEqual(mat.nu, 0.25)
    self.assertEqual(mat.poissons, 0.25)

    self.assertEqual(mat.density, 1.0)

  def test_props_2(self):
    lmbda = 1.3
    mu    = 4.2
    class NewMat(Material):
      _lmbda   = lmbda
      _mu      = mu
      _density = 1.0

    mat = NewMat()

    self.assertEqual(mat.lmbda, 1.3)
    self.assertEqual(mat.lame1, 1.3)

    self.assertEqual(mat.mu, 4.2)
    self.assertEqual(mat.shear, 4.2)

  def test_shear(self):
    E = 1.0
    nu = 0.25
    class NewMat(Material):
      _E       = E
      _nu      = nu
      _density = 1.0

    mat = NewMat()

    G = E / (2*(1+nu))

    self.assertEqual(mat.mu, G)
    self.assertEqual(mat.shear, G)

  def test_bulk(self):
    E = 1.0
    nu = 0.25
    class NewMat(Material):
      _E       = E
      _nu      = nu
      _density = 1.0

    mat = NewMat()

    K = E / (3*(1 - 2*nu))

    self.assertEqual(mat.K, K)
    self.assertEqual(mat.bulk, K)

  def test_lame1(self):
    E = 1.0
    nu = 0.25
    class NewMat(Material):
      _E       = E
      _nu      = nu
      _density = 1.0

    mat = NewMat()

    L = E * nu / ((1+nu)*(1-2*nu))

    self.assertEqual(mat.lmbda, L)
    self.assertEqual(mat.lame1, L)

  def test_youngs(self):
    lmbda = 1.3
    mu    = 4.2
    class NewMat(Material):
      _lmbda   = lmbda
      _mu      = mu
      _density = 1.0

    mat = NewMat()

    E = mu * (3*lmbda + 2*mu) / (lmbda + mu)

    self.assertEqual(mat.E, E)
    self.assertEqual(mat.youngs, E)

  def test_poissons(self):
    lmbda = 1.3
    mu    = 4.2
    class NewMat(Material):
      _lmbda   = lmbda
      _mu      = mu
      _density = 1.0

    mat = NewMat()

    nu = lmbda / (2*(lmbda+mu))

    self.assertEqual(mat.nu, nu)
    self.assertEqual(mat.poissons, nu)

  def test_convert_1(self):
    E = 1.0
    nu = 0.25
    class NewMat1(Material):
      _E       = E
      _nu      = nu
      _density = 1.0

    mat1 = NewMat1()

    class NewMat2(Material):
      _lmbda   = mat1.lmbda
      _mu      = mat1.mu
      _density = 1.0

    mat2 = NewMat2()

    self.assertAlmostEqual(E, mat2.E)
    self.assertAlmostEqual(nu, mat2.nu)

  def test_convert_2(self):
    lmbda = 1.3
    mu    = 4.2
    class NewMat1(Material):
      _lmbda   = lmbda
      _mu      = mu
      _density = 1.0

    mat1 = NewMat1()

    class NewMat2(Material):
      _E       = mat1.E
      _nu      = mat1.nu
      _density = 1.0

    mat2 = NewMat2()

    self.assertAlmostEqual(lmbda, mat2.lmbda)
    self.assertAlmostEqual(mu, mat2.mu)

  def test_bad_pair_1(self):
    class BadMat(Material):
      _E       = 1.0
      _mu      = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_pair_2(self):
    class BadMat(Material):
      _E       = 1.0
      _lmbda   = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_pair_3(self):
    class BadMat(Material):
      _nu      = 1.0
      _lmbda   = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_pair_4(self):
    class BadMat(Material):
      _nu      = 1.0
      _mu      = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_youngs_1(self):
    class BadMat(Material):
      _E       = -1.0
      _nu      = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_youngs_2(self):
    class BadMat(Material):
      _E       = 12000000.0
      _nu      = 1.0
      _density = 1.0

    with self.assertRaises(UnitsError):
      mat = BadMat()

  def test_bad_poissons_1(self):
    class BadMat(Material):
      _E       = 1.0
      _nu      = -0.5
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_poissons_2(self):
    class BadMat(Material):
      _E       = 1.0
      _nu      = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_lmbda_1(self):
    class BadMat(Material):
      _lmbda   = -1.0
      _mu      = 1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_lmbda_2(self):
    class BadMat(Material):
      _lmbda   = 12000000.0
      _mu      = 1.0
      _density = 1.0

    with self.assertRaises(UnitsError):
      mat = BadMat()

  def test_bad_mu_1(self):
    class BadMat(Material):
      _lmbda   = 1.0
      _mu      = -1.0
      _density = 1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_mu_2(self):
    class BadMat(Material):
      _lmbda   = 1.0
      _mu      = 12000000.0
      _density = 1.0

    with self.assertRaises(UnitsError):
      mat = BadMat()

  def test_bad_density_1(self):
    class BadMat(Material):
      _lmbda   = 1.0
      _mu      = 1.0
      _density = -1.0

    with self.assertRaises(MaterialError):
      mat = BadMat()

  def test_bad_density_2(self):
    class BadMat(Material):
      _lmbda   = 1.0
      _mu      = 1.0
      _density = 0.01

    with self.assertRaises(UnitsError):
      mat = BadMat()

  def test_bad_density_3(self):
    class BadMat(Material):
      _lmbda   = 1.0
      _mu      = 1.0
      _density = 10.0

    with self.assertRaises(UnitsError):
      mat = BadMat()
