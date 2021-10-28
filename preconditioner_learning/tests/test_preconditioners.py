from preconditioner_learning.preconditioners import *
import unittest
import numpy.testing as nptest
import jax.numpy as jnp
import jax
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)


class TestPreconditioners(unittest.TestCase):

    def test_identity(self):
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = L @ L.T + 1e-3 * np.eye(n)
            x = np.random.randn(n)

            p = IdentityPreconditioner(A)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(x, apply(x))
            nptest.assert_allclose(x, apply(x))

    def test_diagonal(self):
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = L @ L.T + 1e-3 * np.eye(n)
            x = np.random.randn(n)

            p = DiagonalPreconditioner(A)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(apply_T(x), apply(x))

        n = 10
        A = 4. * np.eye(n)
        x = np.random.randn(n)

        p = DiagonalPreconditioner(A)

        apply = p.get_apply_fn()
        apply_T = p.get_apply_T_fn()

        nptest.assert_allclose(apply(x), x / 2.)

    def test_fixed_diagonal(self):
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = L @ L.T + 1e-3 * np.eye(n)
            B = 4. * np.eye(n)
            x = np.random.randn(n)

            p = FixedDiagonalPreconditioner(A, B)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(apply_T(x), apply(x))
            nptest.assert_allclose(apply(x), x / 2.)

    def test_exact(self):
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = L @ L.T + 1e-3 * np.eye(n)
            x = np.random.randn(n)

            p = ExactPreconditioner(A)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(apply_T(A @ apply(x)), x)

    def test_fixed_matrix(self):
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = np.eye(n) * 4.
            B = L @ L.T + 1e-3 * np.eye(n)
            x = np.random.randn(n)

            p = FixedMatrixPreconditioner(A, B)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(apply_T(B @ apply(x)), x)

    def test_incomplete_chol(self):
        # For a dense matrix, it should behave the same as regular Cholesky.
        # Broken!!!!!
        for _ in range(10):
            n = 10
            L = np.random.randn(n, n)

            # Should be PSD.
            A = L @ L.T + 1e-3 * np.eye(n)
            x = np.random.randn(n)

            p = IncompleteCholPreconditioner(A)

            apply = p.get_apply_fn()
            apply_T = p.get_apply_T_fn()

            nptest.assert_allclose(apply_T(A @ apply(x)), x)


if __name__ == '__main__':
    unittest.main()
