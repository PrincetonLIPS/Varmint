import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

import jax
import jax.numpy as jnp

from scipy.sparse.linalg import spilu, spsolve_triangular
from scipy.sparse import diags


class Preconditioner(object):
    """Base class. Does nothing."""
    def __init__(self, A, *args, **kwargs):
        raise NotImplementedError

    def get_apply_fn(self):
        raise NotImplementedError
    
    def get_apply_T_fn(self):
        raise NotImplementedError


class DiagonalPreconditioner(Preconditioner):
    """Uses the inverse of the diagonal as a preconditioner."""

    def __init__(self, A):
        self.diag = jnp.sqrt(jnp.diag(A))
    
    def get_apply_fn(self):
        def apply(x):
            return x / self.diag
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return x / self.diag
        return apply_T


class FixedDiagonalPreconditioner(Preconditioner):
    """Uses the inverse of the diagonal of a fixed matrix as a preconditioner."""

    def __init__(self, A, B):
        self.diag = jnp.sqrt(jnp.diag(B))
    
    def get_apply_fn(self):
        def apply(x):
            return x / self.diag
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return x / self.diag
        return apply_T


class ExactPreconditioner(Preconditioner):
    """Compute the inverse directly. Should enable 1 step convergence with CG."""

    def __init__(self, A):
        self.L = jnp.linalg.cholesky(jnp.linalg.inv(A))
    
    def get_apply_fn(self):
        def apply(x):
            return self.L @ x
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return self.L.T @ x
        return apply_T


class FixedMatrixPreconditioner(Preconditioner):
    """Ignore the original matrix and use a different matrix inverse."""

    def __init__(self, A, B):
        self.L = jnp.linalg.cholesky(jnp.linalg.inv(B))
    
    def get_apply_fn(self):
        def apply(x):
            return self.L @ x
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return self.L.T @ x
        return apply_T


class IdentityPreconditioner(Preconditioner):
    """Do nothing."""
    def __init__(self, A):
        pass

    def get_apply_fn(self):
        def apply(x):
            return x
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return x
        return apply_T


class IncompleteCholPreconditioner(Preconditioner):
    """Incomplete Cholesky based on SuperLU library. Not differentiable, JITtable, or GPU compatible."""

    def __init__(self, A, **kwargs):
        self.invA_approx = spilu(A, **kwargs)
        U = self.invA_approx.U
        diag_sqrts = np.sqrt(U.diagonal())

        # Want to scale columns of L.
        self.L = self.invA_approx.L @ diags(diag_sqrts)
    
    def get_apply_fn(self):
        def apply(x):
            return jnp.array(spsolve_triangular(self.L.T, x, lower=False))
        return apply
    
    def get_apply_T_fn(self):
        def apply_T(x):
            return jnp.array(spsolve_triangular(self.L, x, lower=True))
        return apply_T