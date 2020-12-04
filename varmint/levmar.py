import jax
import jax.numpy         as np
import jax.numpy.linalg  as npla
import matplotlib.pyplot as plt

from collections    import namedtuple
from scipy.optimize import OptimizeResult

import jax.profiler

# Moré, J.J., 1978. The Levenberg-Marquardt algorithm: implementation
# and theory. In Numerical analysis (pp. 105-116). Springer, Berlin,
# Heidelberg.

def p_solve(lamb, diagD, Ux, diagSx, Vx, Fx):
  mid_solve = np.diag(diagSx / (diagSx**2 + lamb))
  p = -(Vx @ mid_solve @ Ux.T @ Fx) / diagD
  return p
def phi(alpha, diagD, Ux, diagSx, Vx, Fx, delta):
  p = p_solve(alpha, diagD, Ux, diagSx, Vx, Fx)
  return npla.norm(diagD * p) - delta, p
phi_valgrad = jax.jit(jax.value_and_grad(phi, argnums=0, has_aux=True))

@jax.jit
def lm_param(delta, diagD, Ux, diagSx, Vx, Fx, sigma=0.1):
  (phi_0, p), dphi_0 = phi_valgrad(0.0, diagD, Ux, diagSx, Vx, Fx, delta)

  # Initial bounds as in More'.
  upper = npla.norm((Fx.T @ Ux) * diagSx) / delta
  lower = - phi_0 / dphi_0
  alpha = np.array(0.0)

  init_val = (alpha, lower, upper, phi_0, p)

  def cond_fun(val):
    alpha, lower, upper, phi_k, p = val
    return np.logical_and(np.logical_or(alpha != 0.0, phi_k > 0.0), np.abs(phi_k) > sigma*delta)

  def body_fun(val):
    alpha, lower, upper, phi_k, p = val

    alpha = np.where(np.logical_or(alpha <= lower, alpha >= upper),
                     np.maximum( 0.001 * upper, np.sqrt(upper * lower) ),
                     alpha)

    (phi_k, p), dphi_k = phi_valgrad(alpha, diagD, Ux, diagSx, Vx, Fx, delta)

    upper = np.where(phi_k < 0.0, alpha, upper)
    lower = np.maximum(lower, alpha - phi_k/dphi_k)
    alpha = alpha - ( (phi_k + delta)/delta ) * ( phi_k/dphi_k )

    return (alpha, lower, upper, phi_k, p)

  alpha, _, _, _, p = jax.lax.while_loop(cond_fun, body_fun, init_val)

  return alpha, p

LMState = namedtuple('LMState', [
  'x', 'args', 'Fx', 'nFx', 'Jx', 'Ux', 'diagSx', 'VxT', 'diagD',
  'delta', 'hit_xtol', 'hit_ftol', 'nit', 'nfev', 'njev'
])


def get_lmfunc(
    fun,
    maxiters=100,
    xtol=1e-8,
    ftol=1e-8,
    factor=100.0,
    sigma=0.1,
):

  # Compute and jit the Jacobian function.
  jacfun = jax.jit(jax.jacfwd(fun))

  @jax.jit
  def cond_fun(state):
    return np.logical_not(np.logical_or(
        np.logical_or(state.hit_xtol, state.hit_ftol),
        state.nit >= maxiters,
    ))

  @jax.jit
  def body_fun(state):

    # Compute lambda.
    lamb, p = lm_param(
      state.delta,
      state.diagD,
      state.Ux,
      state.diagSx,
      state.VxT.T,
      state.Fx,
      sigma,
    )
    sqrt_lamb = np.sqrt(lamb)

    # Evaluate the function at this new location.
    new_x   = state.x + p
    new_Fx  = fun(new_x, state.args)
    new_nFx = npla.norm(new_Fx)

    # Track function evaluations.
    # FIXME do this later.
    # new_nfev = state.nfev + 1

    # Compute rho, the measure of prediction accuracy: More' Eqn. 4.4
    nJp = npla.norm(state.Jx @ p)
    nDp = npla.norm(state.diagD * p)
    rho = (1-(new_nFx/state.nFx)**2) / \
      ((nJp/state.nFx)**2 + 2*((sqrt_lamb*nDp)/state.nFx)**2)

    # Have we achived ftol? Look at this with the new norms.
    hit_ftol = (nJp/state.nFx)**2 + 2*(sqrt_lamb*(nDp/state.nFx))**2 <= ftol

    # SLOW: we only need these if rho <= 0.25.
    # Compute gamma via More' (above Eqn. 4.5), should be in [-1,0]
    gamma = -((nJp/state.nFx)**2 + (sqrt_lamb*(nDp/state.nFx))**2)
    # Compute mu via More' Eqn. 4.5.
    mu = np.clip(0.5*gamma / (gamma + 0.5*(1 - (new_nFx/state.nFx)**2)), 0.1, 0.5)

    delta = jax.lax.cond(
        rho <= 0.25,
        lambda _: mu * state.delta,
        lambda _: jax.lax.cond(
            np.logical_or(np.logical_and(rho <= 0.75, lamb == 0.0), rho > 0.75),
            lambda _: 2 * nDp,
            lambda _: state.delta,
            None,
        ),
        None,
    )

    # Have we achieved xtol? Look at this after changing delta.
    hit_xtol = delta <= xtol * npla.norm(state.diagD * new_x)

    improved = rho > 0.0001

    # This becomes the new state if we improved.
    x, Fx, nFx, Jx, njev = jax.lax.cond(
        improved,
        lambda _: (new_x, new_Fx, new_nFx, jacfun(new_x, state.args), state.njev+1),
        lambda _: (state.x, state.Fx, state.nFx, state.Jx, state.njev),
        None,
    )

    # Update the scaling if we improved.
    diagD = jax.lax.cond(
        improved,
        lambda _: np.maximum(state.diagD, npla.norm(Jx, axis=0)),
        lambda _: state.diagD,
        None,
    )

    # Recompute the SVD if we improved.
    Ux, diagSx, VxT = jax.lax.cond(
        improved,
        lambda _: npla.svd(Jx/diagD, full_matrices=False),
        lambda _: (state.Ux, state.diagSx, state.VxT),
        None,
    )

    # Return the full state.
    return LMState(
      x        = x,
      args     = state.args,
      Fx       = Fx,
      nFx      = nFx,
      Jx       = Jx,
      Ux       = Ux,
      diagSx   = diagSx,
      VxT      = VxT,
      diagD    = diagD,
      delta    = delta,
      hit_xtol = hit_xtol,
      hit_ftol = hit_ftol,
      nit      = state.nit + 1,
      nfev     = state.nfev + 1,
      njev     = njev,
    )

  @jax.jit
  def optfun(x0, args):
    # Initialize counts.
    nit  = 0
    nfev = 1
    njev = 1

    # Initialize.
    Fx    = fun(x0, args)
    nFx   = npla.norm(Fx)
    Jx    = jacfun(x0, args)
    diagD = npla.norm(Jx, axis=0)
    delta = factor * npla.norm(diagD * x0)
    delta = jax.lax.cond(delta == 0, lambda _: factor, lambda _: delta, None)
    Ux, diagSx, VxT = npla.svd(Jx/diagD, full_matrices=False)
    hit_xtol  = False
    hit_ftol  = False

    init_state = LMState(
      x        = x0,
      args     = args,
      Fx       = Fx,
      nFx      = nFx,
      Jx       = Jx,
      Ux       = Ux,
      diagSx   = diagSx,
      VxT      = VxT,
      diagD    = diagD,
      delta    = delta,
      hit_xtol = hit_xtol,
      hit_ftol = hit_ftol,
      nit      = nit,
      nfev     = nfev,
      njev     = njev,
    )

    return jax.lax.while_loop(
        cond_fun, body_fun, init_state,
    )

  return optfun

import timeit
import numpy.random as npr

@jax.jit
def p1(x, z):
  theta = (1/(2*np.pi)) * np.arctan(x[1]/x[0])
  theta = np.where(x[0] < 0, theta + 0.5, theta)
  return np.array([
    10*(x[2] - 10*theta),
    10*(np.sqrt(x[0]**2 + x[1]**2) - 1),
    x[2],
  ])*z
jac_p1 = jax.jit(jax.jacfwd(p1))
x0 = np.array([-1., 0., 0.])


optfun = get_lmfunc(p1)
res = optfun(x0,1.0)
print(res.nFx, res.x)

print(timeit.repeat(lambda :optfun(x0, npr.rand()), repeat=5, number=1))


@jax.jit
def p4(x, z):
  ti = (np.arange(20)+1.0) * 0.2 * z
  fi = (x[0] + x[1]*ti - np.exp(ti))**2 + (x[2] + x[3]*np.sin(ti) - np.cos(ti))**2
  return fi
jac_p4 = jax.jit(jax.jacfwd(p4))

optfun = get_lmfunc(p4)
print(optfun(x0, 1.0).nFx)


print(timeit.repeat(lambda :optfun(x0, 1.0), repeat=5, number=1))
