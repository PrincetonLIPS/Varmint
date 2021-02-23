import jax
import time
import jax.numpy as np
import jax.numpy.linalg as npla
import scipy.optimize as spopt

from varmint.levmar import get_lmfunc
from varmint.optimizers import get_optfun

def discretize_eulag(L):

  def Ld(q1, q2, dt):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return L(q, qdot)

  #grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  #grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))
  grad_Ld_q1 = jax.grad(Ld, argnums=0)
  grad_Ld_q2 = jax.grad(Ld, argnums=1)

  def DEL(q1, t1, q2, t2, q3, t3):
    return grad_Ld_q1(q2, q3, t3-t2) + grad_Ld_q2(q1, q2, t2-t1)

  return DEL

def discretize_hamiltonian(L):

  def Ld(q1, q2, dt, args):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return L(q, qdot, *args)

  #grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  #grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))
  grad_Ld_q1 = jax.grad(Ld, argnums=0)
  grad_Ld_q2 = jax.grad(Ld, argnums=1)

  return grad_Ld_q1, grad_Ld_q2

def hvp(f, x, v, args):
  return jax.grad(lambda x, args: np.vdot(jax.grad(f)(x, args), v))(x, args)

def get_hamiltonian_stepper(L, F=None, return_residual=False,
                               surrogate=None, optimkind=None):

  # For thinking about forces, see West thesis:
  # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
  # Page 16, Sec 1.5.6.
  # Could include in optimization or in momentum update, I think.
  # Momentum update seems much easier.

  D0_Ld, D1_Ld = discretize_hamiltonian(L)

  def residual_fun(new_q, args):
    old_q, p, dt, l_args = args

    if F is None:
      return p + D0_Ld(old_q, new_q, dt, l_args)
    else:

      q    = (old_q + new_q)/2.0
      qdot = (new_q-old_q) / dt

      return p + D0_Ld(old_q, new_q, dt, l_args) + F(q, qdot, *l_args)

  jac_fun = jax.jacfwd(residual_fun)
  optfun = get_optfun(residual_fun, kind=optimkind, maxiters=50)

  def step_q(q, p, dt, args):
    new_q = optfun(jax.lax.stop_gradient(q), (q, p, dt, args),
                   jac=jac_fun, jacp=None, hess=None, hessp=None)
    return new_q

  def step_p(q1, q2, dt, args):
    if F is None:
      return D1_Ld(q1, q2, dt, args)
    else:
      q = (q1 + q2) / 2
      qdot = (q2 - q1) / dt

      return D1_Ld(q1, q2, dt, args) + F(q, qdot, *args)

  @jax.jit
  def update_p(new_q, q, p, dt, *args):
    return jax.lax.cond(
      np.all(np.isfinite(new_q)),
      lambda _: step_p(q, new_q, dt, args),
      lambda _: np.ones_like(p) + np.nan,
      np.float32(0.0),
    )

  #@jax.jit
  def stepper(q, p, dt, *args):
    if surrogate != None:
      new_q = surrogate(q, p)
    else:
      new_q = step_q(q, p, dt, args)

    # This seems to get recompiled over and over again?
    new_p = update_p(new_q, q, p, dt, *args)
    #new_p = jax.lax.cond(
    #  np.all(np.isfinite(new_q)),
    #  lambda _: step_p(q, new_q, dt, args),
    #  lambda _: np.ones_like(p) + np.nan,
    #  np.float32(0.0),
    #)

    return new_q, new_p

  if return_residual:
    return stepper, residual_fun
  else:
    return stepper
