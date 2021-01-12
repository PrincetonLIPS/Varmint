import jax
import time
import jax.numpy as np
import jax.numpy.linalg as npla
import scipy.optimize as spopt

from varmint.levmar import get_lmfunc

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

def get_hamiltonian_stepper(L, F=None):

  # For thinking about forces, see West thesis:
  # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
  # Page 16, Sec 1.5.6.
  # Could include in optimization or in momentum update, I think.
  # Momentum update seems much easier.

  D0_Ld, D1_Ld = discretize_hamiltonian(L)

  # Trying to track down the source of the 12M files.
  # @jax.jit
  def residual_fun(new_q, args):
    old_q, p, dt, l_args = args

    if F is None:
      return p + D0_Ld(old_q, new_q, dt, l_args)
    else:

      q    = (old_q + new_q)/2.0
      qdot = (new_q-old_q) / dt

      return p + D0_Ld(old_q, new_q, dt, l_args) + F(q, qdot, *l_args)

  optfun = get_lmfunc(residual_fun, maxiters=50)

  #@jax.jit
  def step_q(q, p, dt, args):
    new_q = optfun(jax.lax.stop_gradient(q), (q, p, dt, args))

    # new_q, res = optfun(jax.lax.stop_gradient(q), (q, p, dt, args))
    # print(res.nFx, res.nit, res.nfev, res.njev)
    return new_q

  #@jax.jit
  def step_p(q1, q2, dt, args):

    if F is None:
      return D1_Ld(q1, q2, dt, args)
    else:

      q = (q1 + q2) / 2
      qdot = (q2 - q1) / dt

      return D1_Ld(q1, q2, dt, args) + F(q, qdot, *args)

  @jax.jit
  def stepper(q, p, dt, *args):
    #t0 = time.time()
    new_q = step_q(q, p, dt, args)

    #t1 = time.time()

    # This seems to get recompiled over and over again?
    new_p = jax.lax.cond(
      np.all(np.isfinite(new_q)),
      lambda _: step_p(q, new_q, dt, args),
      lambda _: np.ones_like(p) + np.nan,
      np.float32(0.0),
    )
    #t2 = time.time()
    # print("\tstep_q = %0.2fs  step_p = %0.2fs" % (t1-t0, t2-t1))
    return new_q, new_p

  return stepper
