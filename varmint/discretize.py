import jax
import jax.numpy as np
import jax.numpy.linalg as npla
import scipy.optimize as spopt

from varmint.levmar import get_lmfunc

def discretize_eulag(L):

  def Ld(q1, q2, dt):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return dt * L(q, qdot)

  grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))

  def DEL(q1, t1, q2, t2, q3, t3):
    return grad_Ld_q1(q2, q3, t3-t2) + grad_Ld_q2(q1, q2, t2-t1)

  return DEL

def discretize_hamiltonian(L):

  def Ld(q1, q2, dt, args):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return dt * L(q, qdot, *args)

  grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))

  return grad_Ld_q1, grad_Ld_q2

def get_hamiltonian_stepper(L):

  D0_Ld, D1_Ld = discretize_hamiltonian(L)

  @jax.jit
  def residual_fun(new_q, args):
    old_q, p, dt, l_args = args
    return p + D0_Ld(old_q, new_q, dt, l_args)

  optfun = get_lmfunc(residual_fun, maxiters=200)

  def step_q(q, p, dt, args):
    new_q, res = optfun(jax.lax.stop_gradient(q), (q, p, dt, args))
    print(res.nFx, res.nit, res.nfev, res.njev)
    return new_q

  def step_p(q1, q2, dt, args):
    return D1_Ld(q1, q2, dt, args)

  def stepper(q, p, dt, *args):
    new_q = step_q(q, p, dt, args)
    new_p = step_p(q, new_q, dt, args)
    return new_q, new_p

  return stepper
