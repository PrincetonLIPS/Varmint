import jax
import jax.numpy as np
import jax.numpy.linalg as npla
import scipy.optimize as spopt

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

  def Ld(q1, q2, dt):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return dt * L(q, qdot)

  grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))

  return grad_Ld_q1, grad_Ld_q2

def get_hamiltonian_stepper(L):

  Ld_q1, Ld_q2 = discretize_hamiltonian(L)
  Ld_q1_jac = jax.jit(jax.jacfwd(Ld_q1, argnums=1))

  # TODO: need to figure this out for time dependence.
  @jax.custom_jvp
  def step_q(q, p, dt):

    res = spopt.least_squares(
      lambda q_1: p + Ld_q1(q, q_1, dt),
      q,
      lambda q_1: Ld_q1_jac(q, q_1, dt),
      method='lm',
    )
    # TODO: handle errors.

    return res.x

  @step_q.defjvp
  def step_q_jvp(primals, tangents):
    q, p, dt = primals
    q_tan, p_tan, dt_tan = tangents

    # Can I do this with the above version?
    res = spopt.least_squares(
      lambda q_1: p + Ld_q1(q, q_1, dt),
      q,
      lambda q_1: Ld_q1_jac(q, q_1, dt),
      method='lm',
    )
    # TODO: handle errors.

    new_q = res.x
    jac   = res.jac # verify

    old_q_tan = jvp(lambda old_q: Ld_q1(old_q, new_q, dt), (q,), q_tan)

    new_q_tan = - npla.solve(jac, old_q_tan + p_tan)

    return new_q, new_q_tan

  def step_p(q1, q2, dt):
    return Ld_q2(q1, q2, dt)

  def stepper(q, p, dt):
    new_q = step_q(q, p, dt)
    new_p = step_p(q, new_q, dt)
    return new_q, new_p

  return stepper
