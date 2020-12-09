import jax

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
