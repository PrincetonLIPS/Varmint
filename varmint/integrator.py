
# Just the basic thing here...
# Could be prettier with tuples.
# Will want to include forces.
def construct_DEL(lagrangian):

  def Ld(q1, q2, t1, t2):
    d_q    = (q1 + q2) / 2
    d_qdot = (q2 - q1) / (t2 - t1)
    return L(d_q, d_qdot)

  grad_Ld_q1 = jax.jit(jax.grad(Ld, argnums=0))
  grad_Ld_q2 = jax.jit(jax.grad(Ld, argnums=1))

  def DEL(q1, q2, q3, t1, t2, t3):
    return grad_Ld_q1( q2, q3, t2, t3) + grad_Ld_q2( q1, q2, t1, t2)
