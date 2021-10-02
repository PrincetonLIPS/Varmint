import jax
import jax.numpy as np

def check_grad(f, x, eps=1e-4):
  gradf = jax.grad(f)(x)
  fd = np.zeros_like(x)
  for ii in range(len(x)):

    up = jax.ops.index_add(x, ii, eps)
    upf = f(up)

    dn = jax.ops.index_add(x, ii, -eps)
    dnf = f(dn)

    fd = (upf-dnf)/(2.0*eps)
    print(ii, gradf[ii], fd, upf, dnf)


def map_jacfwd(f, N):
  """Do jacfwd with repeated jvp calls, except partially in sequence to save memory."""
  def mapjac(xk):
    def jvp_fun(v):
      return jax.jvp(f, (xk,), (v,))[1]
    
    all_vs = np.eye(N)
    reshaped_vs = all_vs.reshape((-1, 73, N))
    return jax.lax.map(jax.vmap(jvp_fun), reshaped_vs).reshape((N, N))
  return mapjac