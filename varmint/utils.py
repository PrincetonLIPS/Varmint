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
