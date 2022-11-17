import jax
import jax.numpy as np
import jax.numpy.linalg as npla


vmap_dot = jax.jit(
    jax.vmap(
        np.dot,
        in_axes=(0, 0),
    ),
)

vmap_lsolve = jax.jit(
    jax.vmap(
        npla.solve,
        in_axes=(0, 0),
    ),
)

vmap_rsolve = jax.jit(
    jax.vmap(
        lambda A, B: npla.solve(B.T, A.T).T,
        in_axes=(0, 0),
    ),
)

vmap_tsolve = jax.jit(
    jax.vmap(
        lambda A, B: npla.solve(A.T, B),
        in_axes=(0, 0),
    ),
)

vmap_tensordot = jax.jit(
    jax.vmap(
        np.tensordot,
        in_axes=(0, 0, None),
    ),
    static_argnums=(2,)
)

vmap_inv = jax.jit(
    jax.vmap(
        npla.inv,
        in_axes=(0,),
    ),
)

vmap_det = jax.jit(
    jax.vmap(
        npla.det,
        in_axes=(0,),
    ),
)

vmap_diag = jax.jit(
    jax.vmap(
        np.diag,
        in_axes=(0,),
    ),
)

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
        reshaped_vs = all_vs.reshape((-1, 1, N))
        return jax.lax.map(jax.vmap(jvp_fun), reshaped_vs).reshape((N, N))
    return mapjac
