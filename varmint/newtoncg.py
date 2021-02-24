import jax
import jax.numpy as np

def newtoncg(f, grad, hessp, xtol=1e-8, maxiter=None):
  # Adapted from Scipy source code
  # https://github.com/scipy/scipy/blob/d78daa50c2d462c32b62cfa8cd4bb03b5d9b1403/scipy/optimize/optimize.py#L1681
  # Variable names match Nocedal and Wright 2nd Edition Algorithm 7.1 (Line Search Newton-CG)
  # Currently uses very simple backtracking line search

  def optfun(x0, args):
    nparams = x0.shape[0]

    if maxiter is None:
      _maxiter = nparams * 200
    cg_maxiter = nparams * 20

    _xtol = nparams * xtol
    update = np.ones_like(x0) * 2 * _xtol
    x_k = x0

    k = 0
    float32eps = np.finfo(np.float32).eps

    def loop_cond(val):
      k, x_k, update = val
      return np.linalg.norm(update, ord=1) > _xtol

    def loop_body(val):
      k, x_k, update = val

      # Compute a search direction pk by applying the CG method to
      # def2 f(x_k) p = - grad f(x_k) starting from 0.
      b = -grad(x_k, args)
      maggrad = np.linalg.norm(b, ord=1)
      eta = np.min(np.asarray([0.5, np.sqrt(maggrad)]))
      termcond = eta * maggrad
      z_i = np.zeros_like(x0)

      r_i = -b
      d_i = -r_i
      riri_0 = r_i.T @ r_i
      i = 0

      def cg_cond(val):
        term, i, x_k, z_i, r_i, d_i, riri_0 = val
        return np.logical_not(term)

      def cg_iteration(val):
        term, i, x_k, z_i, r_i, d_i, riri_0 = val

        Ap = hessp(x_k, d_i, args)
        curv = d_i.T @ Ap

        def normal_iter(iter_val):
          Ap, curv, i, x_k, z_i, r_i, d_i, riri_0 = iter_val

          alpha_i = riri_0 / curv
          z_i1 = z_i + alpha_i * d_i
          r_i1 = r_i + alpha_i * Ap
          riri_1 = r_i1.T @ r_i1
          beta_i = riri_1 / riri_0
          d_i1 = -r_i1 + beta_i * d_i
          return (False, i+1, x_k, z_i1, r_i1, d_i1, riri_1)

        def update_break(iter_val):
          Ap, curv, i, x_k, z_i, r_i, d_i, riri_0 = iter_val

          z_i = riri_0 / (-curv) * b
          return (True, i, x_k, z_i, r_i, d_i, riri_0)

        return jax.lax.cond(
                    np.logical_or(
                        np.logical_or(
                            np.linalg.norm(r_i, ord=1) <= termcond,
                            np.logical_and(0 <= curv, curv <= 0.0003 * float32eps)
                        ), np.logical_and(curv < 0, i > 0)),
                    lambda _: (True, i, x_k, z_i, r_i, d_i, riri_0),
                    lambda _: jax.lax.cond(
                        np.logical_and(curv < 0, i == 0),
                        update_break,
                        normal_iter,
                        operand=(Ap, curv, i, x_k, z_i, r_i, d_i, riri_0)
                    ),
            operand=(Ap, curv, i, x_k, z_i, r_i, d_i, riri_0)
        )

      val = jax.lax.while_loop(cg_cond, cg_iteration, (False, 0, x_k, z_i, r_i, d_i, riri_0))
      print(f'finished cg loop with term: {val[0]} i: {val[1]} x_k: {x_k}')

      pk = val[3]

      # Line search with simple backtracking
      # TODO(doktay): Something fancy
      a_i = 1.0
      c_1 = 1e-4
      rho = 0.1

      slope = (-b).T @ pk
      f_k = f(x_k, args)

      def ls_cond(a_i):
        return f(x_k + a_i * pk, args) > f_k + c_1 * a_i * slope

      def ls_body(a_i):
        return a_i * rho

      a_i = jax.lax.while_loop(ls_cond, ls_body, a_i)

      update = a_i * pk
      x_k = x_k + update

      return k + 1, x_k, update

    val = jax.lax.while_loop(loop_cond, loop_body, (0, x_k, update))
    return val[1]
  
  return optfun
