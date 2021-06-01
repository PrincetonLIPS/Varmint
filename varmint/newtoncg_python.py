import jax
import jax.numpy as np

def newtoncg_python(f, grad, hessp, x0, args, xtol=1e-5, maxiter=None):
  # Adapted from scipy source code
  # https://github.com/scipy/scipy/blob/d78daa50c2d462c32b62cfa8cd4bb03b5d9b1403/scipy/optimize/optimize.py#L1681
  # Variable names match Nocedal and Wright 2nd Edition Algorithm 7.1 (Line Search Newton-CG)
  # Currently uses very simple backtracking line search

  nparams = x0.shape[0]

  if maxiter is None:
    maxiter = nparams * 200
  cg_maxiter = nparams * 20

  xtol = nparams * xtol
  update = np.array([2 * xtol])
  x_k = x0

  k = 0
  float32eps = np.finfo(np.float32).eps

  while np.linalg.norm(update, ord=1) > xtol:
    if k >= maxiter:
      print('Maximum iterations reached. Terminating.')
      return x_k

    # Compute a search direction pk by applying the CG method to
    # def2 f(x_k) p = - grad f(x_k) starting from 0.
    b = -grad(x_k, args)
    maggrad = np.linalg.norm(b, ord=1)
    print(maggrad)
    eta = np.min(np.asarray([0.5, np.sqrt(maggrad)]))
    termcond = eta * maggrad
    z_i = np.zeros_like(x0)

    r_i = -b
    d_i = -r_i
    riri_0 = r_i.T @ r_i
    i = 0

    for k2 in range(cg_maxiter):
      if np.linalg.norm(r_i, ord=1) <= termcond:
        print('hitting term cond 1')
        break
      #print(f'norm is {np.linalg.norm(r_i, ord=1)}')
      Ap = hessp(x_k, d_i, args)
      print(np.linalg.norm(Ap, ord=1))
      curv = d_i.T @ Ap
      if 0 <= curv <= 3 * float32eps:
        print(f'hitting term cond 2 with curv: {curv}')
        break
      elif curv < 0:
        if i > 0:
          print('hitting term cond 3')
          break
        else:
          z_i = riri_0 / (-curv) * b
          break
      
      alpha_i = riri_0 / curv
      z_i = z_i + alpha_i * d_i
      r_i = r_i + alpha_i * Ap
      riri_1 = r_i.T @ r_i
      beta_i = riri_1 / riri_0
      d_i = -r_i + beta_i * d_i
      riri_0 = riri_1
      i += 1
    else:
      print('CG did not converge. Hessian not positive definite.')
      return None
    pk = z_i
    print(f'z_i norm is {np.linalg.norm(z_i, ord=1)}')

    # Line search with simple backtracking
    # TODO(doktay): Something fancy
    a_i = 1.0
    c_1 = 1e-4
    rho = 0.1

    slope = (-b).T @ pk
    f_k = f(x_k, args)
    while f(x_k + a_i * pk, args) > f_k + c_1 * a_i * slope:
      print(a_i)
      a_i = a_i * rho
    print(f'chose line search step {a_i}')

    update = a_i * pk
    x_k = x_k + update
    k += 1
    print(f'update magnitude {np.linalg.norm(update, ord=1)}')
  return x_k
