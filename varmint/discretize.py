import jax
import time
import jax.numpy as np

from varmint.optimizers import get_optfun


def discretize_eulag(L):
  def Ld(q1, q2, dt):
    q    = (q1 + q2) / 2
    qdot = (q2 - q1) / dt
    return L(q, qdot)

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

  grad_Ld_q1 = jax.grad(Ld, argnums=0)
  grad_Ld_q2 = jax.grad(Ld, argnums=1)

  return grad_Ld_q1, grad_Ld_q2


class HamiltonianStepper:
  def __init__(self, L, F=None):
    # For thinking about forces, see West thesis:
    # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
    # Page 16, Sec 1.5.6.
    # Could include in optimization or in momentum update, I think.
    # Momentum update seems much easier.
    self.D0_Ld, self.D1_Ld = discretize_hamiltonian(L)
    self.F = F

  def residual_fun(self, new_q, args):
    old_q, p, dt, l_args = args

    if self.F is None:
      return p + self.D0_Ld(old_q, new_q, dt, l_args)
    else:
      q    = (old_q + new_q)/2.0
      qdot = (new_q-old_q) / dt

      return p + self.D0_Ld(old_q, new_q, dt, l_args) + self.F(q, qdot, *l_args)

  def construct_stepper(self, optimkind='levmar', opt_params={}):
    optfun = get_optfun(self.residual_fun, kind=optimkind, **opt_params)

    def step_q(q, p, dt, args):
      new_q = optfun(jax.lax.stop_gradient(q), (q, p, dt, args))
      return new_q

    if optimkind in ['levmar']:
      print(f'Using {optimkind} optimizer. Compiling optimizer.')
      step_q = jax.jit(step_q)
    else:
      print(f'Using {optimkind} optimizer. Does not support compilation yet..')

    def step_p(q1, q2, dt, args):
      if self.F is None:
        return self.D1_Ld(q1, q2, dt, args)
      else:
        q = (q1 + q2) / 2
        qdot = (q2 - q1) / dt

        return self.D1_Ld(q1, q2, dt, args) + self.F(q, qdot, *args)

    @jax.jit
    def update_p(new_q, q, p, dt, *args):
      return jax.lax.cond(
        np.all(np.isfinite(new_q)),
        lambda _: step_p(q, new_q, dt, args),
        lambda _: np.ones_like(p) + np.nan,
        np.float64(0.0),
      )

    def stepper(q, p, dt, *args):
      new_q = step_q(q, p, dt, args)
      new_p = update_p(new_q, q, p, dt, *args)

      return new_q, new_p

    return stepper


class SurrogateStepper:
  def __init__(self, L, F=None):
    # For thinking about forces, see West thesis:
    # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
    # Page 16, Sec 1.5.6.
    # Could include in optimization or in momentum update, I think.
    # Momentum update seems much easier.
    self.D0_Ld, self.D1_Ld = discretize_hamiltonian(L)
    self.F = F

  def construct_stepper(self, predict_fun, radii):
    def step_q(q, p, dt, args):
      _q = np.expand_dims(q, 0)
      _p = np.expand_dims(p, 0)
      _radii = radii.reshape((1, -1))

      new_q = predict_fun(_q, _p, _radii)
      new_q = np.squeeze(new_q, 0)
      return new_q
    step_q = jax.jit(step_q)

    def step_p(q1, q2, dt, args):
      if self.F is None:
        return self.D1_Ld(q1, q2, dt, args)
      else:
        q = (q1 + q2) / 2
        qdot = (q2 - q1) / dt

        return self.D1_Ld(q1, q2, dt, args) + self.F(q, qdot, *args)

    @jax.jit
    def update_p(new_q, q, p, dt, *args):
      return jax.lax.cond(
        np.all(np.isfinite(new_q)),
        lambda _: step_p(q, new_q, dt, args),
        lambda _: np.ones_like(p) + np.nan,
        np.float64(0.0),
      )

    def stepper(q, p, dt, *args):
      new_q = step_q(q, p, dt, args)
      new_p = update_p(new_q, q, p, dt, *args)

      return new_q, new_p

    return stepper


class SurrogateInitStepper:
  def __init__(self, L, F=None):
    # For thinking about forces, see West thesis:
    # https://thesis.library.caltech.edu/2492/1/west_thesis.pdf
    # Page 16, Sec 1.5.6.
    # Could include in optimization or in momentum update, I think.
    # Momentum update seems much easier.
    self.D0_Ld, self.D1_Ld = discretize_hamiltonian(L)
    self.F = F

  def residual_fun(self, new_q, args):
    old_q, p, dt, l_args = args

    if self.F is None:
      return p + self.D0_Ld(old_q, new_q, dt, l_args)
    else:
      q    = (old_q + new_q)/2.0
      qdot = (new_q-old_q) / dt

      return p + self.D0_Ld(old_q, new_q, dt, l_args) + self.F(q, qdot, *l_args)

  def construct_stepper(self, predict_fun, radii, optimkind='levmar', opt_params={}):
    optfun = get_optfun(self.residual_fun, kind=optimkind, **opt_params)

    def step_q(q, p, dt, args):
      _q = np.expand_dims(q, 0)
      _p = np.expand_dims(p, 0)
      _radii = radii.reshape((1, -1))

      new_q = predict_fun(_q, _p, _radii)
      new_q = np.squeeze(new_q, 0)

      # Use result of surrogate as initializer
      new_q = optfun(jax.lax.stop_gradient(new_q), (q, p, dt, args))
      return new_q

    if optimkind in ['levmar']:
      print(f'Using {optimkind} optimizer. Compiling optimizer.')
      step_q = jax.jit(step_q)
    else:
      print(f'Using {optimkind} optimizer. Does not support compilation yet..')

    def step_p(q1, q2, dt, args):
      if self.F is None:
        return self.D1_Ld(q1, q2, dt, args)
      else:
        q = (q1 + q2) / 2
        qdot = (q2 - q1) / dt

        return self.D1_Ld(q1, q2, dt, args) + self.F(q, qdot, *args)

    @jax.jit
    def update_p(new_q, q, p, dt, *args):
      return jax.lax.cond(
        np.all(np.isfinite(new_q)),
        lambda _: step_p(q, new_q, dt, args),
        lambda _: np.ones_like(p) + np.nan,
        np.float64(0.0),
      )

    def stepper(q, p, dt, *args):
      new_q = step_q(q, p, dt, args)
      new_p = update_p(new_q, q, p, dt, *args)

      return new_q, new_p

    return stepper
