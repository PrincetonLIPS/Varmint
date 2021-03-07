import jax
import jax.numpy        as np
import jax.numpy.linalg as npla
import numpy            as onp

import time


def generate_patch_free_energy(patch):
  """Generates a function that computes the free energy for a single patch.

  Assumes homogeneous patches.
  """

  jacobian_u_fn  = patch.get_cached_jacobian_u_fn()
  energy_fn      = patch.get_energy_fn()
  quad_fn        = patch.get_quad_fn()
  deformation_fn = patch.get_cached_deformation_fn()
  vmap_energy_fn = jax.vmap(energy_fn, in_axes=(0,))
  jac_dets_fn    = jax.vmap(npla.det, in_axes=(0,))

  defgrads_fn = jax.vmap(
    lambda A, B: npla.solve(B.T, A.T).T,
    in_axes=(0,0),
  )

  mat_density = patch.material.density
  gravity = 981.0 # cm/s^2

  def free_energy(def_ctrl, ref_ctrl):
    # Jacobian of reference config wrt parent config.
    def_jacs = jacobian_u_fn(def_ctrl)
    ref_jacs = jacobian_u_fn(ref_ctrl)

    # Deformation gradients. def_jacs @ ref_jacs_inv computed via a linear solve.
    defgrads = defgrads_fn(def_jacs, ref_jacs)

    # Jacobian determinants of reference config wrt parent.
    ref_jac_dets = jac_dets_fn(ref_jacs)

    # Strain energy density wrt to parent config.
    strain_energy_density = vmap_energy_fn(defgrads) * np.abs(ref_jac_dets)

    # Total potential energy via integrating over parent config.
    strain_potential = 1e3 * np.sum(quad_fn(strain_energy_density))

    # Mass density in parent config.
    mass_density = mat_density * np.abs(ref_jac_dets)

    # Positions in deformed config.
    positions = deformation_fn(def_ctrl)

    # Work density done by gravity.
    grav_energy_density = positions[:,1] * gravity * mass_density

    # Total work done by gravity integrated over parent config.
    gravity_potential = 1e-7 * np.sum(quad_fn(grav_energy_density))

    # Returning total energy here.
    return strain_potential + gravity_potential

  return free_energy


class DenseStaticsSolver:
  def __init__(self, cell):
    self.cell = cell

  def get_loss_fun(self):
    return self.cell.get_free_energy_fun(patchwise=False)

  def get_solver_fun(self, optimkind='newton', opt_params={}):
    niters = opt_params.get('niters', 10)

    loss_fun = jax.jit(self.get_loss_fun())
    grad_fun = jax.jit(jax.grad(self.get_loss_fun()))
    hess_fun = jax.jit(jax.hessian(self.get_loss_fun()))

    def solve(q, ref_ctrl):
      # Try pure Newton iterations
      print('Beginning optimization...')
      start_t = time.time()
      for i in range(niters):
        print(f'Loss: {loss_fun(q, ref_ctrl)}')
        hess = hess_fun(q, ref_ctrl)

        def hessp(p):
          return hess @ p

        direction = -jax.scipy.sparse.linalg.cg(hessp, grad_fun(q, ref_ctrl))[0]
        q = q + direction
      end_t = time.time()
      print(f'Finished optimization. Took {niters} steps in {end_t - start_t} seconds')

      return q

    return solve


class SparseStaticsSolver:
  def __init__(self, cell):
    self.cell = cell

  def get_loss_fun(self):
    p_free_energy = self.cell.get_free_energy_fun(patchwise=True)
    _, unflatten = self.cell.get_statics_flatten_unflatten()

    def loss(q, ref_ctrl):
      def_ctrl = unflatten(q, ref_ctrl)
      all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
      return np.sum(jax.vmap(lambda x: p_free_energy(x[..., 0], x[..., 1]))(all_args))

    return loss

  def get_solver_fun(self, optimkind='newton', opt_params={}):
    niters = opt_params.get('niters', 10)

    p_free_energy = self.cell.get_free_energy_fun(patchwise=True)
    flatten, unflatten = self.cell.get_statics_flatten_unflatten()
    flatten_add = self.cell.get_statics_flatten_add()

    loss_fun = jax.jit(self.get_loss_fun())
    grad_fun = jax.jit(jax.grad(self.get_loss_fun()))

    @jax.jit
    def block_hess_fn(q, ref_ctrl):
      def_ctrl = unflatten(q, ref_ctrl)
      all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
      return jax.vmap(lambda x: jax.hessian(p_free_energy)(x[..., 0], x[..., 1]))(all_args)

    def single_patch_hvp(patch_hess, patch_ctrl):
      flat_ctrl = patch_ctrl.ravel()
      ravel_len = flat_ctrl.shape[0]

      patch_hess = patch_hess.reshape((ravel_len, ravel_len))
      return (patch_hess @ flat_ctrl).reshape(patch_ctrl.shape)
    multi_patch_hvp = jax.vmap(single_patch_hvp, in_axes=(0, 0))

    def generate_hessp(q, ref_ctrl):
      block_hess = block_hess_fn(q, ref_ctrl)

      @jax.jit
      def hessp(p):
        unflat = unflatten(p, np.zeros_like(ref_ctrl))
        hvp_unflat = multi_patch_hvp(block_hess, unflat)
        return flatten_add(hvp_unflat)

      return hessp

    class MutableFunction:
      def __init__(self, func):
        self.func = func

      def __call__(self, p):
        return self.func(p)

    def solve(q, ref_ctrl):
      hessp = MutableFunction(generate_hessp(q, ref_ctrl))
      # Try pure Newton iterations
      print('Beginning optimization...')
      start_t = time.time()
      for i in range(niters):
        print(f'Loss: {loss_fun(q, ref_ctrl)}')
        hessp.func = generate_hessp(q, ref_ctrl)

        direction = -jax.scipy.sparse.linalg.cg(hessp, grad_fun(q, ref_ctrl))[0]
        q = q + direction
      end_t = time.time()
      print(f'Finished optimization. Took {niters} steps in {end_t - start_t} seconds')

      return q

    return solve
