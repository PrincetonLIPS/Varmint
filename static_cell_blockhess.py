import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import scipy.sparse.linalg
import string

import jax.profiler

from functools import partial

from varmint.patch2d      import Patch2D
from varmint.shape2d      import Shape2D
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D, LinearElastic2D
from varmint.bsplines     import default_knots
from varmint.statics      import generate_patch_free_energy
from varmint.cellular2d   import index_array_from_ctrl, generate_quad_lattice

#from jax.config import config
#config.update("jax_enable_x64", True)

#from varmint.grad_graph import grad_graph
#import jax.profiler
#server = jax.profiler.start_server(9999)

class WigglyMat(Material):
  _E = 0.0005
  _nu = 0.48
  _density = 1.0

class CollapsingMat(Material):
  _E = 0.00001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)
#mat = NeoHookean2D(SiliconeRubber)

npr.seed(5)

# Create patch parameters.c
quad_deg   = 10
spline_deg = 3
num_ctrl   = 5
num_x      = 20
num_y      = 60
xknots     = default_knots(spline_deg, num_ctrl)
yknots     = default_knots(spline_deg, num_ctrl)
widths     = 5*np.ones(num_x)
heights    = 5*np.ones(num_y)

print('Generating radii and control points')
#radii     = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
init_radii = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
init_ctrl  = generate_quad_lattice(widths, heights, init_radii)
n_components, index_arr = index_array_from_ctrl(num_x, num_y, init_ctrl)
left_side  = onp.array(init_ctrl[:,:,:,0] == 0.0)
bottom     = onp.array(init_ctrl[:,:,:,1] == 0.0)
fixed_labels = index_arr[bottom]

def flatten_add(unflat_ctrl):
  almost_flat = jax.ops.index_add(np.zeros((n_components, 2)), index_arr, unflat_ctrl)
  return almost_flat.flatten()

def flatten(unflat_ctrl):
  almost_flat = jax.ops.index_update(np.zeros((n_components, 2)), index_arr, unflat_ctrl)
  return almost_flat.flatten()

fixed_locations = flatten(init_ctrl).reshape((n_components, 2))
fixed_locations = np.take(fixed_locations, fixed_labels, axis=0)

def unflatten(flat_ctrl, fixed_locs):
  flat_ctrl = flat_ctrl.reshape(n_components, 2)
  fixed     = jax.ops.index_update(flat_ctrl, fixed_labels, fixed_locs)
  return np.take(fixed, index_arr, axis=0)

def unflatten_nofixed(flat_ctrl):
  flat_ctrl = flat_ctrl.reshape(n_components, 2)
  return np.take(flat_ctrl, index_arr, axis=0)

print('Creating shape')
# Create the shape.
shape = Shape2D(*[
  Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    None, #labels[ii,:,:],
    fixed_labels, # <-- Labels not locations
  )
  for  ii in range(len(init_ctrl))
])

patch = Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    None, #labels[ii,:,:],
    fixed_labels, # <-- Labels not locations
)

#free_energy = generate_patch_free_energy(shape)
free_energy = generate_patch_free_energy(patch)

def hvp(f, x, v):
  return jax.grad(lambda x: np.vdot(jax.grad(f)(x), v))(x)

def simulate(ref_ctrl):
  q = flatten(ref_ctrl)
  new_q = q

  def loss_wrapped(new_q):
    def_ctrl = unflatten(new_q, fixed_locations)
    all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
    return np.sum(jax.vmap(lambda x: free_energy(x[..., 0], x[..., 1]))(all_args))

  loss_q = jax.jit(loss_wrapped)
  grad_q = jax.jit(jax.grad(loss_wrapped))

  @jax.jit
  def block_hess_fn(new_q):
    def_ctrl = unflatten(new_q, fixed_locations)
    all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
    return jax.vmap(lambda x: jax.hessian(free_energy)(x[..., 0], x[..., 1]))(all_args)

  def single_patch_hvp(patch_hess, patch_ctrl):
    flat_ctrl = patch_ctrl.ravel()
    ravel_len = flat_ctrl.shape[0]

    patch_hess = patch_hess.reshape((ravel_len, ravel_len))
    return (patch_hess @ flat_ctrl).reshape(patch_ctrl.shape)
  multi_patch_hvp = jax.vmap(single_patch_hvp, in_axes=(0, 0))
  
  def generate_hessp(x):
    block_hess = block_hess_fn(x)

    @jax.jit
    #def hessp(new_q, p):
    def hessp(p):
      unflat = unflatten(p, np.zeros_like(fixed_locations))
      hvp_unflat = multi_patch_hvp(block_hess, unflat)
      flattened = flatten_add(hvp_unflat).reshape(-1, 2)
      return jax.ops.index_update(flattened, fixed_labels, 0.).flatten()

    return hessp

  class MutableFunction:
    def __init__(self, func):
      self.func = func
      self.total_calls = 0
      self.last_printed = time.time()

    #def __call__(self, new_q, p):
    def __call__(self, p):
      self.total_calls += 1
      if time.time() - self.last_printed > 2.0:
          print(f'called hvp {self.total_calls} times')
          self.last_printed = time.time()

      return self.func(p)

  hessp = MutableFunction(generate_hessp(new_q))

  #@jax.jit
  #def hessp(new_q, p):
  #  return hvp(loss_wrapped, new_q, p)

  def callback(x):
    print('iteration. updating hessian.')
    hessp.func = generate_hessp(x)

  # Precompile
  loss_q(new_q)
  grad_q(new_q)
  #hessp(new_q, new_q)
  hessp(new_q)

  # Try pure Newton iterations
  print('starting optimization')
  start_t = time.time()
  for i in range(10):
    print('newton iteration')
    print(f'loss: {loss_q(new_q)}')
    direction = -jax.scipy.sparse.linalg.cg(hessp, grad_q(new_q))[0]
    new_q = new_q + direction
    hessp.func = generate_hessp(new_q)
  end_t = time.time()
  print(f'optimization took {end_t - start_t} seconds')

  #print('starting optimization')
  #start_t = time.time()
  #optim = spopt.minimize(loss_q, new_q, method='trust-ncg', jac=grad_q, hessp=hessp,
  #                       callback=callback, options={'disp': True})
  #end_t = time.time()
  #print(f'optimization took {end_t - start_t} seconds')

  #new_q = optim.x

  return unflatten(new_q, fixed_locations)

# Since we're simulating linear elasticity, a single Newton iteration is enough.
#ctrl_seq = simulate(init_ctrl)
#shape.create_movie([ctrl_seq], 'statics_test.mp4', labels=False)

def radii_to_ctrl(radii):
  return generate_quad_lattice(widths, heights, radii)

def loss_and_adjoint_grad(loss_fn, init_radii):
  # loss_fn should be a function of ctrl_seq
  grad_loss = jax.jit(jax.grad(loss_fn))
  ctrl_sol = simulate(radii_to_ctrl(init_radii))
  dJdu = grad_loss(ctrl_sol)

  def inner_loss(radii, def_ctrl):
    ref_ctrl = radii_to_ctrl(radii)

    # So that fixed control points work out. This is hacky.
    flat     = flatten(def_ctrl)
    unflat   = unflatten(flat, fixed_locations)

    all_args = np.stack([def_ctrl, ref_ctrl], axis=-1)
    return np.sum(jax.vmap(lambda x: free_energy(x[..., 0], x[..., 1]))(all_args))

  loss_val = loss_fn(ctrl_sol)
  implicit_fn = jax.jit(jax.grad(inner_loss, argnums=1))
  implicit_vjp = jax.jit(jax.vjp(implicit_fn, init_radii, ctrl_sol)[1])

  def vjp_ctrl(v):
    return implicit_vjp(v)[1]
  
  def vjp_radii(v):
    return implicit_vjp(v)[0]
  
  flat_size = ctrl_sol.flatten().shape[0]
  unflat_size = ctrl_sol.shape

  def spmatvec(v):
    v = v.reshape(ctrl_sol.shape)
    vjp = implicit_vjp(v)[1]
    return vjp.flatten()

  A = scipy.sparse.linalg.LinearOperator((flat_size,flat_size), matvec=spmatvec)

  # Precomputing full Jacobian might be better
  #print('precomputing hessian')
  #hess = jax.jacfwd(implicit_fn, argnums=1)(init_radii, ctrl_sol)
  #print(f'computed hessian with shape {hess.shape}')
  
  print('solving adjoint equation')
  
  adjoint, info = scipy.sparse.linalg.minres(A, dJdu.flatten())
  adjoint = adjoint.reshape(unflat_size)
  grad = vjp_radii(adjoint)

  return loss_val, -grad, ctrl_sol

def sample_loss_fn(ctrl):
  return np.mean(ctrl[..., 0])

def close_to_center_loss_fn(ctrl):
  return np.linalg.norm(ctrl[..., 0] - 10)

print('Starting statics simulation')
radii = init_radii
ctrl_sol = simulate(radii_to_ctrl(radii))
#shape.create_movie([ctrl_sol], 'long-hanging-cells-static.mp4', labels=False)
print('Saving result in figure.')
shape.create_static_image(ctrl_sol, 'tall-cells-static-just-cp.png', just_cp=True)

quit()

print('Starting training')
lr = 1.0
for ii in range(1):
  loss_val, loss_grad, ctrl_sol = loss_and_adjoint_grad(sample_loss_fn, radii)
  print()
  print(loss_val)

  shape.create_movie([ctrl_sol], 'long-hanging-cells-%d.mp4' % (ii+1), labels=False)

  radii = np.clip(radii - lr * loss_grad, 0.05, 0.95)
