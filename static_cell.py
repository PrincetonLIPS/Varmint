import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import scipy.sparse.linalg
import string

from functools import partial

from varmint.patch2d      import Patch2D
from varmint.shape2d      import Shape2D
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D, LinearElastic2D
from varmint.bsplines     import default_knots
from varmint.statics      import generate_free_energy_structured
from varmint.discretize   import get_hamiltonian_stepper
from varmint.levmar       import get_lmfunc
from varmint.cellular2d   import match_labels, generate_quad_lattice

#from varmint.grad_graph import grad_graph
#import jax.profiler
#server = jax.profiler.start_server(9999)

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

class CollapsingMat(Material):
  _E = 0.00001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(CollapsingMat)

npr.seed(5)

# Create patch parameters.c
quad_deg   = 10
spline_deg = 3
num_ctrl   = 5
num_x      = 4
num_y      = 4
xknots     = default_knots(spline_deg, num_ctrl)
yknots     = default_knots(spline_deg, num_ctrl)
widths     = 5*np.ones(num_x)
heights    = 5*np.ones(num_y)

#radii     = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
init_radii = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
init_ctrl  = generate_quad_lattice(widths, heights, init_radii)
labels     = match_labels(init_ctrl, keep_singletons=True)
left_side  = onp.array(init_ctrl[:,:,:,0] == 0.0)
bottom     = onp.array(init_ctrl[:,:,:,1] == 0.0)
fixed_labels = labels[bottom]

# Create the shape.
shape = Shape2D(*[
  Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    labels[ii,:,:],
    fixed_labels, # <-- Labels not locations
  )
  for  ii in range(len(init_ctrl))
])

unflatten  = shape.get_unflatten_fn()
flatten    = shape.get_flatten_fn()

free_energy = generate_free_energy_structured(shape)

def hvp(f, x, v):
  return jax.grad(lambda x: np.vdot(jax.grad(f)(x), v))(x)

def simulate(ref_ctrl):
  # Momentum is throwaway for statics.
  q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))
  fixed_locs = ref_ctrl
  new_q = q

  @jax.jit
  def loss_wrapped(new_q):
      return free_energy(new_q, p, ref_ctrl, fixed_locs)

  def callback(x):
      print('iteration')

  grad_q = jax.jit(jax.grad(loss_wrapped))
  #hess_q = jax.jit(jax.hessian(loss_wrapped))

  @jax.jit
  def hessp(new_q, p):
      return hvp(loss_wrapped, new_q, p)

#  for i in range(n_newton):
#    print(new_q.mean())
#    grad_q = jax.grad(free_energy, argnums=0)(new_q, p, ref_ctrl, fixed_locs)
#    hess_q = jax.hessian(free_energy, argnums=0)(new_q, p, ref_ctrl, fixed_locs)
#    
#    new_q = new_q - np.linalg.solve(hess_q, grad_q)

  optim = spopt.minimize(loss_wrapped, new_q, method='Newton-CG', jac=grad_q, hessp=hessp,
                         callback=callback, options={'disp': True})
  new_q = optim.x

  return unflatten(new_q, np.zeros_like(new_q), ref_ctrl)[0]

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

  def inner_loss(radii, ctrl):
    ref_ctrl = radii_to_ctrl(radii)

    q, p = flatten(ctrl, np.zeros_like(ctrl))
    fixed_locs = ref_ctrl

    return free_energy(q, p, ref_ctrl, fixed_locs)

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

  return loss_val, grad, ctrl_sol

def sample_loss_fn(ctrl):
  return np.mean(ctrl[..., 0])

def close_to_center_loss_fn(ctrl):
  return np.linalg.norm(ctrl[..., 0] - 10)

radii = init_radii

lr = 1.0
for ii in range(10):
  loss_val, loss_grad, ctrl_sol = loss_and_adjoint_grad(close_to_center_loss_fn, radii)
  print()
  #print(radii)
  print(loss_val)

  shape.create_movie([ctrl_sol], 'center-static-cell5-%d.mp4' % (ii+1), labels=False)

  radii = np.clip(radii - lr * loss_grad, 0.05, 0.95)
