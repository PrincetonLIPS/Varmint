import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import string

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

mat = LinearElastic2D(WigglyMat)

npr.seed(5)

# Create patch parameters.c
quad_deg   = 10
spline_deg = 3
num_ctrl   = 5
num_x      = 3
num_y      = 1
xknots     = default_knots(spline_deg, num_ctrl)
yknots     = default_knots(spline_deg, num_ctrl)
widths     = 5*np.ones(num_x)
heights    = 5*np.ones(num_y)

#radii     = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
init_radii = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
init_ctrl  = generate_quad_lattice(widths, heights, init_radii)
labels     = match_labels(init_ctrl, keep_singletons=True)
left_side  = onp.array(init_ctrl[:,:,:,0] == 0.0)
fixed_labels = labels[left_side]

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

def simulate(ref_ctrl, n_newton=5):
  # Momentum is throwaway for statics.
  q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))
  fixed_locs = ref_ctrl
  new_q = q
  for i in range(n_newton):
    print(new_q.mean())
    grad_q = jax.grad(free_energy, argnums=0)(new_q, p, ref_ctrl, fixed_locs)
    hess_q = jax.hessian(free_energy, argnums=0)(new_q, p, ref_ctrl, fixed_locs)
    
    new_q = new_q - np.linalg.inv(hess_q) @ grad_q

  return [unflatten(new_q, np.zeros_like(new_q), ref_ctrl)[0]]

# Since we're simulating linear elasticity, a single Newton iteration is enough.
ctrl_seq = simulate(init_ctrl, n_newton=1)
shape.create_movie(ctrl_seq, 'statics_test.mp4', labels=False)

quit()

def radii_to_ctrl(radii):
  return generate_quad_lattice(widths, heights, radii)

def sim_radii(radii):

  # Construct reference shape.
  ref_ctrl = radii_to_ctrl(radii)

  # Simulate the reference shape.
  QQ, PP, TT = simulate(ref_ctrl)

  # Turn this into a sequence of control point sets.
  ctrl_seq = [
    unflatten(
      qt[0],
      np.zeros_like(qt[0]),
      ref_ctrl + displacement(qt[1]),
    )[0] \
    for qt in zip(QQ, TT)
  ]

  return ctrl_seq

def loss(radii):
  ctrl_seq = sim_radii(radii)

  return -np.mean(ctrl_seq[-1]), ctrl_seq


val, ctrl_seq = loss(init_radii)

shape.create_movie(ctrl_seq, 'cell5.mp4', labels=False)

quit()

valgrad_loss = jax.value_and_grad(loss, has_aux=True)

radii = init_radii

lr = 1.0
for ii in range(5):
  (val, ctrl_seq), gradmo = valgrad_loss(radii)
  print()
  #print(radii)
  print(val)
  print(gradmo)

  shape.create_movie(ctrl_seq, 'cell5-%d.mp4' % (ii+1), labels=False)

  radii = np.clip(radii - lr * gradmo, 0.05, 0.95)
