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
from varmint.constitutive import NeoHookean2D
from varmint.bsplines     import default_knots
from varmint.lagrangian   import generate_lagrangian_structured
from varmint.discretize   import get_hamiltonian_stepper
from varmint.levmar       import get_lmfunc
from varmint.cellular2d   import match_labels, generate_quad_lattice

import jax.profiler

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

friction = 1e-7

npr.seed(5)

# Create patch parameters.
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

fixed_dict = dict(zip(labels[left_side], init_ctrl[left_side,:]))

# Create the shape.
shape = Shape2D(*[
  Patch2D(
    xknots,
    yknots,
    spline_deg,
    mat,
    quad_deg,
    labels[ii,:,:],
    fixed_dict
  )
  for  ii in range(len(init_ctrl))
])

friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

unflatten  = shape.get_unflatten_fn()
flatten    = shape.get_flatten_fn()
lagrangian = generate_lagrangian_structured(shape)
stepper    = get_hamiltonian_stepper(lagrangian, friction_force)

dt = 0.01
T  = 2.0

def simulate(refq):
  ref_ctrl, _ = unflatten(refq, np.zeros_like(refq), fixed_dict)

  # Initially in the ref config with zero momentum.
  q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

  QQ = [ q ]
  PP = [ p ]
  TT = [ 0.0 ]

  while TT[-1] < T:
    print(TT[-1])
    new_q, new_p = stepper(QQ[-1], PP[-1], dt, ref_ctrl, fixed_dict)

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1] + dt)

  return QQ

def radii_to_ctrl(radii):
  return generate_quad_lattice(widths, heights, radii)

def sim_radii(radii):

  # Construct reference shape.
  ref_ctrl = radii_to_ctrl(radii)

  refq, _ = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

  # Simulate the reference shape.
  QQ = simulate(refq)

  # Turn this into a sequence of control point sets.
  ctrl_seq = [ unflatten(q, np.zeros_like(q), fixed_dict)[0] for q in QQ ]

  return ctrl_seq

def loss(radii):
  ctrl_seq = sim_radii(radii)

  return -np.mean(ctrl_seq[-1]), ctrl_seq

#print(loss(init_radii))

valgrad_loss = jax.value_and_grad(loss, has_aux=True)

radii = init_radii

lr = 1.0
for ii in range(10):
  (val, ctrl_seq), gradmo = valgrad_loss(radii)
  print()
  #print(radii)
  print(val)
  print(gradmo)

  shape.create_movie(ctrl_seq, 'cell2-%d.mp4' % (ii+1), labels=False)

  radii = np.clip(radii - lr * gradmo, 0.05, 0.95)
