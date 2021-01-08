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
server = jax.profiler.start_server(9999)

class WigglyMat(Material):
  _E = 0.001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

friction = 1e-6

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

friction_force = lambda q, qdot, ref_ctrl, fixed_dict: -friction * qdot

def displacement(t):
  return np.sin(4 * np.pi * t) * np.ones_like(init_ctrl)

unflatten  = shape.get_unflatten_fn()
flatten    = shape.get_flatten_fn()
lagrangian = generate_lagrangian_structured(shape)
stepper    = get_hamiltonian_stepper(lagrangian, friction_force)

dt = 0.01
T  = 1.0

def simulate(ref_ctrl):

  # Initially in the ref config with zero momentum.
  q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

  QQ = [ q ]
  PP = [ p ]
  TT = [ 0.0 ]

  while TT[-1] < T:

    t0 = time.time()
    fixed_locs = displacement(TT[-1]) + ref_ctrl


    success = False
    this_dt = dt
    while True:
      new_q, new_p = stepper(QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)
      success = np.all(np.isfinite(new_q))
      if success:
        break
      else:
        this_dt = this_dt / 2.0
        print('\tFailed to converge. dt now %f' % (this_dt))

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1] + this_dt)
    t1 = time.time()
    print(TT[-1], t1-t0)

  return QQ, PP, TT

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

#val, ctrl_seq = loss(init_radii)
#shape.create_movie(ctrl_seq, 'cell5.mp4', labels=False)

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
