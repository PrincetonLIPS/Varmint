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
left_side  = init_ctrl[:,:,:,0] == 0.0
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

unflatten  = shape.get_unflatten_fn()
flatten    = shape.get_flatten_fn()
lagrangian, strain_forces = generate_lagrangian_structured(shape)
stepper    = get_hamiltonian_stepper(lagrangian, strain_forces)

dt = 0.01
T  = 1.0

def radii_to_ctrl(radii):
  return generate_quad_lattice(widths, heights, init_radii)

def simulate(ref_ctrl):

  # Initially in the ref config with zero momentum.
  q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

  QQ = [ q ]
  PP = [ p ]
  TT = [ 0.0 ]

  while TT[-1] < T:
    print(TT[-1])
    new_q, new_p = stepper(QQ[-1], PP[-1], dt, ref_ctrl)

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1] + dt)

  return QQ

QQ = simulate(radii_to_ctrl(init_radii))

ctrl_seq = [ unflatten(q, np.zeros_like(q))[0] for q in QQ ]

shape.create_movie(ctrl_seq, 'cell2.mp4', labels=False)
