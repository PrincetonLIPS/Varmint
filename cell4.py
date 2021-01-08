import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import string

#from jax.config import config
#config.update('jax_disable_jit', True)
#config.update("jax_debug_nans", True)
#config.update("jax_enable_x64", True)

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

quad_deg   = 10
spline_deg = 1
num_ctrl   = 3
num_x      = 1
num_y      = 1
xknots     = default_knots(spline_deg, num_ctrl)
yknots     = default_knots(spline_deg, num_ctrl)
widths     = 5*np.ones(num_x)
heights    = 5*np.ones(num_y)
#radii      = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
radii      = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
ref_ctrl   = generate_quad_lattice(widths, heights, radii)

labels = match_labels(ref_ctrl, keep_singletons=True)

left_side = ref_ctrl[:,:,:,0] == 0.0
fixed_dict = dict(zip(labels[left_side], ref_ctrl[left_side,:]))

patches = [ Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  labels[ii,:,:],
  fixed_dict)
            for  ii in range(len(ref_ctrl)) ]

shape = Shape2D(*patches)

unflatten = shape.get_unflatten_fn()
flatten   = shape.get_flatten_fn()

dt = 0.01
t  = dt
T  = 0.02

init_refq, _ = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

def simulate(refq):

  ref_ctrl, _ = unflatten(refq, np.zeros_like(refq))

  lagrangian = generate_lagrangian_structured(shape)
  stepper = get_hamiltonian_stepper(lagrangian)

  def_ctrl = np.array(ref_ctrl)
  def_vels = np.zeros_like(def_ctrl)

  q, _  = flatten(def_ctrl, np.zeros_like(def_ctrl))
  p = np.zeros_like(q)

  # Initial locations and momenta.
  QQ = [ np.array(q) ]
  PP = [ np.zeros_like(q) ]
  TT = [ 0.0 ]

  while TT[-1] < T:
    #print('\nt: %0.4f' % (TT[-1]))

    #t0 = time.time()
    new_q, new_p = stepper(q, p, 0.01, refq)
    #new_q = refq
    #new_p = p

    QQ.append(new_q)
    PP.append(new_p)
    TT.append(TT[-1]+dt)

    #t1 = time.time()
    #print('\t%0.3f sec' % (t1-t0))
  return np.array(QQ), np.array(PP)

def loss(refq):

  QQ, PP = simulate(refq)

  final_q, final_p = unflatten(QQ[-1], PP[-1])

  return np.sum(QQ[-1])

vg_loss = jax.value_and_grad(loss)

print(vg_loss(init_refq))

#ctrl_seq = [ unflatten(q, np.zeros_like(q))[0] for q in QQ ]

#shape.create_movie(ctrl_seq, 'cell.mp4', labels=False)
