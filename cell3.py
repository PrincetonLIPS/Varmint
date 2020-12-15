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

quad_deg   = 10
spline_deg = 3
num_ctrl   = 5
num_x      = 1
num_y      = 1
xknots     = default_knots(spline_deg, num_ctrl)
yknots     = default_knots(spline_deg, num_ctrl)
widths     = 5*np.ones(num_x)
heights    = 5*np.ones(num_y)
radii      = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
#radii      = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
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

def_ctrl = ref_ctrl.copy()
def_vels = np.zeros_like(def_ctrl)

q, qdot  = shape.get_flatten_fn()(def_ctrl, def_vels)

lagrangian = generate_lagrangian_structured(shape, ref_ctrl)

stepper = get_hamiltonian_stepper(lagrangian)

dt = 0.01
t  = dt
T  = 0.5

# Initial locations and momenta.
QQ = [ q.copy() ]
PP = [ np.zeros_like(q) ]
TT = [ 0.0 ]
ctrl_seq = [ unflatten(q, qdot)[0] ]

#server = jax.profiler.start_server(9999)

while TT[-1] < T:
  print('\nt: %0.4f' % (TT[-1]))

  t0 = time.time()
  new_q, new_p = stepper(QQ[-1], PP[-1], dt)

  QQ.append(new_q)
  PP.append(new_p)

  ctrl_seq.append( unflatten(QQ[-1], PP[-1])[0] )

  TT.append(TT[-1]+dt)

  t1 = time.time()
  #print('\t%d' % (res.nfev))
  #print('\t%d' % (res.njev))
  print('\t%0.3f sec' % (t1-t0))

shape.create_movie(ctrl_seq, 'cell.mp4', labels=False)
