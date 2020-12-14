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
from varmint.discretize   import discretize_hamiltonian
from varmint.levmar       import get_lmfunc

import jax.profiler

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

quad_deg   = 10
width      = 25
rand_len   = 5
spline_deg = 3
radius     = 10
num_xctrl  = 5
num_yctrl  = 5
xknots     = default_knots(spline_deg, num_xctrl)
yknots     = default_knots(spline_deg, num_yctrl)

right_perim = np.vstack([np.ones(num_yctrl)*width/2,
                         np.linspace(-width/2, width/2, num_yctrl)]).T
theta = np.linspace(-np.pi/4, np.pi/4, num_yctrl)
left_perim = radius * np.vstack([np.cos(theta), np.sin(theta)]).T
east_ctrl = np.linspace(left_perim, right_perim, num_xctrl)
east_labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')


rot90 = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],
                  [np.sin(np.pi/2), np.cos(np.pi/2)]])

south_ctrl = np.tensordot(east_ctrl, rot90, ((2, 0)))
south_labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')

west_ctrl = np.tensordot(south_ctrl, rot90, ((2, 0)))
west_labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')

north_ctrl = np.tensordot(west_ctrl, rot90, ((2, 0)))
north_labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')

alphabet  = onp.array(list(string.ascii_lowercase))
se_labels = [''.join(row) for row in npr.choice(alphabet, (num_xctrl, rand_len))]
sw_labels = [''.join(row) for row in npr.choice(alphabet, (num_xctrl, rand_len))]
ne_labels = [''.join(row) for row in npr.choice(alphabet, (num_xctrl, rand_len))]
nw_labels = [''.join(row) for row in npr.choice(alphabet, (num_xctrl, rand_len))]

east_labels[:,-1] = ne_labels
east_labels[:,0] = se_labels
south_labels[:,-1] = se_labels
south_labels[:,0] = sw_labels
west_labels[:,-1] = sw_labels
west_labels[:,0] = nw_labels
north_labels[:,0] = ne_labels
north_labels[:,-1] = nw_labels

w_labels = [''.join(row) for row in npr.choice(alphabet, (num_yctrl-2, rand_len))]
west_labels[-1,1:-1] = w_labels
w_labels = west_labels[-1,:]
west_fixed = west_ctrl[-1,:,:]
west_dict = dict(zip(w_labels, west_fixed))

east_patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  east_labels,
)
south_patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  south_labels,
)
west_patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  west_labels,
  west_dict,
)
north_patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  north_labels,
)

shape = Shape2D(east_patch, south_patch, west_patch, north_patch)

ref_ctrl = np.array([east_ctrl, south_ctrl, west_ctrl, north_ctrl])
def_ctrl = ref_ctrl.copy()
def_vels = np.zeros_like(def_ctrl)

q, qdot  = shape.get_flatten_fn()(def_ctrl, def_vels)

lagrangian = generate_lagrangian_structured(shape, ref_ctrl)

Ld_dq1, Ld_dq2 = discretize_hamiltonian(lagrangian)
Ld_dq1_jac = jax.jit(jax.jacfwd(Ld_dq1, argnums=1))

@jax.jit
def residual_fun(q1, args):
  q0, p0, dt = args
  return p0 + Ld_dq1(q0, q1, dt)
optfun = get_lmfunc(residual_fun, maxiters=200)

dt = 0.01
t  = dt
T  = 1.0

unflatten = shape.get_unflatten_fn()

# Initial locations and momenta.
QQ = [ q.copy() ]
PP = [ np.zeros_like(q) ]
TT = [ 0.0 ]
ctrl_seq = [ unflatten(q, qdot)[0] ]

#server = jax.profiler.start_server(9999)

while TT[-1] < T:
  print('\nt: %0.4f' % (TT[-1]))

  t0 = time.time()
  if True:
    res = spopt.least_squares(
      lambda q_1: PP[-1] + Ld_dq1(QQ[-1], q_1, dt), # should be zero
      QQ[-1], # initialize at current location
      lambda q_1: Ld_dq1_jac(QQ[-1], q_1, dt), # Jacobian
      method='lm',
    )
  else:
    res = optfun(QQ[-1], (QQ[-1], PP[-1], dt))

  QQ.append(res.x)#.block_until_ready())

  ctrl_seq.append( unflatten(QQ[-1], PP[-1])[0] )

  # Get the new momentum.
  PP.append( Ld_dq2(QQ[-2], QQ[-1], dt))#.block_until_ready() )
  TT.append(TT[-1]+dt)

  t1 = time.time()
  print('\t%d' % (res.nfev))
  print('\t%d' % (res.njev))
  print('\t%0.3f sec' % (t1-t0))

shape.create_movie(ctrl_seq, 'cell.mp4', labels=False)
