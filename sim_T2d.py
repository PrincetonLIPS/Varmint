import time
import jax
import jax.numpy      as np
import numpy          as onp
import scipy.optimize as spopt
import jax.scipy.optimize as jopt

from varmint.patch2d      import Patch2D
from varmint.shape2d      import Shape2D
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D
from varmint.bsplines     import mesh, default_knots
from varmint.lagrangian   import generate_lagrangian
from varmint.discretize   import discretize_eulag

class WigglyMat(Material):
  _E = 0.001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

spline_deg = 3
quad_deg   = 10

# Length units are centimeters.

# Set up patch 1.
p1_length    = 40
p1_height    = 5
p1_num_xctrl = 10
p1_num_yctrl = 6
p1_ctrl      = mesh(np.linspace(0, p1_length, p1_num_xctrl),
                    np.linspace(0, p1_height, p1_num_yctrl))
p1_xknots = default_knots(spline_deg, p1_num_xctrl)
p1_yknots = default_knots(spline_deg, p1_num_yctrl)
p1_labels = onp.zeros((p1_num_xctrl, p1_num_yctrl), dtype='<U256')
p1_labels[0,:]  = ['A', 'B', 'C', 'D', 'E', 'F']
p1_labels[-1,:] = ['G', 'H', 'I', 'J', 'K', 'L']
p1_fixed = {
  'A': p1_ctrl[0,0,:],
  'B': p1_ctrl[0,1,:],
  'C': p1_ctrl[0,2,:],
  'D': p1_ctrl[0,3,:],
  'E': p1_ctrl[0,4,:],
  'F': p1_ctrl[0,5,:],
}
patch1 = Patch2D(
  p1_xknots,
  p1_yknots,
  spline_deg,
  mat,
  quad_deg,
  labels=p1_labels,
  fixed=p1_fixed
)

# Set up patch 2.
p2_length    = 5
p2_num_xctrl = 5
p2_num_yctrl = 12
p2_ctrl      = mesh(np.linspace(0, p2_length, p2_num_xctrl)+p1_length,
                    np.array([-15, -10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20]))
p2_xknots    = default_knots(spline_deg, p2_num_xctrl)
p2_yknots    = default_knots(spline_deg, p2_num_yctrl)
p2_labels    = onp.zeros((p2_num_xctrl, p2_num_yctrl), dtype='<U256')
p2_labels[0,3:9] = ['G', 'H', 'I', 'J', 'K', 'L']
patch2 = Patch2D(
  p2_xknots,
  p2_yknots,
  spline_deg,
  mat,
  quad_deg,
  labels=p2_labels,
)

shape = Shape2D(patch1, patch2)

# Reference configuration and initial deformation/velocity.
ref_ctrl = [p1_ctrl, p2_ctrl]
def_ctrl = [p1_ctrl.copy(), p2_ctrl.copy()]
def_vels = [np.zeros_like(p1_ctrl), np.zeros_like(p2_ctrl)]
q, qdot  = shape.flatten(def_ctrl, def_vels)

# Get the Lagrangian and then convert to discrete Euler-Lagrange.
lagrangian = generate_lagrangian(shape, ref_ctrl)
disc_eulag = jax.jit(discretize_eulag(lagrangian))
jac_del    = jax.jit(jax.jacfwd(disc_eulag, argnums=4))

dt = 0.01
t  = dt
T  = 3.0
QQ = [ q.copy(), q.copy() ]
VV = [ np.zeros_like(q), np.zeros_like(q) ]
TT = [ 0.0, dt ]

while t < T:
  print('\nt: %0.4f' % (t))

  q1 = QQ[-2]
  q2 = QQ[-1]
  t1 = TT[-2]
  t2 = TT[-1]

  t0 = time.time()
  res = spopt.least_squares(
    lambda q3: disc_eulag(q1, t1, q2, t2, q3, t2+dt),
    q2,
    lambda q3: jac_del(q1, t1, q2, t2, q3, t2+dt),
    method='lm',
  )
  t1 = time.time()
  q3 = res.x
  print('\t%0.3f sec' % (t1-t0))

  QQ.append(q3)
  TT.append(t2+dt)
  VV.append((q3-q2)/dt)
  t = t2 + dt

ctrl_seq = list(map(lambda qv: shape.unflatten(qv[0], qv[1])[0], zip(QQ,VV)))

shape.create_movie(ctrl_seq, 'T2d.mp4', labels=False)
