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
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

# Length units are centimeters.
length    = 25
height    = 5
num_xctrl = 10
num_yctrl = 5
ctrl      = mesh(np.linspace(0, length, num_xctrl),
                 np.linspace(0, height, num_yctrl))

# Make the patch.
spline_deg = 3
quad_deg   = 10
xknots     = default_knots(spline_deg, num_xctrl)
yknots     = default_knots(spline_deg, num_yctrl)
labels = onp.zeros((num_xctrl, num_yctrl), dtype='<U256')
labels[0,:] = ['A', 'B', 'C', 'D', 'E']
fixed = {
  'A': ctrl[0,0,:],
  'B': ctrl[0,1,:],
  'C': ctrl[0,2,:],
  'D': ctrl[0,3,:],
  'E': ctrl[0,4,:],
}
patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  quad_deg,
  labels=labels,
  fixed=fixed
)
shape = Shape2D(patch)

# Reference configuration and initial deformation/velocity.
ref_ctrl = [ctrl]
def_ctrl = [ctrl.copy()]
def_vels = [np.zeros_like(ctrl)]
q, qdot  = shape.flatten(def_ctrl, def_vels)

# Get the Lagrangian and then convert to discrete Euler-Lagrange.
lagrangian = generate_lagrangian(shape, ref_ctrl)
disc_eulag = jax.jit(discretize_eulag(lagrangian))
jac_del    = jax.jit(jax.jacfwd(disc_eulag, argnums=4))

dt = 0.01
t  = dt
T  = 1.0
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

shape.create_movie(ctrl_seq, 'test.mp4', labels=True)
