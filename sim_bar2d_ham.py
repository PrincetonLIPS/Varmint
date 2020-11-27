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
from varmint.discretize   import discretize_hamiltonian
from varmint.levmar       import levenberg_marquardt

class WigglyMat(Material):
  _E = 0.001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

# Length units are centimeters.
length    = 50
height    = 2
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
Ld_dq1, Ld_dq2 = discretize_hamiltonian(lagrangian)
Ld_dq1_jac = jax.jit(jax.jacfwd(Ld_dq1, argnums=1))

minfunc  = jax.jit(lambda p0, q0, q1, dt: np.sum((p0+Ld_dq1(q0, q1, dt))**2))
dminfunc = jax.jit(jax.grad(minfunc, argnums=2))
hessvec  = jax.jit(jax.grad(
  lambda p0, q0, q1, dt, v: dminfunc(p0, q0, q1, dt) @ v,
  argnums=2,
))
hessian  = jax.jit(jax.jacfwd(
  lambda p0, q0, q1, dt: dminfunc(p0, q0, q1, dt),
  argnums=2,
))

dt = 0.01
t  = dt
T  = 1.0

# Initial locations and momenta.
QQ = [ q.copy() ]
PP = [ np.zeros_like(q) ]
TT = [ 0.0 ]

while TT[-1] < T:
  print('\nt: %0.4f' % (TT[-1]))

  t0 = time.time()

  # Solve for the next q.
  if True:
    if False:
      res = spopt.least_squares(
        lambda q_1: PP[-1] + Ld_dq1(QQ[-1], q_1, dt), # should be zero
        QQ[-1], # initialize at current location
        lambda q_1: Ld_dq1_jac(QQ[-1], q_1, dt), # Jacobian
        method='lm',
      )
    else:
      res = levenberg_marquardt(
        lambda q_1: PP[-1] + Ld_dq1(QQ[-1], q_1, dt), # should be zero
        QQ[-1], # initialize at current location
        lambda q_1: Ld_dq1_jac(QQ[-1], q_1, dt), # Jacobian
        init_lamb=0.1,
      )
  elif False:
    res = spopt.minimize(
      lambda q1: minfunc(PP[-1], QQ[-1], q1, dt),
      QQ[-1],
      jac=lambda q1: dminfunc(PP[-1], QQ[-1], q1, dt),
      hess=lambda q1: hessian(PP[-1], QQ[-1], q1, dt),
      # hessp=lambda q1, v: hessvec(PP[-1], QQ[-1], q1, dt, v),
      method='Newton-CG',
    )

  print(Ld_dq1(QQ[-1], res.x, dt) + PP[-1])
  print(minfunc(PP[-1], QQ[-1], res.x, dt))

  t1 = time.time()
  print('\t%s' % (res.success))
  print('\t%d' % (res.nfev))
  print('\t%s' % (res.message))
  print('\t%0.3f sec' % (t1-t0))
  QQ.append(res.x)

  # Get the new momentum.
  PP.append( Ld_dq2(QQ[-2], QQ[-1], dt) )
  TT.append(TT[-1]+dt)

ctrl_seq = list(map(lambda qv: shape.unflatten(qv[0], qv[1])[0], zip(QQ,PP)))

shape.create_movie(ctrl_seq, 'bar2d_ham.mp4', labels=True)
