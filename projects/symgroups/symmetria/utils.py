import jax
import jax.numpy    as jnp
import jax.numpy.linalg as jnpla
import numpy        as np
import numpy.linalg as npla

from shapely       import geometry as geom
from HersheyFonts  import HersheyFonts

# TODO: typing

@jax.jit
def _sqeuclidean(X, Y):
  return jnp.sum((X[:,jnp.newaxis,:]-Y[jnp.newaxis,:,:])**2, axis=2)

def sqeuclidean(X, Y=None):
  Y = X if Y is None else Y
  return _sqeuclidean(X, Y)

def recursive_map(func, items, tolist=True):
  newfunc = lambda x: recursive_map(func, x, tolist=tolist) \
    if type(x) in (list, tuple) else func(x)
  if tolist:
    return list(map(newfunc, items))
  else:
    return map(newfunc, items)

def scitbx2jax(mat):
  ''' Convert `scitbx <https://cctbx.github.io/scitbx/index.html>`_ rationals
  to JAX arrays. '''
  shape = mat.n

  numers = jnp.array(
    recursive_map(lambda x: x.numerator(), mat),
  ).reshape(shape)
  denoms = jnp.array(
    recursive_map(lambda x: x.denominator(), mat),
  ).reshape(shape)

  return numers / denoms

def boost2jax(val):
  ''' Convert `boost rationals
  <https://www.boost.org/doc/libs/1_73_0/libs/rational/rational.html>`_ to JAX
  arrays. '''
  if type(val) in (int,):
    numers = jnp.array(val)
    denoms = jnp.array(1)
  else:
    numers = jnp.array(val.numerator())
    denoms = jnp.array(val.denominator())

  return numers / denoms

def boostlist2jax(mat):
  ''' Convert lists of `boost rationals
  <https://www.boost.org/doc/libs/1_73_0/libs/rational/rational.html>`_ to
  JAX arrays. '''
  numers = jnp.array(
    recursive_map(lambda x: x.numerator(), mat),
  )
  denoms = jnp.array(
    recursive_map(lambda x: x.denominator(), mat),
  )

  return numers / denoms

def laplacian(func, argnums=0):
  r''' Apply the Laplacian operator to a function.

  The `Laplacian operator <https://en.wikipedia.org/wiki/Laplace_operator>`_ is
  the sum of second derivatives. If the function of interest
  is :math:`f:\mathbb{R}^d \to \mathbb{R}` then the Laplacian is
  :math:`\mathcal{L}[f] = \frac{d^2}{dx_1^2}f + \frac{d^2}{dx_1^2}f + \cdots +
  \frac{d^2}{dx_d^2}f`.  This is computed by taking the trace of the `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_.

  Args:
    func (Callable[jnp.DeviceArray, float]): A function that takes in a JAX
      array and returns a float. This is the function whose Laplacian is being
      computed.

    argnums (int): The argument of func to compute the Laplacian with respect
      to.  Defaults to 0.

  Returns:
    laplacian (Callable[jnp.DeviceArray, float]): The sum of second derivatives
      of func.
  '''

  def _lap(*args, **kwargs):
    return jnp.trace(jax.hessian(func, argnums)(*args, **kwargs))
  return _lap

def triangle_area(points):
  # Heron's formula.
  # Use this because it works in 2 and 3 dims.
  sq_dists = sqeuclidean(points)
  a, b, c = jnp.sqrt(sq_dists[jnp.tril_indices(3,-1)])
  s = (a+b+c)/2
  return jnp.sqrt(s*(s-a)*(s-b)*(s-c))

def tet_volume(points):
  M = points[-1,:] - points[:3,:]
  return jnp.abs(jnpla.det(M))/6

def get_text_strokes(scale, basis, text='F', font='timesr'):
  text_lines = []
  thefont = HersheyFonts()
  thefont.load_default_font(font)
  thefont.normalize_rendering(scale)
  for (x1, y1), (x2, y2) in thefont.lines_for_text(text):
    text_lines.append([
      [x1, y1],
      [x2, y2],
    ])
  text_lines = np.array(text_lines)
  text_hull = geom.Polygon(text_lines.reshape(-1,2)).convex_hull
  text_lines = text_lines - np.array(text_hull.centroid.coords)

  text_lines = (npla.solve(basis, text_lines.reshape(-1,2).T).T \
                .reshape(*text_lines.shape))
  return text_lines
