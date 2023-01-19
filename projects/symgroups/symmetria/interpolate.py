import jax
import optax
import logging
import jax.numpy        as jnp
import jax.random       as jrnd
import jax.scipy.linalg as jspla
import jax.numpy.linalg as jnpla
#import haiku            as hk

from .utils import sqeuclidean

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

rbf_kernels = {
  'thin_plate_spline': jax.jit(lambda r2: 0.5 * r2 * jnp.log(r2+1e-8)),
  #'polyharm2': jax.jit(lambda r2: 0.5 * r2 * jnp.log(r2+1e-8)),
  'polyharm2': jax.jit(lambda r2: 0.5 * r2 * jnp.where(r2>0, jnp.log(r2), 0)),
  'polyharm3': jax.jit(lambda r2: r2**(3/2)),
  #'polyharm4': jax.jit(lambda r2: 0.5 * r2**2 * jnp.log(r2+1e-8)),
  'polyharm4': jax.jit(lambda r2: 0.5 * r2**2 * jnp.where(r2>0, jnp.log(r2), 0)),
  'polyharm5': jax.jit(lambda r2: r2**(5/2)),
  'gaussian': jax.jit(lambda r2: jnp.exp(-0.5*r2/0.05**2)),
  'cauchy': jax.jit(lambda r2: 1/(1+r2/0.1**2))
}

def InterpolatorFromName(name, **kwargs):
  if name in rbf_kernels:
    return RBFInterpolator(kernel=name, **kwargs)
  else:
    raise Exception("FIXME")

# TODO: have this just return a jitted callable

class RBFInterpolator:
  def __init__(self, kernel='polyharm2'):
    self.kernel = rbf_kernels[kernel]

  def fit(self, train_X, train_Y, subsample=10000):

    if subsample != 0 and train_X.shape[0] > subsample:
      log.debug('Subsampling %d data for interpolation.' % (subsample))
      key = jrnd.PRNGKey(42)
      order = jrnd.permutation(key, train_X.shape[0])[:subsample]
      train_X = train_X[order,:]
      train_Y = train_Y[order,:]

    N, D = train_X.shape

    R2 = sqeuclidean(train_X)

    A = self.kernel(R2)
    B = jnp.column_stack([
      jnp.ones((N,1)),
      train_X,
    ])
    M = jnp.block([
      [A, B],
      [B.T, jnp.zeros((D+1,D+1))],
    ])
    f = jnp.row_stack([
      train_Y,
      jnp.zeros((D+1,train_Y.shape[1])),
    ])

    self.train_X = train_X

    solved = jspla.solve(M, f)
    self.w = solved[:N,:]
    self.v = solved[N:,:]

  def interpolate(self, X):
    R2 = sqeuclidean(X, self.train_X)
    K = self.kernel(R2)
    B = jnp.column_stack([
      jnp.ones((X.shape[0],1)),
      X,
    ])
    return K @ self.w + B @ self.v

class RFFRegressor:
  def __init__(self, dims, num_basis=500, seed=13, ls=0.1):
    omega_key, phi_key = jrnd.split(jrnd.PRNGKey(seed), 2)
    self.dims      = dims
    self.num_basis = num_basis
    self.omegas    = jrnd.normal(omega_key, shape=(dims, num_basis))/ls
    self.phis      = jrnd.uniform(phi_key, shape=(num_basis,))*2*jnp.pi

  def fit(self, train_X, train_Y):
    B = jnp.cos(train_X @ self.omegas + self.phis)
    print('B', B.shape)
    self.weights, resids, rank, svs = jnpla.lstsq(B, train_Y)
    print('wts', self.weights.shape)
    print('resids', resids)

  def interpolate(self, X):
    B = jnp.cos(X @ self.omegas + self.phis)
    return B @ self.weights

class NNRegressor:
  def __init__(self, in_dims, out_dims):

    def network(x):
      mlp = hk.Sequential([
        hk.Linear(50), jax.nn.softplus,
        hk.Linear(50), jax.nn.softplus,
        hk.Linear(out_dims),
      ])
      return mlp(x)

    base_fn_t = hk.transform(network)
    self.base_fn_t = hk.without_apply_rng(base_fn_t)

    rng = jrnd.PRNGKey(1)
    self.params = base_fn_t.init(rng, jnp.ones(in_dims))

    self.vmap_net = jax.jit(jax.vmap(
      self.base_fn_t.apply,
      in_axes=(None,0),
    ))

  def fit(self, train_X, train_Y, batch_size=50):

    def loss(params, batch_X, batch_Y):
      preds = self.vmap_net(params, batch_X)
      return jnp.mean((preds - batch_Y)**2)

    loss_valgrad = jax.jit(jax.value_and_grad(loss))

    num_batches = int(jnp.floor(train_X.shape[0]/batch_size))
    batches_X = train_X[:batch_size*num_batches,:].reshape(num_batches, batch_size, train_X.shape[1])
    batches_Y = train_Y[:batch_size*num_batches,:].reshape(num_batches, batch_size, train_Y.shape[1])

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(self.params)

    @jax.jit
    def step(params, opt_state, batch_X, batch_Y):
      loss_value, grads = loss_valgrad(params, batch_X, batch_Y)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_value

    params = self.params
    for jj in range(500):
      for ii in range(batches_X.shape[0]):
        params, opt_state, loss_value = step(
          params,
          opt_state,
          batches_X[ii,...],
          batches_Y[ii,...],
        )
      print(jj, loss_value)

    self.params = params

  def interpolate(self, x):
    return self.vmap_net(self.params, jnp.atleast_2d(x))
