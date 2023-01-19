import jax.numpy        as jnp
import jax.scipy.linalg as jspla

from .utils import sqeuclidean

def MDS(
    dists: jnp.DeviceArray,
    num_dims: int,
    max_dims: int=12,
) -> jnp.DeviceArray:
  ''' Perform multidimensional scaling (MDS) on a data set.

  Args:
    dists (jnp.DeviceArray): NxN matrix of distances.

    num_dims (int): Number of dimensions to return.  If 0, select by minimizing
      the absolute error in the distances.

    max_dims (int): Maximum dimensionality to consider when searching. Default
      is 12.

  Returns:
    embeddings (jnp.DeviceArray): Embeddings whose Euclidean distances
      approximate the original distance matrix.
  '''

  num_data = dists.shape[0]

  # Center the squared distance matrix.
  centering = jnp.eye(num_data) - jnp.ones((num_data,num_data))/num_data
  centered  = -0.5 * centering @ dists**2 @ centering

  # Solve the eigenvalue problem.
  evals, evecs = jspla.eigh(centered)

  if num_dims < 1:
    # Possibly search, minimizing absolute distance error.
    best_dims = 0
    best_err  = jnp.inf
    for dd in range(1,max_dims):
      # Take the top eigenvectors and scale by squared eigenvalues.
      embeddings = evecs[:,-dd:] * jnp.sqrt(evals[-dd:])

      emb_dists = jnp.sqrt(sqeuclidean(embeddings))
      absdiff = jnp.sum(jnp.abs(emb_dists - dists))

      if absdiff < best_err:
        best_dims = dd
        best_err  = absdiff
    dims = best_dims
  else:
    dims = num_dims

  # Take the top eigenvectors and scale by squared eigenvalues.
  embeddings = evecs[:,-dims:] * jnp.sqrt(evals[-dims:])

  # Flip the ordering so the big ones come first.
  return embeddings[:,::-1]
