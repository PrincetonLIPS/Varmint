import numpy as np
import jax.numpy as jnp
import jax

import numpy.random as npr

from functools import partial


def random_nside(N, min_side=2, max_side=4, amp=0.2):
    random_angle = npr.uniform(0, 2*np.pi)
    n_sides = npr.randint(min_side, max_side+1)

    angles = np.arange(np.pi*1.25, np.pi*(1.25-2), -2*np.pi/N)
    cps = np.c_[np.cos(angles) * (1. + amp * np.sin(n_sides * (angles + random_angle))),
                np.sin(angles) * (1. + amp * np.sin(n_sides * (angles + random_angle)))]

    return cps


def random_ellipse(N):
    angles = np.arange(np.pi*1.25, np.pi*(1.25-2), -2*np.pi/N)
    circle_cps = np.c_[np.cos(angles), np.sin(angles)]

    random_angle = npr.uniform(0, 2*np.pi)

    rotation_mat = np.array([
        [np.cos(random_angle), -np.sin(random_angle)],
        [np.sin(random_angle),  np.cos(random_angle)]
    ])
    random_scales = np.diag(npr.uniform(0.5, 1.5, size=2))

    return (rotation_mat.T @ random_scales @ rotation_mat @ circle_cps.T).T


def just_circle(N):
    angles = np.arange(np.pi*1.25, np.pi*(1.25-2), -2*np.pi/N)
    circle_cps = np.c_[np.cos(angles), np.sin(angles)]
    return circle_cps


def random_fourier_features(N, num_feats, omegas, phis):
    angles = np.arange(np.pi*1.25, np.pi*(1.25-2), -2*np.pi/N)

    carts = np.row_stack([np.cos(angles), np.sin(angles)])
    feats = np.cos(omegas @ carts + phis).T * np.sqrt(2 / num_feats)
    weights = npr.randn(num_feats)
    func = feats @ weights

    return (carts * np.exp(func)).T


def al_parameterized_rff(N, num_feats, omegas, phis):
    num_thetas = 1000  # Just some high number for quadrature.
    weights = npr.randn(num_feats)
    def x(theta):
      cart = jnp.array([jnp.cos(theta), jnp.sin(theta)])
      feats = jnp.cos(omegas @ cart + phis) * jnp.sqrt(2. / num_feats)
    
      func = feats @ weights
      return cart * jnp.exp(func)

    def x_prime_norm(theta):
      return jnp.linalg.norm(jax.jacfwd(x)(theta))

    thetas = jnp.linspace(0, 2*jnp.pi, num_thetas)

    # We want to arclength parameterize the curve, and choose the starting point.
    arclength_cumsum = 2 * jnp.pi / num_thetas * jnp.cumsum(jax.vmap(x_prime_norm)(thetas))
    total_arclength = arclength_cumsum[-1]
    small_arclengths = total_arclength / N * jnp.arange(N)
    control_indices = jnp.searchsorted(arclength_cumsum, small_arclengths)
    controls = thetas[control_indices]

    new_x_vals = jax.vmap(x)(controls)

    # Start in the bottom left corner, move counterclockwise.
    start_ind = jnp.argmin(jnp.sum(new_x_vals, axis=-1))
    reindexers = jnp.arange(start_ind, start_ind + N) % N
    new_x_vals = new_x_vals[reindexers][::-1, :]  # Move counterclockwise.

    return new_x_vals


def get_shape_target_generator(shape_family, N, generator_params):
    if shape_family == 'circle':
        return partial(just_circle, N)
    elif shape_family == 'ellipse':
        return partial(random_ellipse, N)
    elif shape_family == 'nside':
        return partial(random_nside, N)
    elif shape_family == 'nal_rff':
        npr.seed(10)
        lengthscale = generator_params['lengthscale']
        num_feats = generator_params['num_feats']

        omegas = npr.randn(num_feats, 2) / lengthscale
        phis = npr.rand(num_feats, 1) * 2 * np.pi

        return partial(random_fourier_features, N, num_feats, omegas, phis)
    elif shape_family == 'rff':
        npr.seed(10)
        lengthscale = generator_params['lengthscale']
        num_feats = generator_params['num_feats']

        omegas = npr.randn(num_feats, 2) / lengthscale
        phis = npr.rand(num_feats) * 2 * np.pi

        #import pickle
        #with open('/n/fs/mm-iga/Varmint/nma_mpi/experiments/pore_shape_experiments/dataset_pickle.pkl', 'wb') as f:
        #    pickle.dump((omegas, phis), f)

        return partial(al_parameterized_rff, N, num_feats, omegas, phis)
    else:
        raise ValueError(f'Shape family {shape_family} not available.')

