import numpy as np
import os
import jax, jaxlib
import jax.numpy as jnp
from jax import random, custom_jvp, custom_vjp
import dm_pix as pix
from functools import partial
import scipy, scipy.sparse, scipy.sparse.linalg


@partial(custom_vjp, nondiff_argnums=(0,1,))
def gaussian_filter(filter_width, kernel_size, x): # 2D gaussian blur/filter
    nelx, nely = x.shape
    return jnp.reshape(pix.gaussian_blur(
        jnp.reshape(x.astype(jnp.float32), (nelx, nely, 1)),
        1*filter_width, kernel_size),
        (nelx, nely)).astype(jnp.float64)


def _gaussian_filter_fwd(filter_width, kernel_size, x):
    return (gaussian_filter(filter_width, kernel_size, x),None)


def _gaussian_filter_bwd(filter_width, kernel_size, res, g):
    return (gaussian_filter(filter_width, kernel_size, g),)

gaussian_filter.defvjp(_gaussian_filter_fwd, _gaussian_filter_bwd)


def physical_density(x, f1, f2, use_filter=True):
    return gaussian_filter(f1, f2, x) if use_filter else x  # maybe filter


def mean_density(x, f1, f2, use_filter=True):
    return np.mean(physical_density(x, f1, f2, use_filter)) # / np.mean(args.mask)
