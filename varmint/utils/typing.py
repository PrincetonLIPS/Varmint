from typing import Iterable, Union, Optional, Any

import os

import jax
import jax.numpy as jnp
import numpy as onp


Array1D = Union[jnp.ndarray, onp.ndarray]
Array2D = Union[jnp.ndarray, onp.ndarray]
Array3D = Union[jnp.ndarray, onp.ndarray]
Array4D = Union[jnp.ndarray, onp.ndarray]
ArrayND = Union[jnp.ndarray, onp.ndarray]

CtrlArray = Union[jnp.ndarray, onp.ndarray]

# --- aggregate types
ndarray: type = Union[jnp.ndarray, onp.ndarray]
prng_key: type = jax.random.PRNGKey
pytree: type = Union[dict, list, tuple] # not technically accurate but convenient.

# --- numeric types
FP16: type = jnp.float16
FP32: type = jnp.float32
FP64: type = jnp.float64

INT32: type = jnp.int32
INT64: type = jnp.int64

# --- string types
path_t: type = os.PathLike

