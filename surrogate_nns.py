import jax
import jax.numpy as np

from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh


def get_tanh_net(nfeat):
  return stax.serial(
    Dense(2048), Tanh,
    Dense(2048), Tanh,
    Dense(2048), Tanh,
    Dense(nfeat)
  )
