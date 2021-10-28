import jax
import jax.numpy as np

from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh, elementwise


def get_mlp(nfeat, whidden, nhidden, activation):
    if activation == 'selu':
        act_fun = jax.nn.selu
    elif activation == 'tanh':
        act_fun = np.tanh
    elif activation == 'relu':
        act_fun = jax.nn.relu
    else:
        raise ValueError(f'Invalid activation function {activation}.')

    layers = []
    for _ in range(nhidden):
        layers.extend([Dense(whidden), elementwise(act_fun)])
    layers.extend([Dense(nfeat)])

    return stax.serial(
        *layers
    )
