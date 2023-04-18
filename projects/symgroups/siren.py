import jax
import haiku as hk
import jax.numpy as jnp
import numpy as onp

from haiku.initializers import RandomUniform


def get_siren_layer(in_features, out_features, bias=True, is_first=False, omega_0=30.):
    def sine_nonlinearity(x):
        return jnp.sin(omega_0 * x)

    if is_first:
        w_init = RandomUniform(-1 / in_features,
                                1 / in_features)
    else:
        w_init = RandomUniform(-jnp.sqrt(6 / in_features) / omega_0,
                                jnp.sqrt(6 / in_features) / omega_0)
    linear_layer = hk.Linear(out_features, with_bias=bias, w_init=w_init)
    return [linear_layer, sine_nonlinearity]


def get_siren_network(input_dims, n_layers, n_activations, key,
                      first_omega_0=30.0, hidden_omega_0=30.0,
                      outermost_linear=True):
    def network(x):
        layers = []

        # First layer.
        layers.extend(get_siren_layer(input_dims, n_activations,
                                      is_first=True, omega_0=first_omega_0))

        # Hidden layers.
        for i in range(n_layers):
            layers.extend(get_siren_layer(n_activations, n_activations,
                                          is_first=False, omega_0=hidden_omega_0))

        # Last layer can either also be a sine layer or a linear layer without bias.
        if outermost_linear:
            w_init = RandomUniform(-jnp.sqrt(6 / n_activations) / hidden_omega_0,
                                    jnp.sqrt(6 / n_activations) / hidden_omega_0)
            layers.append(hk.Linear(2, with_bias=False, w_init=w_init))
        else:
            layers.extend(get_siren_layer(n_activations, 1,
                                          is_first=False, omega_0=hidden_omega_0))

        siren_mlp = hk.Sequential(layers)
        return siren_mlp(x)

    base_fn_t = hk.transform(network)
    base_fn_t = hk.without_apply_rng(base_fn_t)

    weights = base_fn_t.init(key, jnp.ones(input_dims))
    return jax.vmap(base_fn_t.apply, in_axes=(None, 0)), weights
