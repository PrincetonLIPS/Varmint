import sys
import os
import argparse
import logging
import jax
import numpy             as np
import numpy.random      as npr
import jax.numpy         as jnp
import jax.random        as jrnd
import matplotlib.pyplot as plt
import haiku             as hk
import symmetria.cli     as cli

from symmetria import Symmetria
from functools import partial

logger = logging.getLogger()

def parse_command_line():
  parser = argparse.ArgumentParser(
    description='''
    Render a random function on the plane with the specified symmetry.
    '''
  )

  cli.add_to_parser(
    parser,
    cli.CLI_GENERAL,
    cli.CLI_ORBIFOLD,
    cli.CLI_PLANE,
    cli.CLI_RFF,
    cli.CLI_CONTOUR,
    cli.CLI_NN,
  )

  return parser.parse_args()


def get_network(input_dims, layers, activation, seed):
  def network(x):
    stack = []
    for layer in layers:
      stack.append(hk.Linear(layer))
      stack.append(activation)
    stack.append(hk.Linear(1))

    mlp = hk.Sequential(stack)
    return mlp(x)

  base_fn_t = hk.transform(network)
  base_fn_t = hk.without_apply_rng(base_fn_t)

  rng = jrnd.PRNGKey(seed)
  weights = base_fn_t.init(rng, jnp.ones(input_dims))

  return jax.jit(jax.vmap(partial(base_fn_t.apply,weights), in_axes=0))


def main():
  args = parse_command_line()

  logger.setLevel(level=args.loglevel)

  logging.getLogger('matplotlib').setLevel('WARNING')
  logging.getLogger('absl').setLevel('WARNING')
  logging.getLogger('numba').setLevel('WARNING')

  # Fix the random seed.
  rng = npr.default_rng(args.seed)

  S = Symmetria.plane_group(args.group)
  basis = S.sg.basic_basis

  xlims = [0, 9]
  ylims = [0, 3]

  pxx = np.linspace(*xlims, args.image_sz)
  pxy = np.linspace(*ylims, args.image_sz)
  px_grid = np.stack(np.meshgrid(pxx, pxy), axis=-1)

  embedder = S.get_orbifold_map(
    num_verts = args.num_verts,
    graph_method = args.graph_method,
    embed_dims = args.embed_dims,
  )

  embedded_px = embedder(px_grid.reshape(-1,2))[0]
  embed_dims = embedded_px.shape[1]
  netfunc = get_network(
    embed_dims,
    args.layers,
    jnp.sin,
    args.seed,
  )

  # Build a random NN function in the embedding space.
  output = netfunc(embedded_px).reshape(args.image_sz, args.image_sz)
  output = output - jnp.min(output)
  output = output / jnp.max(output)

  #plt.scatter(
  #  px_grid[:,:,0],
  #  px_grid[:,:,1],
  #  c=output,
  #  cmap='Greys',
  #)
  plt.contourf(
    px_grid[:,:,0],
    px_grid[:,:,1],
    2*output-1,
    cmap=args.colormap,
  )

  plt.gca().set_aspect('equal')
  plt.savefig('test.png')
  plt.show()

if __name__ == '__main__':
  sys.exit(main())
