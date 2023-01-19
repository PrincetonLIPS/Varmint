import sys
import os
import argparse
import logging
import numpy             as np
import numpy.random      as npr
import matplotlib.pyplot as plt
import symmetria.cli     as cli

from symmetria import Symmetria

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
  )

  return parser.parse_args()

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


  xlims = [-2, 2]
  ylims = [-2, 2]

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

  # Build a random function in the embedding space.
  omegas = rng.standard_normal((embed_dims, args.num_feats)) \
    / args.length_scale
  phis = rng.random(args.num_feats) * 2 * np.pi
  weights = rng.standard_normal((args.num_feats,)) \
    / np.sqrt(args.num_feats)
  output = np.cos(embedded_px @ omegas + phis) @ weights

  plt.contourf(
    px_grid[:,:,0],
    px_grid[:,:,1],
    output.reshape(args.image_sz, args.image_sz),
    cmap=args.colormap,
  )
  plt.gca().set_aspect('equal')
  plt.axis('off')
  plt.savefig('test.png')
  plt.show()

if __name__ == '__main__':
  sys.exit(main())
