import argparse

CLI_GENERAL = {
  'loglevel': {
    'type': str,
    'default': 'WARNING',
    'help': 'Logging level: DEBUG, INFO, WARNING (default), ERROR, or CRITICAL',
  },
  'group': {
    'type': str,
    'help': 'An identifier for the group. Could be integer or string.',
  },
  'num-verts': {
    'type': int,
    'default': 5000,
    'help': 'Number of vertices to use in orbit graph',
  },
  'cache-dir': {
    'type': str,
    'default': '.symmetria.cache',
    'help': 'The directory in which to store the computed embeddings.',
  },
  'graph-method': {
    'type':    str,
    'choices': ['mesh', 'random', 'sobol'],
    'default': 'mesh',
    'help':    'The method to use to generate the graph vertices.',
  },
  'seed': {
    'type': int,
    'default': 1,
    'help': 'Random seed',
  },
  'show': {
    'dest':   'show',
    'action': argparse.BooleanOptionalAction,
    'help':   'Just show the graphic rather than saving it',
  },
}

CLI_HARMONICS = {
  'num-harmonics': {
    'type': int,
    'default': 30,
    'help': 'Number of harmonics to compute.',
  },
  'method': {
    'type': str,
    'choices': ['rr', 'gl'],
    'default': 'rr',
    'help': 'Method to use. rr (Rayleigh-Ritz) or gl (graph Laplacian)',
  },
  'epsilon': {
    'type':    float,
    'default': 0.1,
    'help':    'Kernel length scale [gl only]',
  },
  'dense': {
    'dest':   'dense',
    'action': argparse.BooleanOptionalAction,
    'help':   'Use dense matrix computations [gl only]',
  },
  'quad-degree': {
    'type': int,
    'default': 1,
    'help': 'Degree of quadrature for Rayleigh-Ritz [rr only]',
  },
  'num-basis': {
    'type': int,
    'default': 1000,
    'help': 'Number of basis functions for Rayleigh-Ritz [rr only]',
  },
  'mesh-size': {
    'type': int,
    'default': 10000,
    'help': 'Mesh size [rr only]',
  },
  'ica': {
    'default': True,
    'help': 'Turn on/off ICA resolution of multiplicity.',
    'action': argparse.BooleanOptionalAction,
  },
  'ica-thresh': {
    'type': float,
    'default': 0.01,
    'help': 'Relative difference for eigenvalues to have multiplicity',
  },
  'latex': {
    'default': True,
    'help': 'Generate a LaTeX file that makes a subfigure',
    'action': argparse.BooleanOptionalAction,
  },
}

CLI_ORBIFOLD = {
  'embed-dims': {
    'type': int,
    'default': 0,
  },
  'dis': {
    'dest':   'dis',
    'help':   'Render a discontinuous function instead',
    'action': argparse.BooleanOptionalAction,
  }
}

CLI_PLANE = {
  'image-sz': {
    'type': int,
    'default': 500,
    'help': 'Number of pixels in width and height',
  },
  'view-scale': {
    'type': float,
    'default': 1.5,
  },
  '3d': {
    'dest': 'threed',
    'help': 'Render 3d surface plot',
    'action': argparse.BooleanOptionalAction,
  },
}

CLI_SPACE = {
  'uc': {
    'nargs': '+',
    'type': int,
    'default': [1,1,1],
    'help': 'Unit cells to render',
  },
  'resolution': {
    'type': int,
    'default': 50,
    'help': 'Number of voxels one unit corresponds to',
  },
}

CLI_RFF = {
  'length-scale': {
    'type': float,
    'default': 0.4,
    'help': 'The length scale for the random functions.',
  },
  'num-feats': {
    'type': int,
    'default': 1000,
    'help': 'Number of randomized Fourier features to use.',
  }
}

CLI_CONTOUR = {
  'contour-levels': {
    'type': int,
    'default': 5,
    'help': 'Number of contour levels to render',
  },
  'colormap': {
    'type': str,
    'default': 'viridis',
    'help': 'Colormap to use for rendering contours',
  },
}


def add_to_parser(parser, *args):
  for arglist in args:
    for name, kwargs in arglist.items():
      parser.add_argument('--' + name, **kwargs)
  return parser
