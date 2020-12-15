import numpy as onp
import numpy.random as npr
import jax.numpy as np
import jax.numpy.linalg as npla
import matplotlib.pyplot as plt

from varmint.patch2d  import Patch2D
from varmint.shape2d  import Shape2D
from varmint.bsplines import default_knots, mesh
from varmint.materials    import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D

class WigglyMat(Material):
  _E = 0.0001
  _nu = 0.48
  _density = 1.0

mat = NeoHookean2D(WigglyMat)

def gen_edge(center, corner1, corner2, radii, spline_deg):
  num_ctrl = len(radii)

  right_perim = np.linspace(corner1, corner2, num_ctrl)

  theta_start = np.arctan2(corner1[1]-center[1],
                           corner1[0]-center[0])
  theta_end = np.arctan2(corner2[1]-center[1],
                         corner2[0]-center[0])
  theta = np.linspace(theta_start, theta_end, num_ctrl)

  left_perim = radii[:,np.newaxis] * (right_perim - center) + center

  ctrl = np.linspace(left_perim, right_perim, num_ctrl)

  return ctrl

def gen_cell(corners, radii, spline_deg):
  num_ctrl = (len(radii) // 3) + 1
  centroid = np.mean(corners, axis=0)
  ctrl = []
  for ii in range(3):
    corner1 = corners[ii]
    corner2 = corners[(ii + 1) % 3]
    start = (num_ctrl-1)*ii
    end = start + num_ctrl
    indices = np.arange(start, end)

    ctrl.append(gen_edge(centroid, corner1, corner2, np.take(radii, indices, mode='wrap'), spline_deg))

  return ctrl

'''
num_ctrl = 5
center = np.array([0.0, 0.0])
corner1 = np.array([ 1.0, 1.0 ])
corner2 = np.array([ 1.0, -2.0])
radii = 0.5 * np.ones(num_ctrl)
#radii = npr.rand(num_ctrl)
spline_deg = 3
quad_deg = 10

xknots = default_knots(spline_deg, num_ctrl)
yknots = default_knots(spline_deg, num_ctrl)

patch = Patch2D(
  xknots,
  yknots,
  spline_deg,
  mat,
  10,
)
shape = Shape2D(patch)

ctrl = gen_edge(center, corner1, corner2, radii, spline_deg)

shape.create_movie([[ctrl]], 'cellular.mp4', labels=True)
'''

'''
npr.seed(7)

spline_deg = 3
num_ctrl   = 5
corners    = npr.randn(3,2)
#corners = np.array([[0.0, 0.0],
#                    [0.0, 1.0],
#                    [1.0, 0.0]])
radii      = npr.rand((num_ctrl-1) * 3)
#radii = 0.25 * np.ones((num_ctrl-1)*3)
xknots = default_knots(spline_deg, num_ctrl)
yknots = default_knots(spline_deg, num_ctrl)
quad_deg = 10

patches = [ Patch2D(xknots, yknots, spline_deg, mat, quad_deg) for _ in range(3) ]
shape = Shape2D(*patches)

ctrl = gen_cell(corners, radii, spline_deg)
shape.create_movie([ctrl], 'cellular.mp4', labels=True)
'''

from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

npr.seed(1)

quad_deg = 10

spline_deg = 2
num_ctrl = 5
#num_points = 20

xknots = default_knots(spline_deg, num_ctrl)
yknots = default_knots(spline_deg, num_ctrl)
#points = 10*npr.randn(num_points,2)

hexmesh = mesh(np.arange(2), np.sqrt(3)*np.arange(2))
hexmesh = np.concatenate([hexmesh, hexmesh + np.array([[[0.5,np.sqrt(3)/2]]])])
points = np.reshape(hexmesh, (-1,2))

tri = Delaunay(points)


from varmint.cellular2d import match_labels, _gen_cell, generate_quad_lattice

'''
ctrl = []
for simplex in tri.simplices:
  corners = points[simplex,:]
  #radii = 0.5 * np.ones((num_ctrl-1)*3)
  radii = npr.rand((num_ctrl-1)*3)*0.8 + 0.1
  ctrl.extend( _gen_cell(corners, radii) )
ctrl = np.array(ctrl)
'''

widths = np.ones(3)
heights = np.ones(4)
radii = npr.rand(3,4,(num_ctrl-1)*4)
ctrl = generate_quad_lattice(widths, heights, radii)

labels = match_labels(ctrl, keep_singletons=False)

#epsilon = 1e-6
#unrolled = np.round(np.reshape(ctrl, (-1, 2)), decimals=5)
#dists = squareform(pdist(unrolled))
#adjacency = csr_matrix(dists < epsilon)
#n_components, labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)

#labels = onp.array(list(map(lambda i: 'group-%05d' % (i), labels)))

#labels = onp.reshape(labels, ctrl.shape[:-1])

patches = [ Patch2D(xknots, yknots, spline_deg, mat, quad_deg, labels[ii,:,:]) for ii in range(len(ctrl)) ]

shape = Shape2D(*patches)
shape.create_movie([ctrl], 'cellular.mp4', labels=True, fig_kwargs={'figsize':(10,10)})
