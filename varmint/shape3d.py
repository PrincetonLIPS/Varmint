import jax
import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt
import logging

from mpl_toolkits.mplot3d import Axes3D

from exceptions import (
  DimensionError,
  LabelError,
  )

import bsplines

class Patch3D:
  ''' Class for individual patches in three dimensions.
  '''
  def __init__(self, ctrl, xknots, yknots, zknots, deg, labels=None):
    self.ctrl   = ctrl
    self.xknots = xknots
    self.yknots = yknots
    self.zknots = zknots
    self.deg    = deg

    # Get the shape of the control points.
    xdim, ydim, zdim, _ = ctrl.shape
    logging.debug('xdim=%d ydim=%d zdim=%d' % (xdim, ydim, zdim))

    if labels is None:
      self.labels = onp.zeros((
        ctrl.shape[0],
        ctrl.shape[1],
        ctrl.shape[2],
      ), dtype='<U256')
    else:
      if labels.shape != ctrl.shape[:-1]:
        raise DimensionError('The labels must have shape %d x %d x %d.' \
                             % (ctrl.shape[0], ctrl.shape[1], ctrl.shape[2]))

  def pytree(self):
    return self.ctrl

  def jacobian(self, u):
    return bsplines.bspline3d_derivs(
      u,
      self.ctrl,
      self.xknots,
      self.yknots,
      self.zknots,
      self.deg,
    )

  def has_label(self, label):
    return onp.any(self.labels == label)

  def label2idx(self, label):
    ii, jj, kk = onp.nonzero(self.labels == label)
    if ii.shape[0] > 1 or jj.shape[0] > 1 or kk.shape[0] > 1:
      raise LabelError('More than one control point has label %s.' % (label))
    elif ii.shape[0] == 1 or jj.shape[0] == 1 or kk.shape[0] == 1:
      raise LabelError('No control points have label %s.' % (label))
    return ii[0], jj[0], kk[0]

  def label2ctrl(self, label):
    row, col = self.label2idx(label)
    return self.ctrl[ii,jj,kk,:]

class Shape3D:
  ''' Class for managing collections of 3D patches

  '''

  def __init__(self, *patches):
    self.patches = patches

    self.check_labels()

  def check_labels(self):
    ''' Verify that the labels aren't crazy. '''
    pass

  def pytree(self):
    # May be better to do this with pytree registration in JAX.
    # Need to handle constraints somehow.  Will revisit.
    return [patch.pytree() for patch in self.patches]

  def jacobian(self, u):
    return [patch.jacobian(u) for patch in self.patches]

  def render_wireframe(self, filename=None):
    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    uu = np.linspace(0, 1, 1002)[1:-1]

    rendered_labels = set()

    for patch in self.patches:

      # Plot vertical lines.
      for jj in range(patch.ctrl.shape[1]):
        for kk in range(patch.ctrl.shape[2]):
          xx = bsplines.bspline1d(
            uu,
            patch.ctrl[:,jj,kk,:],
            patch.xknots,
            patch.deg,
          )
          ax.plot(xx[:,0], xx[:,1], xx[:,2], 'k-')


      # Plot horizontal lines.
      for ii in range(patch.ctrl.shape[0]):
        for kk in range(patch.ctrl.shape[2]):
          yy = bsplines.bspline1d(
            uu,
            patch.ctrl[ii,:,kk,:],
            patch.yknots,
            patch.deg,
          )
          ax.plot(yy[:,0], yy[:,1], yy[:,2], 'k-')

      # Plot depth lines.
      for ii in range(patch.ctrl.shape[0]):
        for jj in range(patch.ctrl.shape[1]):
          zz = bsplines.bspline1d(
            uu,
            patch.ctrl[ii,jj,:,:],
            patch.zknots,
            patch.deg,
          )
          ax.plot(zz[:,0], zz[:,1], zz[:,2], 'k-')

      # Plot the control points themselves.
      ax.plot(patch.ctrl[...,0].ravel(),
              patch.ctrl[...,1].ravel(),
              patch.ctrl[...,2].ravel(), 'k.')

      '''
      # Plot labels.
      label_r, label_c = onp.where(patch.labels)
      for ii in range(len(label_r)):
        row = label_r[ii]
        col = label_c[ii]
        text = patch.labels[row,col]
        if text not in rendered_labels:
          rendered_labels.add(text)
        else:
          continue
        ax.annotate(text, patch.ctrl[row,col,:])
      '''

    if filename is None:
      plt.show()
    else:
      plt.savefig(filename)

  def render_surface(self, filename=None):
    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    uu, vv = np.meshgrid(np.linspace(0, 1, 25),
                         np.linspace(0, 1, 25))
    uv = np.vstack([uu.ravel(), vv.ravel()]).T

    rendered_labels = set()

    for patch in self.patches:

      for xface in [0,-1]:
        ww = bsplines.bspline2d(
          uv,
          patch.ctrl[xface,:,:,:],
          patch.yknots,
          patch.zknots,
          patch.deg,
        )
        ax.plot_surface(
          np.reshape(ww[:,0], uu.shape),
          np.reshape(ww[:,1], uu.shape),
          np.reshape(ww[:,2], uu.shape),
          color='b')


      for yface in [0,-1]:
        ww = bsplines.bspline2d(
          uv,
          patch.ctrl[:,yface,:,:],
          patch.xknots,
          patch.zknots,
          patch.deg,
        )
        ax.plot_surface(
          np.reshape(ww[:,0], uu.shape),
          np.reshape(ww[:,1], uu.shape),
          np.reshape(ww[:,2], uu.shape),
          color='b')

      for zface in [0,-1]:
        ww = bsplines.bspline2d(
          uv,
          patch.ctrl[:,:,zface,:],
          patch.xknots,
          patch.yknots,
          patch.deg,
        )
        ax.plot_surface(
          np.reshape(ww[:,0], uu.shape),
          np.reshape(ww[:,1], uu.shape),
          np.reshape(ww[:,2], uu.shape),
          color='b')

      # Plot the control points themselves.
      # ax.plot(patch.ctrl[...,0].ravel(),
      #       patch.ctrl[...,1].ravel(),
      #       patch.ctrl[...,2].ravel(), 'k.')


      '''
      # Plot labels.
      label_r, label_c = onp.where(patch.labels)
      for ii in range(len(label_r)):
        row = label_r[ii]
        col = label_c[ii]
        text = patch.labels[row,col]
        if text not in rendered_labels:
          rendered_labels.add(text)
        else:
          continue
        ax.annotate(text, patch.ctrl[row,col,:])
      '''

    if filename is None:
      plt.show()
    else:
      plt.savefig(filename)


def test_shape1():

  # Create a rectangle.
  r1_deg    = 2
  r1_ctrl   = bsplines.mesh(np.arange(10), np.arange(5), np.arange(4))+0.0
  r1_xknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[0])
  r1_yknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[1])
  r1_zknots = bsplines.default_knots(r1_deg, r1_ctrl.shape[2])
  r1_patch  = Patch3D(r1_ctrl, r1_xknots, r1_yknots, r1_zknots, r1_deg)
  #r1_patch.labels[3:7,0]  = ['A', 'B', 'C', 'D']
  #r1_patch.labels[:3,-1]  = ['E', 'F', 'G']
  #r1_patch.labels[-3:,-1] = ['H', 'I', 'J']


  # Create another rectangle.
  r2_deg    = 2
  r2_ctrl   = bsplines.mesh(np.array([3,4,5,6]),
                            np.array([-4, -3, -2, -1, 0]),
                            np.arange(4))+0.0
  r2_xknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[0])
  r2_yknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[1])
  r2_zknots = bsplines.default_knots(r2_deg, r2_ctrl.shape[2])
  r2_patch  = Patch3D(r2_ctrl, r2_xknots, r2_yknots, r2_zknots, r2_deg)
  #r2_patch.labels[:,-1] = ['A', 'B', 'C', 'D']

  # Bend a u-shaped thing around the top.
  u1_deg  = 2
  band    = np.array([[-4.5, -3.5, -2.5]])
  center  = np.array([4.5, 4, 0])
  u1_ctrl = np.zeros((3,8,4,3))
  for ii, theta in enumerate(np.linspace(-np.pi, 0, 8)):
    u1_ctrl = jax.ops.index_update(u1_ctrl,
                                   jax.ops.index[:,ii,:,0],
                                   band.T * np.cos(theta),
                                   )
    u1_ctrl = jax.ops.index_update(u1_ctrl,
                                   jax.ops.index[:,ii,:,1],
                                   band.T * np.sin(theta))
    u1_ctrl = jax.ops.index_update(u1_ctrl,
                                   jax.ops.index[:,ii,:,2],
                                   np.arange(4)[np.newaxis,:])
  u1_ctrl   = u1_ctrl + center
  u1_xknots = bsplines.default_knots(u1_deg, u1_ctrl.shape[0])
  u1_yknots = bsplines.default_knots(u1_deg, u1_ctrl.shape[1])
  u1_zknots = bsplines.default_knots(u1_deg, u1_ctrl.shape[2])
  u1_patch  = Patch3D(u1_ctrl, u1_xknots, u1_yknots, u1_zknots, u1_deg)
  #u1_patch.labels[:,0]  = ['E', 'F', 'G']
  #u1_patch.labels[:,-1] = ['H', 'I', 'J']

  shape = Shape3D(r1_patch, r2_patch, u1_patch)
  return shape

if __name__ == '__main__':
  test_shape1().render_surface()
