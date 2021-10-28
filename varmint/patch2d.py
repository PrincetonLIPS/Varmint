import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import quadpy

from .exceptions import LabelError
from .bsplines import (
    bspline1d_derivs_hand,
    mesh,
    bspline2d,
    bspline2d_basis,
    bspline2d_basis_derivs,
    bspline2d_derivs,
    bspline2d_derivs_ctrl,
)


class Patch2D:
    ''' Class for individual patches in two dimensions.

    A patch corresponds to a two-dimensional bspline with tensor product basis
    functions.  They are individual maps from [0,1]^2 to R^2, parameterized
    by control points.
    '''

    def __init__(
        self,
        xknots,
        yknots,
        spline_deg,
        material,
        quad_deg,
    ):
        ''' Constructor for two dimensional patch.

        Parameters:
        -----------
         - xknots: A length-M one-dimensional array of bspline knots for the x
                   dimension. These are assumed to be in non-decreasing order from
                   0.0 to 1.0.

         - yknots: A length-N one-dimensional array of bspline knots for the y
                   dimension. These are assumed to be in non-decreasing order from
                   0.0 to 1.0.

         - spline_deg: The degree of the bspline.

        '''
        self.xknots = xknots
        self.yknots = yknots
        self.spline_deg = spline_deg
        self.material = material
        self.quad_deg = quad_deg

        # Determine the number of control points.
        num_xknots = self.xknots.shape[0]
        num_yknots = self.yknots.shape[0]

        self.num_xctrl = num_xknots - self.spline_deg - 1
        self.num_yctrl = num_yknots - self.spline_deg - 1

        self.compute_quad_points()

    def compute_quad_points(self):

        # Each knot span has its own quadrature.
        uniq_xknots = onp.unique(self.xknots)
        uniq_yknots = onp.unique(self.yknots)

        xwidths = onp.diff(uniq_xknots)
        ywidths = onp.diff(uniq_yknots)

        # We need the span volumes for performing integration later.
        self.span_volumes = xwidths[:, np.newaxis] * ywidths[np.newaxis, :]
        self.xwidths = xwidths
        self.ywidths = ywidths

        # Ask quadpy for a quadrature scheme.
        scheme = quadpy.c2.get_good_scheme(self.quad_deg)

        # Scheme for line integrals
        line_scheme = quadpy.c1.gauss_legendre(3 * self.quad_deg)

        # Change the domain from (-1,1)^2 to (0,1)^2
        points = scheme.points.T/2 + 0.5
        line_points = line_scheme.points / 2 + 0.5

        # Repeat the quadrature points for each knot span, scaled appropriately.
        offset_mesh = mesh(uniq_xknots[:-1], uniq_yknots[:-1])
        width_mesh = mesh(xwidths, ywidths)

        self.points = np.reshape(points[np.newaxis, np.newaxis, :, :]
                                 * width_mesh[:, :, np.newaxis, :]
                                 + offset_mesh[:, :, np.newaxis, :],
                                 (-1, 2))
        self.x_line_points = np.reshape(
            line_points[np.newaxis, :]
            * xwidths[:, np.newaxis]
            + uniq_xknots[:-1][:, np.newaxis],
            (-1,))
        self.y_line_points = np.reshape(
            line_points[np.newaxis, :]
            * ywidths[:, np.newaxis]
            + uniq_yknots[:-1][:, np.newaxis],
            (-1,))

        # FIXME: Why don't I have to divide this by 4 to accommodate the change in
        # interval?
        # Answer(doktay): Because for some reason quadpy.c2 weights sum to 1 instead of 4.
        self.weights = np.reshape(scheme.weights, (1, 1, -1))
        self.line_weights = np.reshape(line_scheme.weights / 2, (1, -1))
        #self.line_weights = np.ones_like(self.line_weights)

    def num_quad_pts(self):
        return self.points.shape[0]

    def get_deformation_fn(self):
        ''' Get a function that produces deformations

        Takes in control points and returns a deformation for each quad point.

        This is assumed to be in cm.
        '''
        def deformation_fn(ctrl):
            return bspline2d(
                self.points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            )
        return deformation_fn

    def get_line_deformation_fn(self):
        ''' Get a function that produces deformations along a certain side of the cell.

        The line depends on the orientation: 0 - left, 1 - top, 2 - right, 3 - bottom
        '''
        def line_deformation_fn(ctrl, orientation):
            # Create a (# quad pts, 2) array
            n_points = self.y_line_points.shape[0]
            points = jax.lax.cond(
                orientation == 0,
                lambda _: np.stack(
                    [np.zeros(n_points), self.y_line_points], axis=-1),
                lambda _: jax.lax.cond(
                    orientation == 1,
                    lambda _: np.stack(
                        [self.x_line_points, np.ones(n_points)], axis=-1),
                    lambda _: jax.lax.cond(
                        orientation == 2,
                        lambda _: np.stack(
                            [np.ones(n_points), self.y_line_points], axis=-1),
                        lambda _: np.stack(
                            [self.x_line_points, np.zeros(n_points)], axis=-1),
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            )

            return bspline2d(
                points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            )

        return line_deformation_fn

    def get_jacobian_u_fn(self):
        ''' Take control points, return 2x2 Jacobians wrt quad points. '''
        def jacobian_u_fn(ctrl):
            return bspline2d_derivs(
                self.points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            )
        return jacobian_u_fn

    def get_line_derivs_u_fn(self):
        ''' Take control points, return 2x1 Jacobians wrt boundary quad points. '''
        def jacobian_u_fn(ctrl, orientation):
            points = jax.lax.cond(
                np.logical_or(orientation == 0, orientation == 2),
                lambda _: self.y_line_points,
                lambda _: self.x_line_points,
                operand=None,
            )

            knots = jax.lax.cond(
                np.logical_or(orientation == 0, orientation == 2),
                lambda _: self.yknots,
                lambda _: self.xknots,
                operand=None,
            )

            sel_ctrl = jax.lax.cond(
                orientation == 0,
                lambda _: ctrl[0, :],
                lambda _: jax.lax.cond(
                    orientation == 1,
                    lambda _: ctrl[:, -1],
                    lambda _: jax.lax.cond(
                        orientation == 2,
                        lambda _: ctrl[-1, :],
                        lambda _: ctrl[:, 0],
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            )

            return bspline1d_derivs_hand(
                points,
                sel_ctrl,
                knots,
                self.spline_deg,
            )

        return jacobian_u_fn

    def get_jacobian_ctrl_fn(self):
        ''' Take control points, return Jacobian wrt control points. '''
        def jacobian_ctrl_fn(ctrl):
            return bspline2d_derivs_ctrl(
                self.points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg,
            )
        return jacobian_ctrl_fn

    def get_energy_fn(self):
        ''' Get the energy density function associated with the material model.

        The various material properties are in GPa, and Pa = N/m^3 so GPa is
        billons of Newtons per cubic meter = GN/m^3.  To get a sense of how this
        varies, it is roughly quadratic in the log of the scale of deformation
        gradient.
        '''
        return self.material.get_energy_fn()

    def get_quad_fn(self):
        def quad_fn(ordinates):

            # Need to get into kind of a fancy shape to both broadcast correctly
            # and to be able to sum in two stages with quadrature weights.
            ords = np.reshape(ordinates,
                              (*self.span_volumes.shape, -1, *ordinates.shape[1:]))

            # The transpose makes it possible to sum over additional dimensions.
            return np.sum(np.sum(self.weights.T * ords.T, axis=-3)
                          * self.span_volumes.T, axis=(-1, -2)).T

        return quad_fn

    def get_line_quad_fn(self):
        def line_quad_fn(ordinates, orientation):
            widths = jax.lax.cond(
                np.logical_or(orientation == 0, orientation == 2),
                lambda _: self.ywidths,
                lambda _: self.xwidths,
                operand=None,
            )
            ords = np.reshape(
                ordinates, (*widths.shape, -1, *ordinates.shape[1:]))

            return np.sum(np.sum(self.line_weights.T * ords.T, axis=-2) * widths, axis=-1).T

        return line_quad_fn

    def get_ctrl_shape(self):
        return self.num_xctrl, self.num_yctrl, 2
