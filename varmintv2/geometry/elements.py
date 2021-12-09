from abc import ABC, abstractmethod
from typing import Callable, Iterable, Tuple
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import quadpy

from varmintv2.utils.typing import CtrlArray, ArrayND, Array2D, Array3D
from varmintv2.geometry.bsplines import (
    bspline1d_derivs,
    mesh,
    bspline2d,
    bspline2d_basis,
    bspline2d_basis_derivs,
    bspline2d_derivs,
    bspline2d_derivs_ctrl,
)


class Element(ABC):
    """ Class representing local geometry.

    The physics and geometry in Varmint are defined as an aggregation over
    many local elements.
    
    Each element represents a map between a parent space and
    a part of the geometry in the physical space, where the mapping
    is parameterized by control points.
    
    Integration is handled separately for each element, and is typically
    through quadrature.
    
    Each element contains the quadrature points for integration (which live
    in the parent space). It contains functions to map between parent and
    physical, as well as to get Jacobians between the two spaces.

    The interiors and boundaries are handled separately. There are separate
    quadrature points for each. Boundaries are indexed by an integer,
    with a maximum specified by Element.num_boundaries().

    TODO(doktay): Elements should have an internal sparsity pattern specified.
    This will be used by Geometry to construct the sparsity pattern of
    the Jacobian.
    """

    quad_points: ArrayND

    @abstractmethod
    def get_map_fn(self, points) -> Callable[[CtrlArray], Array2D]:
        """Get function to compute the parent -> physical map at
        specified list of points.
        
        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and returns the deformation of all points provided. Shape $(N, n_d)$,
        where $N$ is number of points given as an argument and $n_d$ is
        embedding dimension.

        """
        pass

    @abstractmethod
    def get_quad_map_fn(self) -> Callable[[CtrlArray], Array2D]:
        """Get function to compute the parent -> physical map at
        each interior quad point, given control points.

        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and returns the deformation of all quad points. Shape $(N, n_d)$, where
        $N$ is number of quad points and $n_d$ is embedding dimension.

        """
        pass

    @abstractmethod
    def get_quad_map_boundary_fn(self, index: int) -> Callable[[CtrlArray], Array2D]:
        """Get function to compute the parent -> physical map at
        each boundary quad point, given control points.

        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and returns the deformation of all quad points. Shape $(N, n_d)$, where
        $N$ is number of quad points and $n_d$ is embedding dimension.

        """
        pass

    @abstractmethod
    def get_map_jac_fn(self, points) -> Callable[[CtrlArray], Array2D]:
        """Get function to compute Jacobian of the parent -> physical map
        at all specified points, given control points.
        
        Returns:
        ========

        Function that takes in control points of shape self.ctrl_shape
        and returns a 3d array of Jacobians of physical space wrt parent space
        for each given point. Ex: In 2D, with N points, the function would
        return shape (N, 2, 2).

        """
        pass

    @abstractmethod
    def get_quad_map_jac_fn(self) -> Callable[[CtrlArray], Array3D]:
        """Get function to compute Jacobian of the parent -> physical map
        at all quad points at the interior of the element, given control points.
        
        Returns:
        ========

        Function that takes in control points of shape self.ctrl_shape
        and returns a 3d array of Jacobians of physical space wrt parent space
        for each quad point. Ex: In 2D, with N quad points, the function would
        return shape (N, 2, 2).

        """
        pass

    @abstractmethod
    def get_quad_map_boundary_jac_fn(self, index: int) -> Callable[[CtrlArray], Array2D]:
        """Get function to compute Jacobian of the parent -> physical map
        at all quad points at the boundary of the element, given control points.
        As the boundary is parameterized by a scalar, the Jacobian will be
        of size $(n_d,)$, where $n_d$ is the size of the embedding space.
        
        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and a boundary index and returns a 3d array of Jacobians of
        physical space wrt parent space for each quad point.
        Ex: In 2D, with N quad points, the function would return shape (N, 2).

        """
        pass

    @abstractmethod
    def get_ctrl_jacobian_fn(self, points) -> Callable[[CtrlArray], ArrayND]:
        """Get function that computes the Jacobian with respect to the control
        points at each given point, given control points.

        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and returns the Jacobian of the deformation at all given points with
        respect to control points.
        Shape is $(N, n_d, \texttt{*self.ctrl_shape})$ where $N$ is
        number points, $n_d$ is embedding dimension.

        """
        pass

    @abstractmethod
    def get_quad_ctrl_jacobian_fn(self) -> Callable[[CtrlArray], ArrayND]:
        """Get function that computes the Jacobian with respect to the control
        points at each interior quad point, given control points.

        Returns:
        ========
        Function that takes in control points of shape self.ctrl_shape
        and returns the Jacobian of the deformation at all quad points with
        respect to control points.
        Shape is $(N, n_d, \texttt{*self.ctrl_shape})$ where $N$ is
        number of quad points, $n_d$ is embedding dimension.

        """
        pass

    @property
    @abstractmethod
    def ctrl_shape(self) -> Iterable[int]:
        """Return shape of expected control points."""
        pass

    @property
    @abstractmethod
    def num_boundaries(self) -> int:
        """Return the number of boundaries for this element."""
        pass

    @property
    @abstractmethod
    def num_quad_pts(self) -> int:
        """Return the number of interior quad points in this element."""
        pass
    
    @abstractmethod
    def num_boundary_quad_pts(self, index: int) -> int:
        """Return the number of boundary quad points for a boundary index."""
        pass

    @abstractmethod
    def get_quad_fn(self) -> Callable[[ArrayND], float]:
        """Compute interior integration by quadrature.
        
        Returns a function that given a quantity computed at all quad points,
        computes the integral of that quantity. The input must have shape
        $(N, *)$, where $N$ is number of quad points on the interior
        of the element.

        Note that integration is done in the parent space, so for
        e.g. Patch2D it will always be $[0, 1]^2$.

        """
        pass

    @abstractmethod
    def get_boundary_quad_fn(self, index: int) -> Callable[[ArrayND], float]:
        """Compute boundary integration by quadrature.
        
        Returns a function that given a quantity computed at all quad points,
        computes the integral of that quantity. The input must have shape
        $(N, *)$, where $N$ is number of quad points on the boundary
        of the element.

        Note that integration is done in the parent space, so for
        e.g. Patch2D it will always be $[0, 1]$.

        """
        pass

    @property
    @abstractmethod
    def n_d(self):
        """Embedding dimension."""
        pass

    @abstractmethod
    def get_sparsity_pattern(self) -> Array2D:
        """Return the sparsity pattern for this element.
        
        Returns a array of shape (n_ind, 2) of pairs of control point
        indices that would have non-zero Jacobian entries within the
        element. n_ind is the number of non-zero entries in the local
        Jacobian for this Element.

        Local control point ordering is in flatten order:
            ctrl_points.reshape(-1, n_d)
        
        Control point ordering in global array should be an offset to its
        local ordering, to enable efficient translation between local and global
        numbering.
        """
        pass


class Patch2D(Element):
    """Element represented by a BSpline 2D patch.

    A patch corresponds to a two-dimensional BSpline with tensor product basis
    functions. They are individual maps from $[0,1]^2$ to $\mathbb{R}^2$,
    parameterized by control points.

    Since the parent space is $[0, 1]^2$, integration is done via Gaussian
    Quadrature.
    """

    def __init__(self, xknots, yknots, spline_deg, quad_deg):
        """Constructor for two dimensional patch.

        Parameters:
        -----------
         - xknots: A length-M one-dimensional array of bspline knots for the x
                   dimension. These are assumed to be in non-decreasing order from
                   0.0 to 1.0.

         - yknots: A length-N one-dimensional array of bspline knots for the y
                   dimension. These are assumed to be in non-decreasing order from
                   0.0 to 1.0.

         - spline_deg: The degree of the bspline.

        """

        self.xknots = xknots
        self.yknots = yknots
        self.spline_deg = spline_deg
        self.quad_deg = quad_deg

        # Determine the number of control points.
        num_xknots = self.xknots.shape[0]
        num_yknots = self.yknots.shape[0]

        self.num_xctrl = num_xknots - self.spline_deg - 1
        self.num_yctrl = num_yknots - self.spline_deg - 1

        self.__compute_quad_points()

    def __compute_quad_points(self):

        # Each knot span has its own quadrature.
        uniq_xknots = onp.unique(self.xknots)
        uniq_yknots = onp.unique(self.yknots)

        xwidths = onp.diff(uniq_xknots)
        ywidths = onp.diff(uniq_yknots)

        # We need the span volumes for performing integration later.
        self.span_volumes = xwidths[:, onp.newaxis] * ywidths[onp.newaxis, :]
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

        self.quad_points = onp.reshape(points[onp.newaxis, onp.newaxis, :, :]
                                       * width_mesh[:, :, onp.newaxis, :]
                                       + offset_mesh[:, :, onp.newaxis, :],
                                       (-1, 2))

        self.x_line_points = onp.reshape(
            line_points[onp.newaxis, :]
            * xwidths[:, onp.newaxis]
            + uniq_xknots[:-1][:, onp.newaxis],
            (-1,))
        self.y_line_points = onp.reshape(
            line_points[onp.newaxis, :]
            * ywidths[:, onp.newaxis]
            + uniq_yknots[:-1][:, onp.newaxis],
            (-1,))

        # FIXME: Why don't I have to divide this by 4 to accommodate the change in
        # interval?
        # Answer(doktay): Because for some reason quadpy.c2 weights sum to 1 instead of 4.
        self.weights = onp.reshape(scheme.weights, (1, 1, -1))
        self.line_weights = onp.reshape(line_scheme.weights / 2, (1, -1))
        #self.line_weights = np.ones_like(self.line_weights)

    def get_map_fn(self, points):
        def deformation_fn(ctrl):
            return bspline2d(
                points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            )
        return deformation_fn

    def get_quad_map_fn(self):
        """ Get a function that produces deformations

        Takes in control points and returns a deformation for each quad point.
        """
        return self.get_map_fn(self.quad_points)

    def get_quad_map_boundary_fn(self, orientation):
        """ Get a function that produces deformations along a certain side of the cell.

        The line depends on the orientation: 0 - left, 1 - top, 2 - right, 3 - bottom
        """

        if orientation == 0:
            y_n_points = self.y_line_points.shape[0]
            points = jnp.stack(
                [jnp.zeros(y_n_points), self.y_line_points], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 1:
            x_n_points = self.x_line_points.shape[0]
            points = jnp.stack(
                [self.x_line_points, jnp.ones(x_n_points)], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 2:
            y_n_points = self.y_line_points.shape[0]
            points = jnp.stack(
                [jnp.ones(y_n_points), self.y_line_points], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 3:
            x_n_points = self.x_line_points.shape[0]
            points = jnp.stack(
                [self.x_line_points, jnp.zeros(x_n_points)], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        else:
            raise ValueError(f"Invalid boundary index {index} for Patch2D.")

        return line_deformation_fn

    def get_map_jac_fn(self, points):
        """ Take control points, return 2x2 Jacobians wrt quad points. """
        def map_jac_fn(ctrl):
            return bspline2d_derivs(
                points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            )
        return map_jac_fn

    def get_quad_map_jac_fn(self):
        """ Take control points, return 2x2 Jacobians wrt quad points. """
        return self.get_map_jac_fn(self.quad_points)

    def get_quad_map_boundary_jac_fn(self, orientation):
        """ Take control points, return 2x1 Jacobians wrt boundary quad points. """
        if orientation == 0:
            def map_boundary_jac_fn(ctrl):
                points = self.y_line_points
                knots = self.yknots

                sel_ctrl = ctrl[0, :]

                return bspline1d_derivs(
                    points,
                    sel_ctrl,
                    knots,
                    self.spline_deg,
                )
        elif orientation == 1:
            def map_boundary_jac_fn(ctrl):
                points = self.x_line_points
                knots = self.xknots

                sel_ctrl = ctrl[:, -1]

                return bspline1d_derivs(
                    points,
                    sel_ctrl,
                    knots,
                    self.spline_deg,
                )
        elif orientation == 2:
            def map_boundary_jac_fn(ctrl):
                points = self.y_line_points
                knots = self.yknots

                sel_ctrl = ctrl[-1, :]

                return bspline1d_derivs(
                    points,
                    sel_ctrl,
                    knots,
                    self.spline_deg,
                )
        elif orientation == 3:
            def map_boundary_jac_fn(ctrl):
                points = self.x_line_points
                knots = self.xknots

                sel_ctrl = ctrl[:, 0]

                return bspline1d_derivs(
                    points,
                    sel_ctrl,
                    knots,
                    self.spline_deg,
                )
        else:
            raise ValueError(f"Invalid boundary index {index} for Patch2D.")

        return map_boundary_jac_fn

    def get_ctrl_jacobian_fn(self, points):
        """ Take control points, return Jacobian wrt control points. """
        def jacobian_ctrl_fn(ctrl):
            return bspline2d_derivs_ctrl(
                points,
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg,
            )
        return jacobian_ctrl_fn

    def get_quad_ctrl_jacobian_fn(self):
        """ Take control points, return Jacobian wrt control points. """
        return self.get_ctrl_jacobian_fn(self.quad_points)

    def get_quad_fn(self):
        def quad_fn(ordinates):

            # Need to get into kind of a fancy shape to both broadcast correctly
            # and to be able to sum in two stages with quadrature weights.
            ords = jnp.reshape(ordinates,
                              (*self.span_volumes.shape, -1, *ordinates.shape[1:]))

            # The transpose makes it possible to sum over additional dimensions.
            return jnp.sum(jnp.sum(self.weights.T * ords.T, axis=-3)
                           * self.span_volumes.T, axis=(-1, -2)).T

        return quad_fn

    def get_boundary_quad_fn(self, orientation):
        if orientation == 0 or orientation == 2:
            def boundary_quad_fn(ordinates):
                widths = self.ywidths
                ords = jnp.reshape(
                    ordinates, (*widths.shape, -1, *ordinates.shape[1:])
                )

                return jnp.sum(jnp.sum(self.line_weights.T * ords.T, axis=-2) * widths, axis=-1).T
        elif orientation == 1 or orientation == 3:
            def boundary_quad_fn(ordinates):
                widths = self.xwidths
                ords = jnp.reshape(
                    ordinates, (*widths.shape, -1, *ordinates.shape[1:])
                )

                return jnp.sum(jnp.sum(self.line_weights.T * ords.T, axis=-2) * widths, axis=-1).T
        else:
            raise ValueError(f"Invalid boundary index {orientation} for Patch2D.")

        return boundary_quad_fn

    @property
    def ctrl_shape(self):
        return self.num_xctrl, self.num_yctrl, 2
    
    @property
    def num_boundaries(self):
        # Orientation: 0 - left
        #              1 - top
        #              2 - right
        #              3 - bottom

        return 4

    @property
    def num_quad_pts(self):
        return self.quad_points.shape[0]

    def num_boundary_quad_pts(self, index):
        if index == 0 or index == 2:
            return self.y_line_points.shape[0]
        elif index == 1 or index == 3:
            return self.x_line_points.shape[0]
        else:
            raise ValueError(f"Invalid boundary index {index} for Patch2D.")
    
    def get_sparsity_pattern(self) -> Array2D:
        local_patch_indices = onp.arange(self.num_xctrl * self.num_yctrl)
        
        # TODO(doktay): This is an overestimate of the sparsity pattern.
        # In reality it should depend on the spline degree, but here for
        # simplicity we just say all points are connected to all points.
        mgrid_indices = onp.stack(onp.meshgrid(
            local_patch_indices, local_patch_indices), axis=-1)

        return mgrid_indices

    @property
    def n_d(self):
        return 2