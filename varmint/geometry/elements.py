from abc import ABC, abstractmethod
from typing import Callable, Iterable, Tuple
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

from numpy.polynomial.legendre import leggauss

from varmint.utils.typing import CtrlArray, ArrayND, Array2D, Array3D
from varmint.geometry.bsplines import (
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
    def get_map_fn_fixed_ctrl(self, ctrl):
        """Get function to compute the parent -> physical map for given
        control points.

        Usually you would want to use get_map_fn, but this is also
        sometimes necessary.

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

    @abstractmethod
    def get_boundary_path(self) -> Array2D:
        """Return a path of points that traces the element in the parent domain.
        
        Used for generating movies/images.
        """
        pass


class IsoparametricQuad2D(Element):
    """Standard bilinear quadrilateral element.
    
    4 control points define the domain:
        - bottom left
        - bottom right
        - top right
        - top left

    TODO(doktay): Extend this to arbitrary order Lagrange elements.
    The implementation should be a bit more general without being particularly
    more complicated.
    """

    def __init__(self, quad_deg):
        """Constructor for Bilinear quadrilateral.

        Parameters:
        -----------
         - spline_deg: The degree of the bspline.

        """
        self.quad_deg = quad_deg
        self.__compute_quad_points()

    @staticmethod
    def __single_point_map(point, ctrl):
        """Mapping of a single point in R2.

        Standard bilinear quadrilateral element.
        The parent space is [-1,1]x[-1,1].

        point is (2,), ctrl is (4, 2)
        """

        shape_fn = 0.25 * jnp.array([
            (1-point[0]) * (1-point[1]),
            (1+point[0]) * (1-point[1]),
            (1+point[0]) * (1+point[1]),
            (1-point[0]) * (1+point[1]),
        ])

        return shape_fn @ ctrl

    def __compute_quad_points(self):
        # Ask quadpy for a quadrature scheme.
        scheme = quadpy.c2.get_good_scheme(self.quad_deg)

        # Scheme for line integrals
        line_scheme = quadpy.c1.gauss_legendre(3 * self.quad_deg)

        self.quad_points = scheme.points.T
        self.line_points = line_scheme.points

        self.weights = scheme.weights * 4  # quadpy c2 weights sum to 1 instead
                                           # of 4 even though the domain is
                                           # [-1,1]x[-1,1]
        self.line_weights = line_scheme.weights

    def get_map_fn(self, points):
        vmap_map = jax.vmap(IsoparametricQuad2D.__single_point_map,
                            in_axes=(0, None), out_axes=0)
        def deformation_fn(ctrl):
            return vmap_map(points, ctrl)
        return deformation_fn
    
    def get_map_fn_fixed_ctrl(self, ctrl):
        def deformation_fn(point):
            return IsoparametricQuad2D.__single_point_map(point, ctrl)

        return deformation_fn

    def get_quad_map_fn(self):
        return self.get_map_fn(self.quad_points)

    def get_quad_map_boundary_fn(self, orientation):
        """ Get a function that produces deformations along a certain side of the cell.

        The line depends on the orientation: 0 - left, 1 - top, 2 - right, 3 - bottom
        """
        n_points = self.line_points.shape[0]

        if orientation == 0:
            points = jnp.stack(
                [-jnp.ones(n_points), self.line_points], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 1:
            points = jnp.stack(
                [self.line_points, jnp.ones(n_points)], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 2:
            points = jnp.stack(
                [jnp.ones(n_points), self.line_points], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        elif orientation == 3:
            points = jnp.stack(
                [self.line_points, -jnp.ones(n_points)], axis=-1)
            line_deformation_fn = self.get_map_fn(points)

        else:
            raise ValueError(f"Invalid boundary index {index} for Patch2D.")

        return line_deformation_fn

    def get_map_jac_fn(self, points):
        jac_map = jax.jacfwd(IsoparametricQuad2D.__single_point_map, argnums=0)
        vmap_jac_map = jax.vmap(jac_map, in_axes=(0, None), out_axes=0)

        def map_jac_fn(ctrl):
            return vmap_jac_map(points, ctrl)
        return map_jac_fn

    def get_quad_map_jac_fn(self):
        return self.get_map_jac_fn(self.quad_points)

    def get_quad_map_boundary_jac_fn(self, orientation):
        """ Take control points, return 2x1 Jacobians wrt boundary quad points. """
        n_points = self.line_points.shape[0]
        jac_map = jax.jacfwd(IsoparametricQuad2D.__single_point_map, argnums=0)
        vmap_jac_map = jax.vmap(jac_map, in_axes=(0, None), out_axes=0)

        if orientation == 0:
            points = jnp.stack(
                [-jnp.ones(n_points), self.line_points], axis=-1)

            def map_boundary_jac_fn(ctrl):
                return vmap_jac_map(points, ctrl)[..., 1]

        elif orientation == 1:
            points = jnp.stack(
                [self.line_points, jnp.ones(n_points)], axis=-1)

            def map_boundary_jac_fn(ctrl):
                return vmap_jac_map(points, ctrl)[..., 0]

        elif orientation == 2:
            points = jnp.stack(
                [jnp.ones(n_points), self.line_points], axis=-1)

            def map_boundary_jac_fn(ctrl):
                return vmap_jac_map(points, ctrl)[..., 1]

        elif orientation == 3:
            points = jnp.stack(
                [self.line_points, -jnp.ones(n_points)], axis=-1)

            def map_boundary_jac_fn(ctrl):
                return vmap_jac_map(points, ctrl)[..., 0]

        else:
            raise ValueError(f"Invalid boundary index {index} for Patch2D.")

        return map_boundary_jac_fn

    def get_ctrl_jacobian_fn(self, points):
        jac_map = jax.jacfwd(IsoparametricQuad2D.__single_point_map, argnums=1)
        vmap_jac_map = jax.vmap(jac_map, in_axes=(0, None), out_axes=0)

        def map_jac_fn(ctrl):
            return vmap_jac_map(points, ctrl)
        return map_jac_fn

    def get_quad_ctrl_jacobian_fn(self):
        """ Take control points, return Jacobian wrt control points. """
        return self.get_ctrl_jacobian_fn(self.quad_points)

    @property
    def ctrl_shape(self):
        return 4, 2
    
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
        if index in [0, 1, 2, 3]:
            return self.line_points.shape[0]
        else:
            raise ValueError(f"Invalid boundary index {index} for IsoparametericQuad2D.")

    @property
    def n_d(self):
        return 2

    def get_sparsity_pattern(self) -> Array2D:
        local_patch_indices = onp.arange(4)
        
        mgrid_indices = onp.stack(onp.meshgrid(
            local_patch_indices, local_patch_indices), axis=-1)

        return mgrid_indices

    def get_quad_fn(self):
        def quad_fn(ordinates):
            return jnp.sum(self.weights.T * ordinates.T, axis=-1).T

        return quad_fn

    def get_boundary_quad_fn(self, orientation):
        def boundary_quad_fn(ordinates):
            return jnp.sum(self.line_weights * ordinates.T, axis=-1).T

        return boundary_quad_fn

    def get_boundary_path(self):
        N = 10
        uu = onp.linspace(-1+1e-6, 1-1e-6, N)
        path = onp.hstack([
            onp.vstack([uu[0]*onp.ones(N), uu]),
            onp.vstack([uu, uu[-1]*onp.ones(N)]),
            onp.vstack([uu[-1]*onp.ones(N), uu[::-1]]),
            onp.vstack([uu[::-1], uu[0]*onp.ones(N)]),
        ]).T

        return path


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

        # Get quadrature weights from scipy.
        points, weights = leggauss(self.quad_deg)
        line_points, line_weights = leggauss(3 * self.quad_deg)

        # Change the domain from (-1,1)^2 to (0,1)^2
        points = points / 2 + 0.5
        points = onp.stack(onp.meshgrid(points, points), axis=-1).reshape(-1, 2)

        weights = weights.reshape(-1, 1) @ weights.reshape(1, -1) / 4
        weights = weights.reshape(-1, 1)

        line_points = line_points / 2 + 0.5

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

        self.weights = onp.reshape(weights, (1, 1, -1))
        self.line_weights = onp.reshape(line_weights / 2, (1, -1))
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

    def get_map_fn_fixed_ctrl(self, ctrl):
        def deformation_fn(point):
            return bspline2d(
                point[onp.newaxis, :],
                ctrl,
                self.xknots,
                self.yknots,
                self.spline_deg
            ).squeeze()

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
        p = self.spline_deg
        
        # Bandwidth of order p basis functions is 2p+1
        bw = onp.arange(-p, p+1)

        # Create 2d bandwidth offset matrix
        # (2p+1) x (2p+1) x 2
        twod_bw = onp.stack(onp.meshgrid(bw, bw), axis=-1)

        xs = onp.arange(self.num_xctrl)
        ys = onp.arange(self.num_yctrl)

        # n_x x n_y x 2
        all_indices = onp.stack(onp.meshgrid(xs, ys), axis=-1)

        # Create dummy indices for broadcasting
        # n_x x n_y x 1 x 1 x 2 
        all_indices = onp.expand_dims(all_indices, axis=(-2, -3))

        # Broadcasting magic
        # n_x x n_y x (2p+1) x (2p+1) x 2
        pairs = twod_bw + all_indices

        # We want to concatenate all_indices with pairs:
        all_indices = onp.broadcast_to(all_indices, pairs.shape)
        concat = onp.concatenate((all_indices, pairs), axis=-1)

        # all_pairs contains all candidates, but many of them will be
        # out of bounds. Filter those out.
        all_pairs = concat.reshape((-1, 4))
        valid_indices = onp.all(
            (all_pairs[..., 2:] >= onp.array([0, 0])) & \
            (all_pairs[..., 2:] <  onp.array([self.num_xctrl, self.num_yctrl])), axis=-1)

        valid_pairs = all_pairs[valid_indices]

        # Compute raveled indices
        p1 = onp.ravel_multi_index((valid_pairs[:, 0], valid_pairs[:, 1]), (self.num_xctrl, self.num_yctrl))
        p2 = onp.ravel_multi_index((valid_pairs[:, 2], valid_pairs[:, 3]), (self.num_xctrl, self.num_yctrl))

        return onp.stack((p1, p2), axis=-1)

    @property
    def n_d(self):
        return 2

    def get_boundary_path(self, N=20):
        uu = onp.linspace(1e-6, 1-1e-6, N)
        path = onp.hstack([
            onp.vstack([uu[0]*onp.ones(N), uu]),
            onp.vstack([uu, uu[-1]*onp.ones(N)]),
            onp.vstack([uu[-1]*onp.ones(N), uu[::-1]]),
            onp.vstack([uu[::-1], uu[0]*onp.ones(N)]),
        ]).T

        return path
