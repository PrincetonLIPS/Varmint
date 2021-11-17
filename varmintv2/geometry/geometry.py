from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from sympy.core.symbol import Str

from varmintv2.utils.typing import Array1D, ArrayND
from varmintv2.geometry.elements import Element


class Geometry(ABC):
    """ Class representing global geometry.
    
    Represents the combination of elements over the domain. Keeps track of
    a way to initialize the space, mapping between global and local coordinates,
    traction and Dirichlet conditions, global Lagrangian/Energy, and sparsity
    pattern of degrees of freedom.
    """

    @abstractmethod
    def get_global_local_maps(self) \
        -> Tuple[Callable[[ArrayND, ArrayND], Tuple[ArrayND, ArrayND]],
                 Callable[[ArrayND, ArrayND, ArrayND, ArrayND], Tuple[ArrayND, ArrayND]]]:
        """Get the mapping functions between global and local coordinates.
        
        Returns:
        ========

        Two functions, one for global -> local and another for local -> global.

        The global -> local function takes in 4 ndarray arguments:
            - global positions
            - global velocities
            - values of fixed positions
            - values of fixed velocities
        
        All the above arrays should have the same shape. The function will then
        return the local positions and velocities that obey the fixed values.

        The local -> global function takes in 2 ndarray arguments:
            - local positions
            - local velocities

        All the above arrays should have the same shape. The function will then
        return the global positions and velocities. It will choose an arbitrary
        value amongst incident local control points (the user should ensure the
        values are the same).
        """
        pass

    def unflatten_dynamics_sequence(self, positions, velocities,
                                    fixed_positions, fixed_velocities):
        """Helper function to convert a sequence of global coordinates to local."""
        _, g2l_map = self.get_global_local_maps()
        local_pos, local_vel = zip(
            *[g2l_map(q, p, f, v) for q, p, f, v in \
                zip(positions, velocities, fixed_positions, fixed_velocities)]
        )

        return local_pos, local_vel
    
    @abstractmethod
    def get_lagrangian_fn(self):
        """Return the Lagrangian.
        
        Returns a function that computes the Lagrangian over the domain
        given global coordinates.

        Returns:
        ========

        The Lagrangian function takes in the arguments:
            - cur_g_position: Current global position.
            - cur_g_velocity: Current global velocity.
            - ref_l_position: Reference local position.
            - fix_l_position: Fixed values of local position.
            - fix_l_velocity: Fixed values of local velocity.
            - traction: Values of traction force.

        The function then returns the Lagrangian as a scalar.
        """
        pass

    @abstractmethod
    def get_traction_fn(self):
        """Get a function that returns traction values through time.
        
        Returns:
        ========

        Returns a function that takes time (t) as an argument, and returns
        the traction forces on each boundary. The return value can be used
        as an input to the Lagrangian function.
        """
        pass

    @abstractmethod
    def get_fixed_locs_fn(self, ref_l_position):
        """Get a function that returns Dirichlet conditions through time.
        
        Returns:
        ========

        Returns a function that takes time (t) as an argument, and returns
        the values of the Dirichlet constrained positions and velocities.
        The return values can be used as an input to the Lagrangian function.
        """
        pass


@dataclass
class SEBoundaryConditions:
    """Contains prerequisites to construct boundary conditions in a SingleElementGeometry."""

    # Number of elements and element description.
    n_elements: int
    element: Element

    # Maps a Dirichlet boundary condition label to a boolean array in local
    # coordinates that selects out control points in the group. Each item is
    # of size (n_elements, element.ctrl_shape[:-1]).
    dirichlet_labels: Dict[Str, ArrayND]

    # Maps a traction boundary condition label to a boolean array 
    # that selects out boundaries in the group. Each item is
    # of size (n_elements, element.num_boundaries).
    traction_labels: Dict[Str, ArrayND]

    # Maps a Dirichlet boundary condition label to functions that map time to
    # the displacement and velocity of control points (given time) within the group.
    dirichlet_fns: Dict[Str, Tuple[Callable, Callable]] = field(default_factory=dict)

    # Maps a traction boundary condition label to a function that gives the
    # traction force on a control point (given time) within the group.
    traction_fns: Dict[Str, Callable] = field(default_factory=dict)

    @property
    def active_traction_boundaries(self):
        """Helper function to aggregate all active traction boundaries."""
        return onp.zeros((self.n_elements, self.element.num_boundaries)) + \
                    sum(self.traction_labels[g]
                        for g in self.traction_labels)

    def register_dirichlet_bc(self, group):
        """Decorator function to declare a Dirichlet condition."""

        def inner(fn):
            vel_fn = jax.jacfwd(fn)  # Differentiate position to get velocity.

            def decorated(t):
                return fn(t) * self.dirichlet_labels[group][..., jnp.newaxis]

            def decorated_vel(t):
                return vel_fn(t) * self.dirichlet_labels[group][..., jnp.newaxis]

            self.dirichlet_fns[group] = (decorated, decorated_vel)
            return decorated

    def register_traction_bc(self, group):
        """Decorator function to declare a traction condition."""

        def inner(fn):
            def decorated(t):
                return fn(t) * self.traction_labels[group][..., jnp.newaxis]

            self.traction_fns[group] = decorated
            return decorated
        return inner


@dataclass
class SEGlobalLocalMap:
    """Contains prerequisites to construct global <-> local maps in a SingleElementGeometry."""

    # Number of elements and element description.
    n_elements: int
    element: Element

    # Gives the component of each control point in local coordinates. Control
    # points in the same component are considered the same. This has shape
    # (n_elements, element.ctrl_shape[:-1]). Note that the last dimension of
    # control points is not included, as dimensions (e.g. x and y) always
    # belong to the same component.
    index_array: ArrayND

    # Linear array of components that should be considered fixed.
    fixed_labels: Array1D

    # Linear array of components that should not be considered fixed.
    nonfixed_labels: Array1D

    # The number of total components.
    n_components: int

    def get_global_local_maps(self) -> Tuple[Callable, Callable]:
        def local_to_global(local_pos, local_vel):
            kZeros = jnp.zeros((self.n_components, self.element.n_d))

            global_pos = jax.ops.index_update(kZeros, self.index_array, local_pos)
            global_vel = jax.ops.index_update(kZeros, self.index_array, local_vel)

            global_pos = jnp.take(global_pos, self.nonfixed_labels, axis=0)
            global_vel = jnp.take(global_vel, self.nonfixed_labels, axis=0)

            return global_pos.flatten(), global_vel.flatten()

        def global_to_local(global_pos, global_vel, fixed_pos, fixed_vel):
            kZeros = jnp.zeros((self.n_components, self.element.n_d))

            fixed_locs = jax.ops.index_update(
                kZeros, self.index_array, fixed_pos)
            fixed_locs = jnp.take(fixed_locs, self.fixed_labels, axis=0)

            fixed_vels = jax.ops.index_update(
                kZeros, self.index_array, fixed_vel)
            fixed_vels = jnp.take(fixed_vels, self.fixed_labels, axis=0)

            local_pos = kZeros
            local_vel = kZeros

            global_pos = global_pos.reshape((-1, self.element.n_d))
            global_vel = global_vel.reshape((-1, self.element.n_d))

            local_pos = jax.ops.index_update(
                local_pos, self.nonfixed_labels, global_pos)
            local_vel = jax.ops.index_update(
                local_vel, self.nonfixed_labels, global_vel)

            fixed_pos = jax.ops.index_update(local_pos, self.fixed_labels,
                                             fixed_locs)
            fixed_vel = jax.ops.index_update(local_vel, self.fixed_labels,
                                             fixed_vels)

            return jnp.take(fixed_pos, self.index_array, axis=0), \
                jnp.take(fixed_vel, self.index_array, axis=0)

        return local_to_global, global_to_local


class SingleElementGeometry(Geometry):
    """Geometry composed of a single type of Element."""

    element: Element
    element_lagrangian: Callable

    boundary_conditions: SEBoundaryConditions
    globallocal_params: SEGlobalLocalMap

    ###########################################################
    ##### Utilities to keep track of boundary conditions. #####
    ###########################################################
    
    def get_fixed_locs_fn(self, ref_l_position):
        pos_fns = []
        vel_fns = []

        for group in self.boundary_conditions.dirichlet_labels:
            pos, vel = self.boundary_conditions.dirichlet_fns.get(group, (None, None))

            if pos is not None:
                pos_fns.append(pos)
                vel_fns.append(vel)
        
        def fixed_locs_fn(t):
            return ref_l_position + sum(fn(t) for fn in pos_fns)
        
        def fixed_vels_fn(t):
            return jnp.zeros_like(ref_l_position) + sum(fn(t) for fn in vel_fns)
        
        return fixed_locs_fn, fixed_vels_fn
    
    def get_traction_fn(self):
        fns = []
        for group in self.boundary_conditions.traction_labels:
            fns.append(self.boundary_conditions.traction_fns.get(group, lambda _: 0.0))

        def traction_fn(t):
            return jnp.zeros((self.global_local_map.index_array.shape[0], \
                              self.element.num_boundaries, \
                              self.element.n_d)) + \
                sum(fn(t) for fn in fns)
        
        return traction_fn

    ########################################################################
    ##################### Global <-> Local conversion. #####################
    ########################################################################

    def get_global_local_maps(self) -> Tuple[Callable, Callable]:
        return self.global_local_map.get_global_local_maps()

    ########################################################################
    ######################### Lagrangian function. #########################
    ########################################################################

    def get_lagrangian_fn(self):
        l2g, g2l = self.get_global_local_maps()

        def lagrangian(cur_g_position, cur_g_velocity, ref_l_position,
                       fix_l_position, fix_l_velocity, traction):
            def_ctrl, def_vels = g2l(cur_g_position, cur_g_velocity,
                                     fix_l_position, fix_l_velocity)
            
            return jnp.sum(jax.vmap(self.element_lagrangian))(
                def_ctrl, def_vels, ref_l_position,
                self.boundary_conditions.active_traction_boundaries, traction
            )

        return lagrangian

    def __init__(self, element: Element, material,
                 boundary_conditions: SEBoundaryConditions,
                 global_local_map: SEGlobalLocalMap):
        self.element = element

        #TODO(doktay): Construct element_lagrangian

        self.boundary_conditions = boundary_conditions
        self.global_local_map = global_local_map