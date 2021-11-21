from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from re import I
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import scipy.sparse

from varmintv2.physics.constitutive import PhysicsModel
from varmintv2.physics.energy import generate_element_lagrangian
from varmintv2.utils import geometry_utils
from varmintv2.utils import sparsity

from varmintv2.utils.typing import Array1D, Array2D, ArrayND
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
    def get_traction_fn(self) -> Callable:
        """Get a function that returns traction values through time.
        
        Returns:
        ========

        Returns a function that takes time (t) as an argument, and returns
        the traction forces on each boundary. The return value can be used
        as an input to the Lagrangian function.
        """
        pass

    @abstractmethod
    def get_fixed_locs_fn(self, ref_l_position) -> Tuple[Callable, Callable]:
        """Get a function that returns Dirichlet conditions through time.
        
        Returns:
        ========

        Returns a function that takes time (t) as an argument, and returns
        the values of the Dirichlet constrained positions and velocities.
        The return values can be used as an input to the Lagrangian function.
        """
        pass

    @property
    @abstractmethod
    def jac_sparsity_graph(self):
        pass

    @property
    @abstractmethod
    def jac_reconstruction_tangents(self) -> Array2D:
        """Returns tangent vectors necessary to reconstruct the Jacobian."""
        pass

    @abstractmethod
    def get_jac_reconstruction_fn(self) -> Callable:
        """Returns a function that reconstructs the Jacobian as a csc_matrix.

        Given the result of JVPs with the tangent vectors given by
        self.jac_reconstruction_tangents, returns the Jacobian as a csc_matrix.
        """
        pass

class SingleElementGeometry(Geometry):
    """Geometry composed of a single type of Element."""

    element: Element
    n_elements: int

    element_lagrangian: Callable

    ###########################################################
    ##### Utilities to keep track of boundary conditions. #####
    ###########################################################

    # Maps a Dirichlet boundary condition label to a boolean array in local
    # coordinates that selects out control points in the group. Each item is
    # of size (n_elements, element.ctrl_shape[:-1]).
    dirichlet_labels: Dict[str, ArrayND]

    # Maps a traction boundary condition label to a boolean array 
    # that selects out boundaries in the group. Each item is
    # of size (n_elements, element.num_boundaries).
    traction_labels: Dict[str, ArrayND]

    # Maps a Dirichlet boundary condition label to functions that map time to
    # the displacement and velocity of control points (given time) within the group.
    dirichlet_fns: Dict[str, Tuple[Callable, Callable]]

    # Maps a traction boundary condition label to a function that gives the
    # traction force on a control point (given time) within the group.
    traction_fns: Dict[str, Callable]

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
        return inner

    def register_traction_bc(self, group):
        """Decorator function to declare a traction condition."""

        def inner(fn):
            def decorated(t):
                return fn(t) * self.traction_labels[group][..., jnp.newaxis]

            self.traction_fns[group] = decorated
            return decorated
        return inner

    def get_fixed_locs_fn(self, ref_l_position):
        pos_fns = []
        vel_fns = []

        for group in self.dirichlet_labels:
            pos, vel = self.dirichlet_fns.get(group, (None, None))

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
        for group in self.traction_labels:
            fns.append(self.traction_fns.get(group, lambda _: 0.0))

        def traction_fn(t):
            return jnp.zeros((self.index_array.shape[0], \
                              self.element.num_boundaries, \
                              self.element.n_d)) + \
                sum(fn(t) for fn in fns)
        
        return traction_fn

    ########################################################################
    ##################### Global <-> Local conversion. #####################
    ########################################################################

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

    ########################################################################
    ######################### Lagrangian function. #########################
    ########################################################################

    def get_lagrangian_fn(self):
        l2g, g2l = self.get_global_local_maps()

        def lagrangian(cur_g_position, cur_g_velocity, ref_l_position,
                       fix_l_position, fix_l_velocity, traction):
            def_ctrl, def_vels = g2l(cur_g_position, cur_g_velocity,
                                     fix_l_position, fix_l_velocity)
            
            return jnp.sum(jax.vmap(self.element_lagrangian)(
                def_ctrl, def_vels, ref_l_position,
                self.active_traction_boundaries, traction
            ))

        return lagrangian

    @property
    def jac_sparsity_graph(self):
        return self._jac_sparsity_graph

    @property
    def jac_reconstruction_tangents(self) -> Array2D:
        return self._jac_reconstruction_tangents

    def get_jac_reconstruction_fn(self) -> Callable:
        return self._jac_reconstruction_fn

    def __init__(self, element: Element, material: PhysicsModel,
                 init_ctrl: ArrayND, constraints: Tuple[Array1D, Array1D],
                 dirichlet_labels: Dict[str, ArrayND],
                 traction_labels: Dict[str, ArrayND]):
        """Initialize a SingleElementGeometry.

        The node numbers of the local coordinate system will be in flatten
        order. As in,
            local_indices = onp.arange(n_cp).reshape(init_ctrl.shape[:-1])
        where n_cp = init_ctrl.size // init_ctrl.shape[-1]

        Parameters:
        ===========

        - element: Instance of Element that defines local geometry. Contains
                   methods for integration as well as intra-element sparsity.
        
        - material: PhysicsModel capturing the material physical properties.

        - init_ctrl: Control points in the reference configuration. Has shape
                     (n_elements, element.ctrl_shape).

        - constraints: Constraints between the nodes. Will be used to construct
                       an adjacency matrix. Consists of two 1-D ndarrays of 
                       the same size, (row, col) denoting a row.size constraints
                       between index row[i] and col[i].

        - dirichlet_labels: Maps a Dirichlet boundary condition label to a
                            boolean array in local coordinates that selects
                            out control points in the group. Each item is of
                            size (n_elements, element.ctrl_shape[:-1]).
                            Does not necessarily have to obey adjacency
                            constraints. Will be modified to obey constraints
                            during initialization.

        - traction_labels: Maps a traction boundary condition label to a boolean
                           array that selects out boundaries in the group. Each
                           item is of size (n_elements, element.num_boundaries).

        """
        if not geometry_utils.verify_constraints(init_ctrl, constraints):
            print('WARNING: Constraints are not satisfied by init_ctrl.')
        
        self.dirichlet_fns = {}
        self.traction_fns = {}

        self.element = element
        self.element_lagrangian = generate_element_lagrangian(element, material)

        # Construct adjacency matrix of the nodes.
        n_cp = init_ctrl.size // init_ctrl.shape[-1]
        local_indices = onp.arange(n_cp).reshape(init_ctrl.shape[:-1])
        self.n_elements = init_ctrl.shape[0]

        all_rows, all_cols = constraints
        spmat = csr_matrix((onp.ones_like(all_rows), (all_rows, all_cols)),
                           shape=(n_cp, n_cp), dtype=onp.int8)

        # The connected components in the adjacency matrix will correspond
        # to the clusters of control points that are incident.
        n_components, labels = connected_components(
            csgraph=spmat,
            directed=False,
            return_labels=True
        )
        self.n_components = n_components
        self.index_array = onp.reshape(labels, init_ctrl.shape[:-1])

        self.traction_labels = traction_labels

        # Complete the dirichlet_labels according to the sparsity graph.
        self.dirichlet_labels = {}
        self.all_dirichlet_indices = onp.zeros(init_ctrl.shape[:-1])
        for group in dirichlet_labels:
            indices = local_indices[dirichlet_labels[group] > 0]
            group_all_dists = dijkstra(spmat, directed=False, indices=indices,
                                       unweighted=True, min_only=True)
            group_all_dists = onp.reshape(group_all_dists, init_ctrl.shape[:-1])
            self.dirichlet_labels[group] = group_all_dists < onp.inf

            # Aggregate all fixed indices to create fixed_labels needed for
            # global <-> local conversion.
            self.all_dirichlet_indices = \
                self.all_dirichlet_indices + self.dirichlet_labels[group]

        self.fixed_labels = onp.unique(self.index_array[self.all_dirichlet_indices > 0])
        self.nonfixed_labels = onp.unique(self.index_array[self.all_dirichlet_indices == 0])

        ###################################################################
        # Using the local sparsity pattern for the Element, construct the #
        # sparsity pattern of the full geometry.                          #
        ###################################################################
        local_sparsity = element.get_sparsity_pattern()
        n_ctrl_per_element = init_ctrl[0].size // init_ctrl.shape[-1]

        # Offset the indices of each element to get global offset.
        all_local_sparsities = \
            [local_sparsity.copy() + n_ctrl_per_element * i
                for i in range(self.n_elements)]
        all_local_sparsities = onp.stack(all_local_sparsities, axis=0)
        jac_spy_edges = \
            self.index_array.flatten()[all_local_sparsities].reshape((-1, 2))

        jac_entries = onp.ones_like(jac_spy_edges[:, 0])
        jac_rows = jac_spy_edges[:, 0]
        jac_cols = jac_spy_edges[:, 1]
        jac_sparsity_graph = \
            csc_matrix((jac_entries, (jac_rows, jac_cols)),
                       (n_components, n_components), dtype=onp.int8)

        # Remove the fixed labels from the Jacobian, and use kron to duplicate
        # for each dimension.
        jac_sparsity_graph = jac_sparsity_graph[:, self.nonfixed_labels]
        jac_sparsity_graph = jac_sparsity_graph[self.nonfixed_labels, :]
        self._jac_sparsity_graph = scipy.sparse.kron(
            jac_sparsity_graph, onp.ones((element.n_d, element.n_d)), format='csc'
        )

        self._jac_reconstruction_tangents, self._jac_reconstruction_fn = \
            sparsity.pattern_to_reconstruction(self._jac_sparsity_graph)