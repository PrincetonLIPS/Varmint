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

import scipy.optimize

import scipy.sparse

from varmintv2.physics.constitutive import PhysicsModel
from varmintv2.physics.energy import generate_stress_fn, generate_total_energy_fn
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
        -> Tuple[Callable[[ArrayND], ArrayND],
                 Callable[[ArrayND, ArrayND], ArrayND]]:
        """Get the mapping functions between global and local coordinates.
        
        Returns:
        ========

        Two functions, one for global -> local and another for local -> global.

        The global -> local function takes in 4 ndarray arguments:
            - global coordinates
            - values of fixed coordinates
        
        All the above arrays should have the same shape. The function will then
        return the local coordinates that obey the fixed values.

        The local -> global function takes a single ndarray argument:
            - local coordinates

        All the above arrays should have the same shape. The function will then
        return the global coordinates. It will choose an arbitrary value amongst
        incident local control points (the user should ensure the values are the
        same).
        """
        pass

    def unflatten_sequence(self, value, fixed_value):
        _, g2l_map = self.get_global_local_maps()

        local_value = [g2l_map(q, f) for q, f in \
                zip(value, fixed_value)]
        
        return local_value

    def unflatten_dynamics_sequence(self, positions, velocities,
                                    fixed_positions, fixed_velocities):
        """Helper function to convert a sequence of global coordinates to local."""
        return self.unflatten_sequence(positions, fixed_positions), \
               self.unflatten_sequence(velocities, fixed_velocities)
    
    @abstractmethod
    def get_lagrangian_fn(self) -> Callable:
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

    @abstractmethod
    def get_stress_field_fn(self) -> Tuple[ArrayND, Callable]:
        """Returns a function that computes the stress field.
        
        Returns a tuple. The first element is the list of points where the 
        stress field will be computed. The second is a function that takes in
        deformed and reference control points and outputs the Cauchy stress
        tensor at each point.
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
                return 0.0 * vel_fn(t) * self.dirichlet_labels[group][..., jnp.newaxis]

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
    
    def fixed_locs_from_dict(self, ref_l_position, displacements):
        """Used for statics. Given a dictionary of displacements, create
        a fixed locations array."""

        absolute_positions = \
            {key: val * self.dirichlet_labels[key][..., jnp.newaxis] 
                    for key, val in displacements.items()}
        
        all_pos = ref_l_position
        for group in self.dirichlet_labels:
            pos = absolute_positions.get(group, None)

            if pos is not None:
                all_pos = all_pos + pos
        return all_pos

    def tractions_from_dict(self, tractions):
        """Used for statics. Given a dictionary of traction forces, create
        a traction specification array."""

        parsed_tractions = \
            {key: val * self.traction_labels[key][..., jnp.newaxis]
                    for key, val in tractions.items()}

        zero_tractions = jnp.zeros((self.index_array.shape[0], \
                                    self.element.num_boundaries, \
                                    self.element.n_d))

        return zero_tractions + sum(parsed_tractions)

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
        def local_to_global(local_coords):
            kZeros = jnp.zeros((self.n_components, self.element.n_d))

            # Transform from local form to component form.
            global_coords = jax.ops.index_update(kZeros, self.index_array, local_coords)
            global_coords = global_coords.flatten()

            # Get nonfixed components.
            global_coords_without_nonfixed = jnp.take(global_coords, self.nonfixed_labels, axis=0)

            # Now get rigid components.
            rigid_centers = []
            for indices in self.rigidity_nonfixed_labels:
                # Use as a dof the difference of mean value from reference config.
                mean_value = jnp.mean(jnp.take(global_coords, indices, axis=0))
                ref_mean_value = jnp.mean(jnp.take(self.ref_ctrl_component, indices, axis=0))
                rigid_centers.append(jnp.array([mean_value - ref_mean_value]))

            return jnp.concatenate([global_coords_without_nonfixed] + rigid_centers)

        def global_to_local(global_coords, fixed_coords):
            # Component dimensions.
            kZeros = jnp.zeros((self.n_components, self.element.n_d))

            nonrigid_global_coords = global_coords[:self.nonfixed_labels.size]
            rigid_global_coords = global_coords[self.nonfixed_labels.size:]

            # Find fixed locations array in component form.
            fixed_locs = jax.ops.index_update(
                kZeros, self.index_array, fixed_coords)
            fixed_locs = fixed_locs.flatten()
            fixed_locs = jnp.take(fixed_locs, self.fixed_labels, axis=0)

            local_pos = jnp.zeros(self.n_components * self.element.n_d)

            # Fill in the non-fixed locations
            local_pos = jax.ops.index_update(
                local_pos, self.nonfixed_labels, nonrigid_global_coords)

            # Fill in the fixed locations
            fixed_pos = jax.ops.index_update(local_pos, self.fixed_labels,
                                             fixed_locs)

            # Fill in the rigid patch locations
            for i, indices in enumerate(self.rigidity_nonfixed_labels):
                values = jnp.take(self.ref_ctrl_component, indices, axis=0)
                offset_values = values + rigid_global_coords[i]

                fixed_pos = jax.ops.index_update(fixed_pos, indices,
                                                 offset_values)
            
            # Unflattened component form
            fixed_pos = fixed_pos.reshape((-1, self.element.n_d))

            # Convert from component form to local form.
            return jnp.take(fixed_pos, self.index_array, axis=0)

        return local_to_global, global_to_local

    ########################################################################
    ######################### Lagrangian function. #########################
    ########################################################################

    def get_lagrangian_fn(self):
        l2g, g2l = self.get_global_local_maps()

        def lagrangian(cur_g_position, cur_g_velocity, ref_l_position,
                       fix_l_position, fix_l_velocity, traction):
            def_ctrl = g2l(cur_g_position, fix_l_position)
            def_vels = g2l(cur_g_velocity, fix_l_velocity)
            
            K, G, S, T = jax.vmap(self.element_energy_fn)(
                def_ctrl, def_vels, ref_l_position,
                self.active_traction_boundaries, traction
            )

            return jnp.sum(K - G - S - T)

        return lagrangian

    def get_potential_energy_fn(self, ref_l_position):
        l2g, g2l = self.get_global_local_maps()

        def potential_energy(cur_g_position, fix_l_position, traction):
            def_ctrl = g2l(cur_g_position, fix_l_position)
            
            _, G, S, T = jax.vmap(self.element_energy_fn)(
                def_ctrl, jnp.zeros_like(def_ctrl), ref_l_position,
                self.active_traction_boundaries, traction
            )

            return jnp.sum(G + S + T)

        return potential_energy

    def get_strain_energy_fn(self, ref_l_position):
        l2g, g2l = self.get_global_local_maps()

        def strain_energy(cur_g_position, fix_l_position, traction):
            def_ctrl = g2l(cur_g_position, fix_l_position)
            
            _, G, S, T = jax.vmap(self.element_energy_fn)(
                def_ctrl, jnp.zeros_like(def_ctrl), ref_l_position,
                self.active_traction_boundaries, traction
            )

            return jnp.sum(S)

        return strain_energy

    def get_stress_field_fn(self):
        # get quad points for element object.
        # compute map of quad points for each element in the geometry.
        # compute generate_stress_fn for each
        points = self.element.quad_points
        stress_fn = generate_stress_fn(self.element, self.material, points)
        map_fn = self.element.get_map_fn(points)

        vmap_map_fn = jax.vmap(map_fn, in_axes=0)
        vmap_stress_fn = jax.vmap(stress_fn, in_axes=(0, 0))

        return vmap_map_fn, vmap_stress_fn

    @property
    def jac_sparsity_graph(self):
        return self._jac_sparsity_graph

    @property
    def jac_reconstruction_tangents(self) -> Array2D:
        return self._jac_reconstruction_tangents

    def get_jac_reconstruction_fn(self) -> Callable:
        return self._jac_reconstruction_fn
    
    def point_to_patch_and_parent(self, point, l_ref_ctrl):
        """Given a point in the domain and reference configuration,
        find the patch that contains the point, as well as the 
        coordinates in parent space for that point.
        
        Uses a root-finding algorithm initialized by closest quad point.
        """

        all_patches_map_fn = jax.vmap(self.element.get_quad_map_fn())
        all_quad_maps = all_patches_map_fn(l_ref_ctrl)

        # Find closest amongst quad points
        dists = onp.linalg.norm(all_quad_maps - point, axis=-1)
        ind = onp.unravel_index(onp.argmin(dists), all_quad_maps.shape[:-1])
        
        patch_ind = ind[0]
        quad_pt = self.element.quad_points[ind[1:]]

        # Root finding to fix
        map_fn = self.element.get_map_fn_fixed_ctrl(l_ref_ctrl[patch_ind])

        @jax.jit
        def fn_to_optimize(p):
            return jnp.linalg.norm(map_fn(p) - point)
        res = scipy.optimize.minimize(fn_to_optimize, quad_pt)

        return patch_ind, res.x
    
    def patch_and_parent_to_point(self, patch_ind, parent_pt, l_ctrl):
        """Given patch index and parent coordinate, return deformation."""

        map_fn = self.element.get_map_fn(parent_pt[onp.newaxis, :])
        return map_fn(l_ctrl[patch_ind]).squeeze()

    def __init__(self, element: Element, material: PhysicsModel,
                 init_ctrl: ArrayND, constraints: Tuple[Array1D, Array1D],
                 dirichlet_labels: Dict[str, ArrayND],
                 traction_labels: Dict[str, ArrayND],
                 rigid_patches_boolean: Array1D = None):
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

        - rigid_patches_boolean: A 1D boolean array specifying which patches are rigid.

        """
        if not geometry_utils.verify_constraints(init_ctrl, constraints):
            print('WARNING: Constraints are not satisfied by init_ctrl.')
        
        if rigid_patches_boolean is None:
            rigid_patches_boolean = onp.zeros(init_ctrl.shape[0], dtype=onp.bool)

        self.constraints = constraints
        self.dirichlet_fns = {}
        self.traction_fns = {}

        self.element = element
        self.material = material
        self.element_energy_fn = generate_total_energy_fn(element, material)

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

        # Compute reference configuration in component form. Needed for 
        # rigid patches.
        self.ref_ctrl_component = onp.zeros((self.n_components, self.element.n_d))
        self.ref_ctrl_component[self.index_array] = init_ctrl
        self.ref_ctrl_component = self.ref_ctrl_component.flatten()

        # Create a graph between rigid patches that are incident, and find
        # connected components in that graph to determine rigidity groups.
        # Incidence is defined as having control points in common. This is
        # a stronger condition than physically possible. Two patches could be
        # connected via a single control point leading two patches to have
        # separate rotation dofs, but we will not consider this for now.

        # Extract patch numbers from constraints using first element of index.
        patch_nums_rows = onp.unravel_index(all_rows, init_ctrl.shape[:-1])[0]
        patch_nums_cols = onp.unravel_index(all_cols, init_ctrl.shape[:-1])[0]
    
        # Convert to array of patch numbers.
        # TODO(doktay): Would it be better to just pass in array of patch
        # numbers directly to the initializer?
        rigid_patches = onp.nonzero(rigid_patches_boolean)[0]  # Should be sorted
        row_constraints_rigid = onp.isin(patch_nums_rows, rigid_patches)
        col_constraints_rigid = onp.isin(patch_nums_cols, rigid_patches)

        rigidity_group_constraint = row_constraints_rigid & col_constraints_rigid
        rigid_patch_rows = patch_nums_rows[rigidity_group_constraint]
        rigid_patch_cols = patch_nums_cols[rigidity_group_constraint]

        rigid_patch_incidence_graph = csr_matrix(
            (onp.ones_like(rigid_patch_rows), (rigid_patch_rows, rigid_patch_cols)),
            shape=(init_ctrl.shape[0], init_ctrl.shape[0]), dtype=onp.int8)
        n_rigid_groups, all_patches_rigid_group_labels = connected_components(
            csgraph=rigid_patch_incidence_graph,
            directed=False,
            return_labels=True,
        )

        rigid_group_labels = all_patches_rigid_group_labels[rigid_patches_boolean]
        all_rigid_groups = onp.unique(rigid_group_labels)

        rigid_labels_incomplete = {}
        for group in all_rigid_groups:
            group_indices = onp.zeros_like(self.index_array, dtype=onp.int8)
            group_indices[all_patches_rigid_group_labels == group] = 1

            rigid_labels_incomplete[group] = group_indices

        # Complete the rigidity groups according to the sparsity pattern
        self.rigid_labels = {}
        self.all_rigid_indices = onp.zeros(init_ctrl.shape[:-1], dtype=onp.bool)
        for group in all_rigid_groups:
            label = rigid_labels_incomplete[group]
            indices = local_indices[label > 0]
            group_all_dists = dijkstra(spmat, directed=False, indices=indices,
                                       unweighted=True, min_only=True)
            group_all_dists = onp.reshape(group_all_dists, init_ctrl.shape[:-1])
            self.rigid_labels[group] = group_all_dists < onp.inf

            # Aggregate all rigid points
            self.all_rigid_indices = \
                self.all_rigid_indices | self.rigid_labels[group]

        # Complete the dirichlet_labels according to the sparsity graph.
        self.dirichlet_labels = {}
        self.dirichlet_label_fixeddims = {}
        self.all_dirichlet_indices = onp.zeros(init_ctrl.shape[:-1], dtype=onp.bool)
        for group in dirichlet_labels:

            # If dirichlet entry is a tuple, then one is the 
            # index array and the other is the dimensions that are fixed.
            if isinstance(dirichlet_labels[group], tuple):
                label, fixed_dims = dirichlet_labels[group]
            else:
                label = dirichlet_labels[group]
                fixed_dims = onp.array([1, 1])

            indices = local_indices[label > 0]
            group_all_dists = dijkstra(spmat, directed=False, indices=indices,
                                       unweighted=True, min_only=True)
            group_all_dists = onp.reshape(group_all_dists, init_ctrl.shape[:-1])
            self.dirichlet_labels[group] = group_all_dists < onp.inf
            self.dirichlet_label_fixeddims[group] = fixed_dims

            # If any dirichlet CP is in a rigidity group, then all CPs in that
            # group must also be a dirichlet CP.
            if onp.any(self.all_rigid_indices & self.dirichlet_labels[group]):
                for rgroup in self.rigid_labels:
                    if onp.any(self.rigid_labels[rgroup] & self.dirichlet_labels[group]):
                        self.dirichlet_labels[group] = self.dirichlet_labels[group] | self.rigid_labels[rgroup]

            # Aggregate all fixed indices to create fixed_labels needed for
            # global <-> local conversion.
            self.all_dirichlet_indices = \
                self.all_dirichlet_indices | self.dirichlet_labels[group]

        # np.unique will make sure this is sorted
        fixed_labels = \
            onp.unique(self.index_array[self.all_dirichlet_indices])

        # For each of the fixed labels, keep track of which dimensions are fixed.
        fixed_labels_dimension_index = \
            onp.zeros((fixed_labels.shape[0], self.element.n_d))
        for group in self.dirichlet_labels:
            label = self.dirichlet_labels[group]
            fixed_dims = self.dirichlet_label_fixeddims[group]

            group_indices = onp.unique(self.index_array[label])
            inds_into_fixedlabels = onp.searchsorted(fixed_labels, group_indices)
            fixed_labels_dimension_index[inds_into_fixedlabels, :] = fixed_labels_dimension_index[inds_into_fixedlabels, :] + fixed_dims

        # When component form is flattened, component i becomes indices 
        # n_d * i + (0, 1, ..., n_d-1)
        # From the fixed_labels array generated above, modify it to refer to
        # indices in the flattened component form.

        fixed_labels_withdims = self.element.n_d * fixed_labels
        fixed_labels_withdims = onp.stack((fixed_labels_withdims,) * self.element.n_d, axis=-1)
        fixed_labels_withdims = fixed_labels_withdims + onp.arange(self.element.n_d)

        # When adding boundary conditions over a single dimension,
        # instead of flattening here make an index array. 
        # Make sure to add whatever was not picked in the index array to nonfixed_labels.
        self.fixed_labels = fixed_labels_withdims[fixed_labels_dimension_index > 0]

        # Now figure out rigidity.
        # Rigid CPs that have not been labeled as fixed will be translated to 
        # a translation and rotation. If a single dimension of the rigid group
        # is fixed, the other dimension is just a single scalar (no rotation).
        # TODO(doktay): For simplicity, ignore rotations in rigid objects for now.
        # We don't need it for our current applications.

        # We will operate on the component form as above. Figure out which indices
        # in control form belong to which group. Treat different dimensions as
        # different ridigity groups, which we can do because we assume translation
        # only.

        # Need to redeclare this
        fixed_labels = \
            onp.unique(self.index_array[self.all_dirichlet_indices])

        self.rigidity_nonfixed_labels = []
        all_inds_into_fixedlabels = [onp.array([], dtype=onp.int64)]
        for group in self.rigid_labels:
            group_nonfixed_labels = \
                onp.unique(self.index_array[(~self.all_dirichlet_indices) & self.rigid_labels[group]])
            group_nonfixed_labels = self.element.n_d * group_nonfixed_labels
            group_nonfixed_labels = onp.stack((group_nonfixed_labels,) * self.element.n_d, axis=-1)
            group_nonfixed_labels = group_nonfixed_labels + onp.arange(self.element.n_d)

            # Treat different dimensions as separate rigidity groups.
            if group_nonfixed_labels.size > 0:
                for d in range(self.element.n_d):
                    self.rigidity_nonfixed_labels.append(group_nonfixed_labels[..., d])
            
            # Search to see if this rigidity group is in fixed labels
            group_fixed_labels = \
                onp.unique(self.index_array[self.all_dirichlet_indices & self.rigid_labels[group]])
            if group_fixed_labels.size > 0:
                # Guaranteed that fixedlabels contains entire rigidity group.
                inds_into_fixedlabels = onp.searchsorted(fixed_labels, group_fixed_labels)
                all_inds_into_fixedlabels.append(inds_into_fixedlabels)
                group_fixed_labels = self.element.n_d * group_fixed_labels
                group_fixed_labels = onp.stack((group_fixed_labels,) * self.element.n_d, axis=-1)
                group_fixed_labels = group_fixed_labels + onp.arange(self.element.n_d)
                
                rigid_but_partially_fixed = group_fixed_labels[fixed_labels_dimension_index[inds_into_fixedlabels, :] == 0]
                if rigid_but_partially_fixed.size > 0:
                    self.rigidity_nonfixed_labels.append(rigid_but_partially_fixed)

        all_inds_into_fixedlabels = onp.concatenate(all_inds_into_fixedlabels)
        all_rigid_fixed_labels = onp.zeros(fixed_labels_withdims.shape[0], dtype=onp.bool)
        all_rigid_fixed_labels[all_inds_into_fixedlabels] = True

        # np.unique will make sure this is sorted
        nonfixed_labels = \
            onp.unique(self.index_array[~self.all_dirichlet_indices & ~self.all_rigid_indices])

        nonfixed_labels = self.element.n_d * nonfixed_labels
        nonfixed_labels = onp.stack((nonfixed_labels,) * self.element.n_d, axis=-1)
        nonfixed_labels = nonfixed_labels + onp.arange(self.element.n_d)
        nonfixed_labels = nonfixed_labels.flatten()

        fixed_labels_without_rigid = fixed_labels_withdims[~all_rigid_fixed_labels]
        dimension_index_without_rigid = fixed_labels_dimension_index[~all_rigid_fixed_labels]
        self.nonfixed_labels = \
            onp.concatenate((nonfixed_labels, fixed_labels_without_rigid[dimension_index_without_rigid == 0]))

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

        # Use kron to duplicate per dimension, and then use nonfixed_labels
        # to get the nonfixed indices.
        jac_sparsity_graph = scipy.sparse.kron(
            jac_sparsity_graph, onp.ones((element.n_d, element.n_d)), format='csc'
        )
        jac_sparsity_graph = jac_sparsity_graph[:, self.nonfixed_labels]
        self._jac_sparsity_graph = jac_sparsity_graph[self.nonfixed_labels, :]

        self._jac_reconstruction_tangents, self._jac_reconstruction_fn = \
            sparsity.pattern_to_reconstruction(self._jac_sparsity_graph)
