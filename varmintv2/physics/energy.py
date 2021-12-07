from typing import Any, Callable
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import numpy as onp
import logging

from varmintv2.geometry.elements import Element
from varmintv2.physics.constitutive import PhysicsModel


def generate_element_lagrangian(element: Element, material: PhysicsModel):
    """ Generate a Lagrangian function over an Element. """

    jacobian_u_fn = element.get_map_jac_fn()

    line_jacobian_u_fns = []
    for i in range(element.num_boundaries):
        line_jacobian_u_fns.append(element.get_map_boundary_jac_fn(i))

    energy_fn = material.get_energy_fn()
    quad_fn = element.get_quad_fn()

    line_quad_fns = []
    for i in range(element.num_boundaries):
        line_quad_fns.append(element.get_boundary_quad_fn(i))

    deformation_fn = element.get_map_fn()

    line_deformation_fns = []
    for i in range(element.num_boundaries):
        line_deformation_fns.append(element.get_map_boundary_fn(i))

    jacobian_ctrl_fn = element.get_ctrl_jacobian_fn()
    vmap_energy_fn = jax.vmap(energy_fn, in_axes=(0,))
    jac_dets_fn = jax.vmap(jnpla.det, in_axes=(0,))

    defgrads_fn = jax.vmap(
        lambda A, B: jnpla.solve(B.T, A.T).T,
        in_axes=(0, 0),
    )

    kinetic_energy_fn = \
        lambda mm, vv: 0.5 * jnp.tensordot(
            jnp.tensordot(mm, vv, ((3, 4, 5), (0, 1, 2))
                         ), vv, ((0, 1, 2), (0, 1, 2))
        )

    mat_density = material.density
    gravity = 0.0 # 981.0  # cm/s^2
    #gravity = 981.0  # cm/s^2

    def lagrangian(def_ctrl, def_vels, ref_ctrl, active_boundaries, traction):
        """Compute the Lagrangian of this element.

        In the following, n_b is element.num_boundaries, and n_d is element.n_d.

        Parameters:
        -----------
        def_ctrl: array_like
          The control points corresponding to the positions of the body in the
          deformed configuration. Shape: element.ctrl_shape
        def_vels: array_like
          The control points corresponding to the momenta of the body in the
          deformed configuration. Shape: element.ctrl_shape
        ref_ctrl: array_like
          The control points corresponding to the positions of the body in the
          reference configuration. Shape: element.ctrl_shape
        active_boundaries: array_like
          Bit array specifying whether the corresponding edge has a Neumann boundary
          condition applied to it. Shape is (n_b,), representing boundary indices.
        traction: array_like
          Vector of traction force for each of the boundaries. Shape: (n_b, n_d)

        Returns
        -------
        Scalar Lagrangian over the element integrated using numerical quadrature.

        """
        # Jacobian of reference config wrt parent config.
        def_jacs = jacobian_u_fn(def_ctrl)
        ref_jacs = jacobian_u_fn(ref_ctrl)

        # Deformation gradients. def_jacs @ ref_jacs_inv computed via a linear solve.
        # Should be unitless.
        defgrads = defgrads_fn(def_jacs, ref_jacs)

        # Jacobian determinants of reference config wrt parent.
        ref_jac_dets = jac_dets_fn(ref_jacs)

        # Strain energy density wrt parent config.
        # Units are GPa = 10^9 J / m^3 in the reference configuration.
        # Convert to J / cm^3 by multiplying by 10^3.
        strain_energy_density = vmap_energy_fn(defgrads) * jnp.abs(ref_jac_dets)

        # Total potential energy via integrating over parent config.
        strain_potential = 1e3 * jnp.sum(quad_fn(strain_energy_density))

        # Mass density in parent config.
        # I'm going to assume each patch is uniform in the reference
        # configuration. Each patch might have a different density because it's
        # a diff material. Densities are g / cm^3.
        mass_density = mat_density * jnp.abs(ref_jac_dets)

        # Positions in deformed config.
        positions = deformation_fn(def_ctrl)

        # Work density done by gravity.
        # Compute a gravitational energy density, in J / cm^3.
        # position = cm
        # gravity = cm / s^2
        # mass_density = g / cm^3
        # result = cm * (cm / s^2) * g / cm^3 = cm * (g * cm / s^2) / cm^3
        # This is ergs per cubic centimeter, i.e., 10^-7 J / cm ^3, so we need to
        # divide by 10^7 to put it into the same units as strain potential.
        # Should we do this before or after the quadrature?
        grav_energy_density = positions[:, 1] * gravity * mass_density

        # Work done by traction on one side of the patch.
        # Compute traction
        def traction_in_dir(bd_idx):
            line_ref_jacs = line_jacobian_u_fns[bd_idx](ref_ctrl)
            line_positions = line_deformation_fns[bd_idx](def_ctrl)
            traction_density = jnp.linalg.norm(line_ref_jacs, axis=-1) * \
                jnp.sum(traction[bd_idx] * line_positions, axis=-1)
            return jnp.sum(line_quad_fns[bd_idx](traction_density))

        total_traction_potential = 0.0

        # Should be unrolled by the compiler, and shouldn't be too bad since it's only 4 loops.
        for i in range(element.num_boundaries):
            total_traction_potential += jax.lax.cond(
                active_boundaries[i],
                lambda _: traction_in_dir(i),
                lambda _: 0.0,
                operand=None,
            )

        # Total work done by gravity integrated over parent config.
        gravity_potential = 1e-7 * jnp.sum(quad_fn(grav_energy_density))

        ctrl_jacs = jacobian_ctrl_fn(def_ctrl)
        ctrl_jacTjac = jax.vmap(jnp.tensordot, in_axes=(
            0, 0, None))(ctrl_jacs, ctrl_jacs, (0, 0,))

        mass_matrices = (ctrl_jacTjac.T * mass_density.T).T
        mass_matrix = quad_fn(mass_matrices)

        # Compute the inertia with this matrix.
        # This should be g * cm/s * cm/s = g * cm^2 / s^2 = erg.
        # So also divide by 10^7 to get Joules.
        kinetic_energy = 1e-7 * \
            jnp.sum(kinetic_energy_fn(mass_matrix, def_vels))

        # Compute and return lagrangian.
        return kinetic_energy - gravity_potential - \
            strain_potential + total_traction_potential

    return lagrangian
