from typing import Any, Callable
import jax
import jax.numpy as np
import jax.numpy.linalg as npla
import numpy as onp
import logging

from varmint.patch2d import Patch2D

from .vmap_utils import *


def generate_patch_lagrangian(patch: Patch2D):
    """ Generate a Lagrangian function for a Patch2D object. """

    jacobian_u_fn = patch.get_jacobian_u_fn()
    line_jacobian_u_fn = patch.get_line_derivs_u_fn()
    energy_fn = patch.get_energy_fn()
    quad_fn = patch.get_quad_fn()
    line_quad_fn = patch.get_line_quad_fn()
    deformation_fn = patch.get_deformation_fn()
    line_deformation_fn = patch.get_line_deformation_fn()
    jacobian_ctrl_fn = patch.get_jacobian_ctrl_fn()
    vmap_energy_fn = jax.vmap(energy_fn, in_axes=(0,))
    jac_dets_fn = jax.vmap(npla.det, in_axes=(0,))

    defgrads_fn = jax.vmap(
        lambda A, B: npla.solve(B.T, A.T).T,
        in_axes=(0, 0),
    )

    kinetic_energy_fn = \
        lambda mm, vv: 0.5 * np.tensordot(
            np.tensordot(mm, vv, ((3, 4, 5), (0, 1, 2))
                         ), vv, ((0, 1, 2), (0, 1, 2))
        )

    mat_density = patch.material.density
    gravity = 981.0  # cm/s^2

    def lagrangian(def_ctrl, def_vels, ref_ctrl, orientation, traction):
        """Compute the Lagrangian of this patch.

        Parameters:
        -----------
        def_ctrl: array_like
          The control points corresponding to the positions of the body in the
          deformed configuration. Shape: (n_cp, n_cp, 2)
        def_vels: array_like
          The control points corresponding to the momenta of the body in the
          deformed configuration. Shape: (n_cp, n_cp, 2)
        ref_ctrl: array_like
          The control points corresponding to the positions of the body in the
          reference configuration. Shape: (n_cp, n_cp, 2)
        orientation: array_like
          Bit array specifying whether the corresponding edge has a Neumann boundary
          condition applied to it. Shape is (4,), representing edges
          (left, top, right, bottom) in order.
        traction: array_like
          Indicating the direction and magnitude of traction force for each of the
          orientations. Shape: (4, 2)

        Returns
        -------
        Scalar Lagrangian over the patch integrated using numerical quadrature over the patch.

        """
        # Jacobian of reference config wrt parent config.
        def_jacs = jacobian_u_fn(def_ctrl)
        ref_jacs = jacobian_u_fn(ref_ctrl)

        # Deformation gradients. def_jacs @ ref_jacs_inv computed via a linear solve.
        defgrads = defgrads_fn(def_jacs, ref_jacs)

        # Jacobian determinants of reference config wrt parent.
        ref_jac_dets = jac_dets_fn(ref_jacs)

        # Strain energy density wrt parent config.
        # Units are GPa = 10^9 J / m^3 in the reference configuration.
        # Convert to J / cm^3 by multiplying by 10^3.
        strain_energy_density = vmap_energy_fn(defgrads) * np.abs(ref_jac_dets)

        # Total potential energy via integrating over parent config.
        strain_potential = 1e3 * np.sum(quad_fn(strain_energy_density))

        # Mass density in parent config.
        # I'm going to assume each patch is uniform in the reference
        # configuration. Each patch might have a different density because it's
        # a diff material. Densities are g / cm^3.
        mass_density = mat_density * np.abs(ref_jac_dets)

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
        # Orientation: 0 - left
        #              1 - top
        #              2 - right
        #              3 - bottom

        # Compute traction
        def traction_in_dir(orientation):
            line_ref_jacs = line_jacobian_u_fn(ref_ctrl, orientation)
            line_positions = line_deformation_fn(def_ctrl, orientation)
            traction_density = np.linalg.norm(line_ref_jacs, axis=-1) * \
                np.sum(traction[orientation] * line_positions, axis=-1)
            return np.sum(line_quad_fn(traction_density, orientation))

        total_traction_potential = 0.0

        # Should be unrolled by the compiler, and shouldn't be too bad since it's only 4 loops.
        for i in range(4):
            total_traction_potential += jax.lax.cond(
                orientation[i],
                lambda _: traction_in_dir(i),
                lambda _: 0.0,
                operand=None,
            )

        # Total work done by gravity integrated over parent config.
        gravity_potential = 1e-7 * np.sum(quad_fn(grav_energy_density))

        ctrl_jacs = jacobian_ctrl_fn(def_ctrl)
        ctrl_jacTjac = jax.vmap(np.tensordot, in_axes=(
            0, 0, None))(ctrl_jacs, ctrl_jacs, (0, 0,))

        mass_matrices = (ctrl_jacTjac.T * mass_density.T).T
        mass_matrix = quad_fn(mass_matrices)

        # Compute the inertia with this matrix.
        # This should be g * cm/s * cm/s = g * cm^2 / s^2 = erg.
        # So also divide by 10^7 to get Joules.
        kinetic_energy = 1e-7 * \
            np.sum(kinetic_energy_fn(mass_matrix, def_vels))

        # Compute and return lagrangian.
        return kinetic_energy - gravity_potential - \
            strain_potential + total_traction_potential

    return lagrangian
