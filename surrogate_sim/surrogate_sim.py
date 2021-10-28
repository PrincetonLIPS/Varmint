import time
import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize as spopt
import string

from varmint.patch2d import Patch2D
from varmint.shape2d import Shape2D
from varmint.materials import Material, SiliconeRubber
from varmint.constitutive import NeoHookean2D
from varmint.bsplines import default_knots
from varmint.lagrangian import generate_patch_lagrangian
from varmint.discretize import get_hamiltonian_stepper
from varmint.levmar import get_lmfunc
from varmint.cellular2d import index_array_from_ctrl, generate_quad_lattice

import experiment_utils as exputils

import json
import logging
import random
import argparse
import time
import os


def sim_with_surrogate(surrogate):
    npr.seed(0)

    mat = NeoHookean2D(WigglyMat)

    friction = 1e-7

    # Create patch parameters.c
    quad_deg = 10
    spline_deg = 3
    num_ctrl = 5
    num_x = 3
    num_y = 1

    xknots = default_knots(spline_deg, num_ctrl)
    yknots = default_knots(spline_deg, num_ctrl)
    widths = 5*np.ones(num_x)
    heights = 5*np.ones(num_y)

    init_radii = npr.rand(num_x, num_y, (num_ctrl-1)*4)*0.9 + 0.05
    init_radii = np.array(init_radii)
    #init_radii = np.ones((num_x,num_y,(num_ctrl-1)*4))*0.5
    init_ctrl = generate_quad_lattice(widths, heights, init_radii)
    n_components, index_arr = index_array_from_ctrl(num_x, num_y, init_ctrl)
    left_side = onp.array(init_ctrl[:, :, :, 0] == 0.0)
    fixed_labels = index_arr[left_side]

    def flatten_add(unflat_ctrl, unflat_vel):
        almost_flat = jax.ops.index_add(
            np.zeros((n_components, 2)), index_arr, unflat_ctrl)
        almost_flat_vel = jax.ops.index_add(
            np.zeros((n_components, 2)), index_arr, unflat_vel)
        return almost_flat.flatten(), almost_flat_vel.flatten()

    def flatten(unflat_ctrl, unflat_vel):
        almost_flat = jax.ops.index_update(
            np.zeros((n_components, 2)), index_arr, unflat_ctrl)
        almost_flat_vel = jax.ops.index_update(
            np.zeros((n_components, 2)), index_arr, unflat_vel)
        return almost_flat.flatten(), almost_flat_vel.flatten()

    fixed_locations = flatten(init_ctrl, np.zeros_like(init_ctrl))[
        0].reshape((n_components, 2))
    fixed_locations = np.take(fixed_locations, fixed_labels, axis=0)

    def unflatten(flat_ctrl, flat_vels, fixed_locs):
        fixed_locs = flatten(fixed_locs, np.zeros_like(fixed_locs))[
            0].reshape((n_components, 2))
        fixed_locs = np.take(fixed_locs, fixed_labels, axis=0)

        flat_ctrl = flat_ctrl.reshape(n_components, 2)
        flat_vels = flat_vels.reshape(n_components, 2)
        fixed = jax.ops.index_update(flat_ctrl, fixed_labels, fixed_locs)
        fixed_vels = jax.ops.index_update(
            flat_vels, fixed_labels, np.zeros_like(fixed_locs))
        return np.take(fixed, index_arr, axis=0), np.take(fixed_vels, index_arr, axis=0)

    def unflatten_nofixed(flat_ctrl, flat_vels):
        flat_ctrl = flat_ctrl.reshape(n_components, 2)
        flat_vels = flat_vels.reshape(n_components, 2)
        return np.take(flat_ctrl, index_arr, axis=0), np.take(flat_vels, index_arr, axis=0)

    # Create the shape.
    shape = Shape2D(*[
        Patch2D(
            xknots,
            yknots,
            spline_deg,
            mat,
            quad_deg,
            None,  # labels[ii,:,:],
            fixed_labels,  # <-- Labels not locations
        )
        for ii in range(len(init_ctrl))
    ])

    patch = Patch2D(
        xknots,
        yknots,
        spline_deg,
        mat,
        quad_deg,
        None,  # labels[ii,:,:],
        fixed_labels,  # <-- Labels not locations
    )

    def friction_force(q, qdot, ref_ctrl, fixed_dict): return -friction * qdot

    def displacement(t):
        return np.sin(4 * np.pi * t) * np.ones_like(init_ctrl)

    p_lagrangian = generate_patch_lagrangian(patch)

    def full_lagrangian(q, qdot, ref_ctrl, displacement):
        def_ctrl, def_vels = unflatten(q, qdot, displacement)
        return np.sum(jax.vmap(p_lagrangian)(def_ctrl, def_vels, ref_ctrl))

    stepper, residual_fun = \
        get_hamiltonian_stepper(full_lagrangian, friction_force,
                                return_residual=True, surrogate=surrogate)

    dt = np.float32(0.005)
    T = 0.5

    def simulate(ref_ctrl):

        # Initially in the ref config with zero momentum.
        q, p = flatten(ref_ctrl, np.zeros_like(ref_ctrl))

        QQ = [q]
        PP = [p]
        TT = [0.0]

        while TT[-1] < T:

            t0 = time.time()
            #fixed_locs = displacement(TT[-1]) + ref_ctrl
            fixed_locs = ref_ctrl

            success = False
            this_dt = dt
            while True:
                new_q, new_p = stepper(
                    QQ[-1], PP[-1], this_dt, ref_ctrl, fixed_locs)
                print('4 surrogate is ', np.all(np.isfinite(new_q)))
                success = np.all(np.isfinite(new_q))
                if success:
                    break
                else:
                    this_dt = this_dt / 2.0
                    print('\tFailed to converge. dt now %f' % (this_dt))

            QQ.append(new_q)
            PP.append(new_p)
            TT.append(TT[-1] + this_dt)
            t1 = time.time()
            print(TT[-1], t1-t0)

        return QQ, PP, TT

    def radii_to_ctrl(radii):
        return generate_quad_lattice(widths, heights, radii)

    def sim_radii(radii):

        # Construct reference shape.
        ref_ctrl = radii_to_ctrl(radii)

        # Simulate the reference shape.
        QQ, PP, TT = simulate(ref_ctrl)

        # Turn this into a sequence of control point sets.
        ctrl_seq = [
            unflatten(
                qt[0],
                np.zeros_like(qt[0]),
                ref_ctrl,  # + displacement(qt[1]),
            )[0] \
            for qt in zip(QQ, TT)
        ]

        return ctrl_seq

    def loss(radii):
        ctrl_seq = sim_radii(radii)

        return -np.mean(ctrl_seq[-1]), ctrl_seq

    val, ctrl_seq = loss(init_radii)

    print('Saving result in video.')
    shape.create_movie(ctrl_seq, 'surrogate.mp4', labels=False)
