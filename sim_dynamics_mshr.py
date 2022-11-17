from comet_ml import Experiment
import time
import os
import argparse


from varmintv2.geometry.cell2d import construct_cell2D, generate_bertoldi_radii
from varmintv2.geometry.elements import IsoparametricQuad2D, Patch2D
from varmintv2.geometry.geometry import Geometry, SingleElementGeometry
from varmintv2.physics.constitutive import NeoHookean2D
from varmintv2.physics.materials import Material
from varmintv2.solver.discretize import HamiltonianStepper
from varmintv2.utils.movie_utils import create_movie, create_static_image

from varmintv2.geometry.metamaterial_fem import make_metamaterial_mesh

import varmintv2.utils.analysis_utils as autils
import varmintv2.utils.experiment_utils as eutils

import scipy.optimize
from scipy.spatial import KDTree

import numpy.random as npr
import numpy as onp
import jax.numpy as np
import jax

from mshr import *
import dolfin

# Let's do 64-bit. Does not seem to degrade performance much.
from jax.config import config
config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
eutils.prepare_experiment_args(
    parser, exp_root='/n/fs/mm-iga/Varmint/experiments')


# Geometry parameters.
parser.add_argument('-c', '--ncp', type=int, default=5)
parser.add_argument('-q', '--quaddeg', type=int, default=10)
parser.add_argument('-s', '--splinedeg', type=int, default=3)

parser.add_argument('--simtime', type=float, default=50.0)
parser.add_argument('--dt', type=float, default=0.5)

parser.add_argument('--mat_model', choices=['NeoHookean2D', 'LinearElastic2D'],
                    default='NeoHookean2D')
parser.add_argument('--E', type=float, default=0.005)
parser.add_argument('--comet', dest='comet', action='store_true')

parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--strategy', choices=['ilu_preconditioning', 'superlu', 'lu'],
                    default='ilu_preconditioning')


class WigglyMat(Material):
    _E = 0.03
    _nu = 0.48
    _density = 1.0


def main():
    args = parser.parse_args()
    eutils.prepare_experiment_directories(args)
    # args.seed and args.exp_dir should be set.

    eutils.save_args(args)
    npr.seed(args.seed)

    if args.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="gTBUDHLLNaIqxMWyjHKQgtlkW",
            project_name="general",
            workspace="denizokt",
        )
    else:
        experiment = None

    WigglyMat._E = args.E
    mat = NeoHookean2D(WigglyMat)

    element = IsoparametricQuad2D(quad_deg=args.quaddeg)

    #domain =   Rectangle(dolfin.Point(0., 0.), dolfin.Point(5., 5.)) \
    #        - Rectangle(dolfin.Point(2., 1.25), dolfin.Point(3., 1.75)) \
    #        - Circle(dolfin.Point(1, 4), .25) \
    #        - Circle(dolfin.Point(4, 4), .25)
    #mesh2d = generate_mesh(domain, 45)

    domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(5., 5.)) \
             - Circle(dolfin.Point(2.5, 2.5), 0.5)
    mesh2d = generate_mesh(domain, 30)
    #n_cells = 9
    #mesh2d = make_metamaterial_mesh(1.0, 0.0, 0.0, 80, 0.15, 20, n_cells, 0.5)

    points = mesh2d.coordinates()
    cells = mesh2d.cells()

    # mshr creates triangles. Convert to degenerate quadrilaterials instead.
    cells = onp.concatenate((cells, cells[:, 2:3]), axis=-1)
    ctrls = points[cells]  # too easy

    flat_ctrls = ctrls.reshape((-1, 2))
    print('Finding constraints.')
    kdtree = KDTree(flat_ctrls)
    constraints = kdtree.query_pairs(1e-14)
    constraints = np.array(list(constraints))
    print('\tDone.')

    group_1 = np.abs(ctrls[..., 1] - 0.0) < 1e-14
    group_2 = np.abs(ctrls[..., 1] - 5.0) < 1e-10

    dirichlet_groups = {
        '1': group_1,
        '2': group_2
    }

    traction_groups = {
        # empty
    }

    cell = SingleElementGeometry(
        element=element,
        material=mat,
        init_ctrl=ctrls,
        constraints=(constraints[:, 0], constraints[:, 1]),
        dirichlet_labels=dirichlet_groups,
        traction_labels=traction_groups,
    )

    reference_ctrl = ctrls
    reference_velo = onp.zeros_like(reference_ctrl)

    dt = 0.5
    T = 50

    @cell.register_dirichlet_bc('1')
    def group_1_movement(t):
        return t / T * np.array([0.0, 0.0])

    @cell.register_dirichlet_bc('2')
    def group_2_movement(t):
        return t / T * np.array([0.0, -2.0])

    def friction_force(q, qdot, ref_ctrl, fixed_pos, fixed_vel, tractions):
        return -1e-7 * qdot

    local_to_global, global_to_local = cell.get_global_local_maps()
    full_lagrangian = cell.get_lagrangian_fn()

    # Initially in the ref config with zero momentum.
    q = local_to_global(reference_ctrl)
    p = local_to_global(reference_velo)
    print(f'Simulation has {q.shape[0]} degrees of freedom.')

    # Construct solver
    stepper = HamiltonianStepper(
        full_lagrangian, geometry=cell, F=friction_force, save=True)
    step, optimizer = stepper.construct_stepper(q.shape[0], strategy='ilu_preconditioning')

    # Get functions that determine boundary conditions throughout time.
    fixed_locs_fn, fixed_vels_fn = cell.get_fixed_locs_fn(reference_ctrl)
    traction_fn = cell.get_traction_fn()

    # Do time stepping. At each timestep, call one iteration of the solver and
    # update current control points and velocities.

    QQ = [q]
    PP = [p]
    TT = [0.0]
    all_fixed = [fixed_locs_fn(TT[-1])]
    all_fixed_vels = [fixed_vels_fn(TT[-1])]

    while TT[-1] < T:
        t0 = time.time()
        fixed_locs = all_fixed[-1]
        fixed_vels = all_fixed_vels[-1]

        success = False
        this_dt = dt
        while True:
            new_q, new_p = step(
                QQ[-1], PP[-1], this_dt, reference_ctrl, fixed_locs, fixed_vels, traction_fn(TT[-1]))

            success = onp.all(onp.isfinite(new_q))
            if success:
                break
            else:
                this_dt = this_dt / 2.0
                print('\tFailed to converge. dt now %f' % (this_dt))

        QQ.append(new_q)
        PP.append(new_p)
        TT.append(TT[-1] + this_dt)
        all_fixed.append(fixed_locs_fn(TT[-1]))
        all_fixed_vels.append(fixed_locs_fn(TT[-1]))
        t1 = time.time()
        print(f'stepped to time {TT[-1]} in {t1-t0} seconds')

    print('Saving result in video.')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        QQ, PP, all_fixed, all_fixed_vels)
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)
    print(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    main()
