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

    domain =   Rectangle(dolfin.Point(0., 0.), dolfin.Point(5., 5.)) \
            - Rectangle(dolfin.Point(2., 1.25), dolfin.Point(3., 1.75)) \
            - Circle(dolfin.Point(1, 4), .25) \
            - Circle(dolfin.Point(4, 4), .25)
    mesh2d = generate_mesh(domain, 15)
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

    ref_ctrl = ctrls
    potential_energy_fn = cell.get_potential_energy_fn(ref_ctrl)
    strain_energy_fn = jax.jit(cell.get_strain_energy_fn(ref_ctrl))

    grad_potential_energy_fn = jax.grad(potential_energy_fn)
    hess_potential_energy_fn = jax.hessian(potential_energy_fn)

    potential_energy_fn = jax.jit(potential_energy_fn)
    grad_potential_energy_fn = jax.jit(grad_potential_energy_fn)
    hess_potential_energy_fn = jax.jit(hess_potential_energy_fn)

    l2g, g2l = cell.get_global_local_maps()

    sim_time = time.time()
    curr_g_pos = l2g(ref_ctrl)
    print(f'Optimizing with {curr_g_pos.shape[0]} degrees of freedom.')

    n_increments = 10
    strain_energies = []
    increments = []
    all_displacements = []
    all_velocities = []

    all_fixed_locs = []
    all_fixed_vels = []
    for i in range(n_increments):
        # Increment displacement a little bit.
        fixed_displacements = {
            '1': np.array([0.0, 0.0]),
            '2': np.array([0.0, -1.0 / n_increments * i]),
        }

        tractions = {}

        fixed_locs = cell.fixed_locs_from_dict(ref_ctrl, fixed_displacements)
        all_fixed_locs.append(fixed_locs)
        all_fixed_vels.append(np.zeros_like(fixed_locs))

        tractions = cell.tractions_from_dict(tractions)

        # Solve for new state
        print(f'Starting optimization at iteration {i}.')
        opt_start = time.time()
        results = scipy.optimize.minimize(potential_energy_fn, curr_g_pos,
                                          args=(fixed_locs, tractions),
                                          method='trust-ncg',
                                          jac=grad_potential_energy_fn,
                                          hess=hess_potential_energy_fn,
                                          #options={'maxiter': 10000}
                                          )
        print(f'Optimization succeeded: {results.success}.')
        print(f'Took {time.time() - opt_start} seconds.')
        if not results.success:
            print(f'Optimization failed with status {results.status}.')
            print(results.message)
            break

        curr_g_pos = results.x
        all_displacements.append(curr_g_pos)
        all_velocities.append(np.zeros_like(curr_g_pos))
        strain_energy = strain_energy_fn(curr_g_pos, fixed_locs, tractions)
        print(f'Total strain energy is: {strain_energy} J')

    print('Saving result in video.')
    image_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.png')
    vid_path = os.path.join(args.exp_dir, f'sim-{args.exp_name}.mp4')
    create_static_image(cell.element, g2l(curr_g_pos, fixed_locs), image_path, just_cp=False)
    ctrl_seq, _ = cell.unflatten_dynamics_sequence(
        all_displacements, all_velocities, all_fixed_locs, all_fixed_vels)
    create_movie(cell.element, ctrl_seq, vid_path, comet_exp=None)
    print(f'Finished simulation {args.exp_name}')


if __name__ == '__main__':
    main()
