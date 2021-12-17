import mshr
import fenics as fa
import math
import numpy as np


def make_metamaterial_mesh(
    L0, c1, c2, pore_radial_resolution, min_feature_size, resolution, n_cells, porosity, withpore=True,
):

    material_domain = None
    base_pore_points = None
    for i in range(n_cells):
        for j in range(n_cells):
            c1_ = c1[i, j] if isinstance(c1, np.ndarray) else c1
            c2_ = c2[i, j] if isinstance(c2, np.ndarray) else c2

            if isinstance(c1, np.ndarray) or base_pore_points is None:
                base_pore_points, radii, thetas = build_base_pore(
                    L0, c1_, c2_, pore_radial_resolution, porosity
                )

                verify_params(base_pore_points, radii, L0, min_feature_size)

            cell = make_cell(i, j, L0, base_pore_points, withpore)
            material_domain = (
                cell if material_domain is None else cell + material_domain
            )

    return mshr.generate_mesh(material_domain, resolution * n_cells)


def build_base_pore(L0, c1, c2, n_points, porosity):
    # pdb.set_trace()
    r0 = L0 * math.sqrt(2 * porosity) / math.sqrt(math.pi * (2 + c1 ** 2 + c2 ** 2))

    def coords_fn(theta):
        return r0 * (1 + c1 * fa.cos(4 * theta) + c2 * fa.cos(8 * theta))

    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [coords_fn(float(i) * 2 * math.pi / n_points) for i in range(n_points)]
    points = [
        (rtheta * np.cos(theta), rtheta * np.sin(theta))
        for rtheta, theta in zip(radii, thetas)
    ]
    return np.array(points), np.array(radii), np.array(thetas)


def build_pore_polygon(base_pore_points, offset):
    points = [fa.Point(p[0] + offset[0], p[1] + offset[1]) for p in base_pore_points]
    pore = mshr.Polygon(points)
    return pore


def make_cell(i, j, L0, base_pore_points, withpore=True):
    pore = build_pore_polygon(base_pore_points, offset=(L0 * (i + 0.5), L0 * (j + 0.5)))

    cell = mshr.Rectangle(
        fa.Point(L0 * i, L0 * j), fa.Point(L0 * (i + 1), L0 * (j + 1))
    )

    if withpore:
        material_in_cell = cell - pore
    else:
        material_in_cell = cell
    return material_in_cell


def verify_params(pore_points, radii, L0, min_feature_size):
    """Verify that params correspond to a geometrically valid structure"""
    # check Constraint A
    tmin = L0 - 2 * pore_points[:, 1].max()
    if tmin / L0 <= min_feature_size:
        raise ValueError(
            "Minimum material thickness violated. Params do not "
            "satisfy Constraint A from Overvelde & Bertoldi"
        )

    # check Constraint B
    # Overvelde & Bertoldi check that min radius > 0.
    # we check it is > min_feature_size > 2.0, so min_feature_size can be used
    # to ensure the material can be fabricated
    if radii.min() <= min_feature_size / 2.0:
        raise ValueError(
            "Minimum pore thickness violated. Params do not "
            "satisfy (our stricter version of) Constraint B "
            "from Overvelde & Bertoldi"
        )
