import fenics as fe
from mshr import *
import numpy as np


def solve_slab_with_hole():
    # Density
    rho = fe.Constant(7850.0)  # kg /m^3

    # Young's modulus and Poisson's ratio
    # Steel material
    E = 200e9  # Pa
    nu = 0.3

    # Lame's constants
    lambda_ = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)

    l_x, l_y = 0.10, 0.05  # Domain dimensions (m)

    # Displacement
    d_x = 0.01

    # Load
    b_z = 0.0
    b = fe.Constant((0.0, b_z))

    # --------------------
    # Functions and classes
    # --------------------

    def left(x, on_boundary):
        return (on_boundary and fe.near(x[0], 0.0))

    def right(x, on_boundary):
        return (on_boundary and fe.near(x[0], l_x))

    # Strain function
    def epsilon(u):
        return fe.sym(fe.grad(u))

    model = "plane_strain"

    if model == "plane_stress":
        lambda_ = 2*mu*lambda_/(lambda_+2*mu)

    # Stress function
    def sigma(u):
        return lambda_ * fe.div(u)*fe.Identity(2) + 2 * mu * epsilon(u)
    # lambda is a reserved python keyword, naming convention recommends using a 
    # single trailing underscore for such cases.

    # --------------------
    # Geometry
    # --------------------
    domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(l_x, l_y)) \
            - Circle(dolfin.Point(l_x / 2.0, l_y / 2.0), l_y / 5.0)
    mesh = generate_mesh(domain, 30)

    # --------------------
    # Function spaces
    # --------------------
    V = fe.VectorFunctionSpace(mesh, "CG", 1)
    u_tr = fe.TrialFunction(V)
    u_test = fe.TestFunction(V)

    # --------------------
    # Boundary conditions
    # --------------------
    bc1 = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), left)
    bc2 = fe.DirichletBC(V, fe.Constant((d_x, 0.0)), right)

    # --------------------
    # Weak form
    # --------------------
    a = fe.inner(sigma(u_tr), epsilon(u_test))*fe.dx
    l = rho*fe.dot(b, u_test)*fe.dx # + fe.inner(g, u_test)*ds(1)

    # --------------------
    # Solver
    # --------------------
    u = fe.Function(V)
    A, L = fe.assemble_system(a, l, [bc1, bc2])

    fe.solve(A, u.vector(), L)

    stress = sigma(u)

    s = stress - (1./3)*fe.tr(stress)*fe.Identity(2)
    von_Mises = fe.sqrt(3./2*fe.inner(s, s))

    V = fe.FunctionSpace(mesh, 'P', 1)
    p_von_Mises = fe.project(von_Mises, V)

    def stress_at(point):
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
        ind = np.argmin(np.linalg.norm(dofs_x - point, axis=-1))
        return p_von_Mises.compute_vertex_values().reshape((-1, 2))[ind]

    w0 = u.compute_vertex_values(mesh)
    nv = mesh.num_vertices()

    X = mesh.coordinates()
    X = [X[:, i] for i in range(2)]
    X = np.stack(X, axis=-1)
    U = [w0[i * nv: (i + 1) * nv] for i in range(2)]
    U = np.stack(U, axis=-1)

    def deformation_at(point):
        ind = np.argmin(np.linalg.norm(X - point, axis=-1))
        return U[ind]

    def ref_at(point):
        ind = np.argmin(np.linalg.norm(X - point, axis=-1))
        return X[ind]

    return (u, A, L), stress_at, deformation_at, ref_at