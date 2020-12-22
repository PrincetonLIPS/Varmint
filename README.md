# Varmint
Variational Material Integrator

Goal: A robust and differentiable simulator for the dynamics of large
deformation continuum mechanics.  Should handle collisions and simple
constraints, and ideally run on a GPU with 32-bit floats.

Make testing a priority.

## Milestones
* [completed] Simulate composite shapes in two dimensions using a variational integrator
  of the Lagrangian.
* Introduce adaptive time-stepping to improve performance and stability.
* [completed] Implement the Hamiltonian variant of the variational integrator.
* Extend to three dimensions.
* ~~Implement collisions with external surfaces.~~
* Implement self collisions.
* Compute gradients through variational integrator.