# Varmint
Formerly: The Variational Material Integrator

A robust and differentiable simulator for the statics and dynamics of large
deformation continuum mechanics, with a focus on designing Neuromechanical Autoencoders.

Uses Automatic Differentiation as a first-class citizen. Optimization is done directly
by computing gradients and sparse Hessians of the potential and kinetic energies.
Differentiation through the nonlinear solver is carried out through adjoint methods.

## Installation
```
conda env create -f environment.yml
conda activate varmint
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Then create a file called `local_config.py` containing:
```
exp_root = USE_YOUR_ROOT_EXPERIMENT_DIRECTORY
source_root = ABSOLUTE_ROOT_DIRECTORY_OF_REPO
```

## Running with MPI
After activating the conda environment, on ionic you can use commands of the form:
```
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_nma_digital_mnist_finetune.py -n exp_name
```
to launch jobs with MPI. The above command will use 4 nodes, using 4 GPUs per node (16 MPI ranks in total).

## Publications
Neuromechanical Autoencoders: Learning to Couple Elastic and Neural Network Nonlinearity\
Deniz Oktay, Mehran Mirramezani, Eder Medina, Ryan P. Adams - In submission

A rapid and automated approach to the design of multistable metastructures\
Mehran Mirramezani\*, Deniz Oktay\*, Ryan P. Adams - In preparation

