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

## Running large paramter sweeps/experiments
The `launch_batch_experiments.py` file allows you to launch a large number of experiments with different parameters. It is compatible with MPI, so each experiment can use multiple GPUs/nodes. To use, first create a file called `experiment.txt` that contains the commands you want to run, with some allowed substitutions. E.g.
```
python examples/topopt_mmb_simulate.py --config.nx=50 --config.ny=50 -n {wid}-nxny50 --exp_root {expdir}
python examples/topopt_mmb_simulate.py --config.nx=80 --config.ny=80 -n {wid}-nxny80 --exp_root {expdir}
```
Each line will be run independently as a command with MPI. Each command becomes a "worker" and is assigned a unique worker ID (wid). The string `{wid}` will be substituted for the wid, and `{expdir}` will be substituted with a directory generated for the worker.

The script also supports combinatorial parameter sweeps:
```
python examples/topopt_mmb_simulate.py --config.nx={P:50,60,70:P} --config.ny={P:30,40,50:P} -n {wid}-nxny50 --exp_root {expdir}
```
The single line experiment above corresponds to 9 workers.

You can run the command with the following:
```
python launch_batch_experiments.py experiment.txt --experiment-name=test_experiment --slurm-overrides="ngpus=1,nnodes=1,time:01:00:00" --nrepeats 1 --experiment-dir=YOUR_ROOT_EXPERIMENT_DIRECTORY
```
The `nrepeats` command is used to launch multiple copies of each worker, each being a Slurm singleton. This will allow jobs to queue up and launch continuously even if each has a time limit. This can be used to abuse the fair scheduler. Please do not do this.

## Publications
Neuromechanical Autoencoders: Learning to Couple Elastic and Neural Network Nonlinearity\
Deniz Oktay, Mehran Mirramezani, Eder Medina, Ryan P. Adams - In submission

A rapid and automated approach to the design of multistable metastructures\
Mehran Mirramezani\*, Deniz Oktay\*, Ryan P. Adams - In preparation

