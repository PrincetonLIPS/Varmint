For launching with 16 MPI tasks:
```bash
conda activate varmint
export LD_LIBRARY_PATH=/n/fs/mm-iga/miniconda3/envs/varmint/lib:$LD_LIBRARY_PATH
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_shape_matching.py -n shape_matching_mpi16
```
To freeze the geometry:
```bash
conda activate varmint
export LD_LIBRARY_PATH=/n/fs/mm-iga/miniconda3/envs/varmint/lib:$LD_LIBRARY_PATH
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_shape_matching.py -n shape_matching_mpi16_nogeometry --config.freeze_radii=True --config.perturb_mesh=False
```
