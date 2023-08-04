For launching with 16 MPI tasks:
```bash
conda activate varmint
export LD_LIBRARY_PATH=/n/fs/mm-iga/miniconda3/envs/varmint/lib:$LD_LIBRARY_PATH
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_rotation.py -n rotation_16mpi
```
For launching double rotation, the only difference is in the config file:
```bash
conda activate varmint
export LD_LIBRARY_PATH=/n/fs/mm-iga/miniconda3/envs/varmint/lib:$LD_LIBRARY_PATH
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_rotation.py -n double_rotation_16mpi --config config/double_rotation.py
```
