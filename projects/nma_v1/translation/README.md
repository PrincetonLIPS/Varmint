For launching with 16 MPI tasks:
```bash
srun -A lips --mpi=pmi2 -t 48:00:00 --gres=gpu:4 -N 4 --ntasks-per-node 4 python launch_translation.py -n translation_16mpi
```
