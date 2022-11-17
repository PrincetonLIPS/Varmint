#!/bin/bash
#
#SBATCH --job-name=pore_shape_rff_10_feats_mpitest
#SBATCH -A lips                
#SBATCH --nodes=1             ## Node count
#SBATCH --ntasks-per-node=4   ## Tasks per node
## SBATCH --gres=gpu:1          ## Total number of GPUs
#SBATCH -t 00:10:00            ## Walltime
#
# send mail if the process fails
#SBATCH --mail-type=fail
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=doktay@cs.princeton.edu
#

__conda_setup="$('/n/fs/mm-iga/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/n/fs/mm-iga/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/n/fs/mm-iga/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/n/fs/mm-iga/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate igaconda38
srun --mpi=pmi2 python launch_nma_pore_shape.py -n pore_shape_rff_10_feats_mpitest

