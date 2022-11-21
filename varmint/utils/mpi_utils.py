import time
import collections

import jax
import numpy as np

from mpi4py import MPI
from absl import logging


def rprint(*args, comm=MPI.COMM_WORLD, **kwargs):
    if comm.rank == 0:
        print(*args, flush=True, **kwargs)

def all_proc_names(comm):
    """Get a list of the proccess names for each rank."""
    proc_name = MPI.Get_processor_name()
    rank = comm.rank

    if comm.rank == 0:
        all_proc_ranks = comm.gather([proc_name, rank], root=0)
        return dict(collections.Counter([proc[0] for proc in all_proc_ranks]))
    else:
        return None


def find_local_rank(comm=MPI.COMM_WORLD):
    """Figure out a local rank on a node."""
    proc_name = MPI.Get_processor_name()
    rank = comm.rank

    all_proc_ranks = comm.gather([proc_name, rank], root=0)
    global_local_rank_map = None
    if comm.rank == 0:
        # Sort by rank first, then proc_name w/ stable sort.
        all_proc_ranks = sorted(all_proc_ranks, key=lambda x: x[1])
        all_proc_ranks = sorted(all_proc_ranks, key=lambda x: x[0])

        global_local_rank_map = [0] * len(all_proc_ranks)
        local_counter = 0
        global_local_rank_map[all_proc_ranks[0][1]] = local_counter
        for i in range(1, len(all_proc_ranks)):
            if all_proc_ranks[i-1][0] != all_proc_ranks[i][0]:
                local_counter = 0
            else:
                local_counter += 1

            global_local_rank_map[all_proc_ranks[i][1]] = local_counter
    global_local_rank_map = comm.bcast(global_local_rank_map, root=0)
    return global_local_rank_map[comm.rank]


def pytree_reduce(comm, pytree, scale=1.0):
    raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
    reduce_sum = comm.allreduce(raveled.block_until_ready(), op=MPI.SUM)

    return unravel(reduce_sum * scale)


def test_pytrees_equal(comm, pytree):
    if comm.rank == 0:
        print('Testing if parameters have deviated.')
        vtime = time.time()
    raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
    all_params = comm.gather(raveled.block_until_ready(), root=0)
    if comm.rank == 0:
        for i in range(comm.Get_size() - 1):
            assert np.allclose(all_params[i], all_params[i+1])
        print(f'\tVerified in {time.time() - vtime} s.')

