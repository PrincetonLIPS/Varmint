import os
import time
import collections

import jax
import numpy as np

from mpi4py import MPI
from absl import logging


# Adapted from https://gist.github.com/muammar/2baec60fa8c7e62978720686895cdb9f
class MPIFileHandler(logging.logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD ): 
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        self.setFormatter(logging.PythonFormatter())
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
           logging.logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept 
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
            #self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None


def rprint(*args, comm=MPI.COMM_WORLD, **kwargs):
    if comm.rank == 0:
        print(*args, flush=True, **kwargs)


def all_proc_names(comm=MPI.COMM_WORLD):
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


def pytree_reduce(pytree, comm=MPI.COMM_WORLD, scale=1.0):
    raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
    reduce_sum = comm.allreduce(raveled.block_until_ready(), op=MPI.SUM)

    return unravel(reduce_sum * scale)


def test_pytrees_equal(pytree, comm=MPI.COMM_WORLD, verbose=False):
    if comm.rank == 0 and verbose:
        print('Testing if parameters have deviated.')
        vtime = time.time()
    raveled, unravel = jax.flatten_util.ravel_pytree(pytree)
    all_params = comm.gather(raveled.block_until_ready(), root=0)
    if comm.rank == 0:
        for i in range(comm.Get_size() - 1):
            assert np.allclose(all_params[i], all_params[i+1])
        if verbose:
            print(f'\tVerified in {time.time() - vtime} s.')

