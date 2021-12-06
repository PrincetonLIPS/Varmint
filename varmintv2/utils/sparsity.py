import numpy as np
import jax.numpy as jnp

from scipy.sparse import csc_matrix, diags


def row_inds(spm, i):
    # i is the column index
    return spm.indices[spm.indptr[i]:spm.indptr[i+1]]


def row_inds_exclude(spm, i, exc_set):
    indices = set(row_inds(spm, i))
    return indices - exc_set


class IndexCacher:
    def __init__(self, spm):
        self.spm = spm
        self.indices = {}
        self.k = 0
        self.exc_list = set()
        self.inc_list = set(range(spm.shape[0]))

        # Store a iteration number k along with the included index pattern.
        for i in self.inc_list:
            self.indices[i] = (row_inds_exclude(
                self.spm, i, self.exc_list), self.k)

    def row_inds(self, i, k):
        # Get the indices of column i at iteration k.
        # If the indices have not been updated, update them.
        col_indices, cur_k = self.indices[i]
        if cur_k < k:
            self.indices[i] = (row_inds_exclude(self.spm, i, self.exc_list), k)
        elif cur_k > k:
            print('inconsistency!! something is wrong.')

        return self.indices[i][0]  # returns a python set

# NOTE: Must be a csc_matrix


def jvp_groups(sparsemat_csc):
    ic = IndexCacher(sparsemat_csc)
    ncols = sparsemat_csc.shape[0]

    k = 0
    groups = []
    rows = {}  # Dictionary keeping track of row indices for each column.
    while len(ic.exc_list) < ncols:
        k += 1

        # Find the columns that cover the maximum number of
        # row indices that are not accounted for by symmetry.
        sorted_indices = sorted(ic.inc_list, key=lambda x: len(
            ic.row_inds(x, k)), reverse=True)

        # A list of column indices belonging to this group
        kth_group = []

        # The row indices covered by every column vector in this group
        kth_rows = set()

        for i in sorted_indices:
            # Row indices that this column vector will cover
            inds = ic.row_inds(i, k)

            # If the row indices in this column vector will not overlap
            # with existing vectors in this group, add it to the group.
            if len(inds.intersection(kth_rows)) == 0:
                kth_group.append(i)
                rows[i] = inds  # This column will provide these row indices
                kth_rows = kth_rows.union(inds)  # Update row indices

        # Keep track of all groups
        groups.append(kth_group)

        # Convert row indices to a set for set operations
        kth_set = set(kth_group)

        # These rows will be accounted for by symmetry.
        ic.exc_list = ic.exc_list.union(kth_set)
        ic.inc_list = ic.inc_list - kth_set

    return groups, rows


def construct_jvp_mat(sparsemat_csc, groups, rows):
    jvps_mat = np.zeros((len(groups), sparsemat_csc.shape[0]))

    for i, group in enumerate(groups):
        jvps_mat[i][group] = 1

    return jvps_mat


def jvps_to_spmat(sparsemat_csc, groups, rows):
    all_rows = []
    all_cols = []

    jvp_indexer = np.zeros(sparsemat_csc.shape[0] * len(groups), dtype=np.bool)

    for i, g in enumerate(groups):
        indices = []
        all_group_rows = []
        for col in g:
            for row in rows[col]:
                all_group_rows.append(row + i * sparsemat_csc.shape[0])
                indices.append((row, col))
        indices.sort(key=lambda x: x[0])
        for inds in indices:
            all_rows.append(inds[0])
            all_cols.append(inds[1])

        jvp_indexer[all_group_rows] = True

    # row indices, col indices, and binary index array to get data for sparse matrix... (probably)
    return all_rows + all_cols, all_cols + all_rows, jvp_indexer


def pattern_to_reconstruction(sparsemat_csc):
    groups, rows = jvp_groups(sparsemat_csc)

    jvp_mat = jnp.array(construct_jvp_mat(sparsemat_csc, groups, rows))
    all_rows, all_cols, jvp_indexer = jvps_to_spmat(
        sparsemat_csc, groups, rows)

    def reconstruct(jvp_result):
        """Takes as input (njvps, ndof) matrix."""
        nnz = len(all_rows)
        result = jvp_result.flatten()

        data = np.zeros(nnz)
        data[:nnz//2] = result[jvp_indexer]
        data[nnz//2:] = result[jvp_indexer]

        reconstructed = csc_matrix((data, (all_rows, all_cols)))

        # Diagonals will be duplicated, so de-duplicate.
        reconstructed = reconstructed - diags(reconstructed.diagonal()) / 2

        return reconstructed

    return jvp_mat, reconstruct
