import numpy as np


def abs_smallest_prune(J, params_dict):
    """
    J is an adjacency matrix, but because our graph only has at most a single edge between two nodes,
    only the upper triangle of J encodes the edge weights (ie - J[2,3] is referring to the same edge as
    J[3,2] so we only store it once). As such, we need to compute the cutoff on only the upper triangle
    (the nonzero values) of J. Also, np.percentile doesn't take into account magnitude, only value, so
    in order to not lose all negative values, we use the absolute value of J.

    Parameters:
    - J                     Matrix of couplings between qubits
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed
    """
    shape = np.shape(J)
    flat_J = np.ndarray.flatten(J)
    abs_J = np.abs(flat_J)
    triu_abs_J = np.abs(J[np.triu_indices(shape[0], k=1)])
    cutoff_val = np.percentile(triu_abs_J, params_dict["cutoff_percentile"])
    new_J = np.where(abs_J > cutoff_val, flat_J, 0)

    return np.reshape(new_J, shape)


def no_prune(J, params_dict):
    pass
