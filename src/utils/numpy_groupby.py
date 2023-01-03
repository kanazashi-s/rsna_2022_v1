import numpy as np


def mean(value, group):
    _, group = np.unique(group, return_inverse=True)
    group = group.astype(np.intp, copy=False)
    counts = np.bincount(group)
    sums = np.bincount(group, value)

    result_mean = sums/counts
    return result_mean