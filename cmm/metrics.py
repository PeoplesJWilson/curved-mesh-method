"""Error metrics of Section 5, Equations (eigenvalue-metric) and (eigenvector-metric)."""

import numpy as np


def eigenvalue_error(reference, estimate):
    """Average relative error over the leading modes."""
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    return np.mean(np.abs(reference - estimate) / reference)


def eigenspaces(values, tolerance=1e-6):
    """Sizes of the groups of equal reference eigenvalues, in order."""
    values = np.asarray(values)
    sizes, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or abs(values[i] - values[start]) > tolerance:
            sizes.append(i - start)
            start = i
    return sizes


def eigenvector_error(reference_fields, estimated_fields, group_sizes):
    """Mean square error between reference fields (L, N, n) and their least
    squares fit by the estimated fields of the matching eigenspace."""
    n_modes, n_points, _ = reference_fields.shape
    total, start = 0.0, 0
    for size in group_sizes:
        stop = start + size
        basis = estimated_fields[start:stop].reshape(size, -1).T
        for field in reference_fields[start:stop]:
            target = field.ravel()
            coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
            total += np.sum((target - basis @ coefficients) ** 2)
        start = stop
    return total / (n_points * n_modes)
