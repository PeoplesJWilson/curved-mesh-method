"""Sparse assembly of mass and stiffness matrices from local kernels."""

import numpy as np
from scipy.sparse import coo_matrix

from .geometry import ChartGeometry
from .quadrature import edge_function, hat_function

CHUNK = 20000


def _chunks(total):
    for start in range(0, total, CHUNK):
        yield slice(start, min(start + CHUNK, total))


def _entries(charts, diagonal):
    """Owner, triangle vertices and column point of each local integral."""
    if diagonal:
        owner, triangles = charts.ring_owner, charts.ring_triangles
        return owner, triangles, owner
    return charts.edge_owner, charts.edge_triangles, charts.edge_neighbor


def _integrate(charts, kernel, nodes, weights, owner, triangles, diagonal):
    p1 = charts.tangential[owner, triangles[:, 1]]
    p2 = charts.tangential[owner, triangles[:, 2]]
    geo = ChartGeometry(charts.coefficients[owner], p1, p2, nodes)
    phi_i, dphi_i = hat_function(nodes)
    phi_j, dphi_j = (phi_i, dphi_i) if diagonal else edge_function(nodes)
    values = kernel(geo, phi_i, dphi_i, phi_j, dphi_j)
    return np.einsum("q,bq...->b...", weights, values), p1, p2


def _symmetrize(matrix):
    matrix = matrix.tocsr()
    return 0.5 * (matrix + matrix.T)


def assemble_function_matrix(charts, kernel, nodes, weights):
    """N x N matrix with entries sum over triangles of int kernel(e_i, e_j)."""
    rows, cols, data = [], [], []
    for diagonal in (True, False):
        owner, triangles, column = _entries(charts, diagonal)
        for chunk in _chunks(len(owner)):
            values, _, _ = _integrate(
                charts, kernel, nodes, weights, owner[chunk], triangles[chunk], diagonal
            )
            rows.append(owner[chunk])
            cols.append(column[chunk])
            data.append(values)
    n = charts.n_points
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return _symmetrize(matrix)


def assemble_vector_matrix(charts, kernel, nodes, weights):
    """2N x 2N block matrix for the nodal vector fields e_i t^(i)_k.

    The kernel is evaluated in the coordinate basis of each triangle. The
    coefficients of t^(i)_k and t^(j)_l in that basis are R^{-1} and
    R^{-1} T_i^T T_j with R = [p_1 p_2], as in Appendix A.
    """
    bases = charts.tangent_bases
    rows, cols, data = [], [], []
    for diagonal in (True, False):
        owner, triangles, column = _entries(charts, diagonal)
        for chunk in _chunks(len(owner)):
            i, j = owner[chunk], column[chunk]
            values, p1, p2 = _integrate(
                charts, kernel, nodes, weights, i, triangles[chunk], diagonal
            )
            r_inverse = np.linalg.inv(np.stack([p1, p2], axis=2))
            coef_i = r_inverse
            if diagonal:
                coef_j = r_inverse
            else:
                overlap = np.einsum("bnk,bnl->bkl", bases[i], bases[j])
                coef_j = r_inverse @ overlap
            blocks = np.einsum("bpk,bpq,bql->bkl", coef_i, values, coef_j)

            k, l = np.meshgrid([0, 1], [0, 1], indexing="ij")
            rows.append((2 * i[:, None, None] + k[None]).ravel())
            cols.append((2 * j[:, None, None] + l[None]).ravel())
            data.append(blocks.ravel())
    n = 2 * charts.n_points
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return _symmetrize(matrix)
