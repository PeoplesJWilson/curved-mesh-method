"""Example manifolds from Section 5: sampling and reference eigensolutions."""

import itertools

import numpy as np
from scipy.linalg import eigh


def sample_sphere(n_points, rng, noise=0.0):
    """Uniform samples on the unit sphere. With ``noise`` = eta > 0 every point
    is scaled by 1 + epsilon with epsilon uniform in [-eta/2, eta/2]."""
    points = rng.standard_normal((n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if noise > 0:
        points *= 1 + rng.uniform(-noise / 2, noise / 2, size=(n_points, 1))
    return points


def torus_embedding(theta, phi, major=2.0, minor=1.0):
    return np.stack(
        [
            (major + minor * np.cos(theta)) * np.cos(phi),
            (major + minor * np.cos(theta)) * np.sin(phi),
            minor * np.sin(theta),
        ],
        axis=-1,
    )


def sample_torus(n_points, rng, major=2.0, minor=1.0):
    """Uniform samples on the torus with respect to its volume, by rejection."""
    points = []
    while sum(len(p) for p in points) < n_points:
        theta, phi, w = rng.uniform(0, 2 * np.pi, (3, n_points))
        w = w / (2 * np.pi)
        keep = w <= (major + minor * np.cos(theta)) / (major + minor)
        points.append(torus_embedding(theta[keep], phi[keep], major, minor))
    return np.concatenate(points)[:n_points]


def sphere_eigenvalues(operator, n_modes):
    """Leading eigenvalues of the Bochner or Hodge Laplacian on the unit
    sphere, with multiplicities. Degree l has eigenvalue l(l + 1) for Hodge
    and l(l + 1) - 1 for Bochner, each with multiplicity 2(2l + 1)."""
    shift = {"hodge": 0, "bochner": -1}[operator]
    values = []
    degree = 1
    while len(values) < n_modes:
        values += [degree * (degree + 1) + shift] * (2 * (2 * degree + 1))
        degree += 1
    return np.array(values[:n_modes], dtype=float)


def _harmonics(degree):
    """Independent spherical harmonics of the given degree as functions of
    v in R^3, together with their Euclidean gradients."""
    if degree == 1:
        indices = [(i,) for i in range(3)]
    elif degree == 2:
        indices = [(i, k) for i, k in itertools.combinations_with_replacement(range(3), 2) if (i, k) != (2, 2)]
    elif degree == 3:
        indices = [
            idx for idx in itertools.combinations_with_replacement(range(3), 3) if idx.count(2) < 2
        ]
    else:
        raise ValueError("analytic eigenvector fields are available for degrees 1 to 3")

    def harmonic(v, idx):
        if len(idx) == 1:
            (i,) = idx
            f = v[:, i]
            grad = np.zeros_like(v)
            grad[:, i] = 1
        elif len(idx) == 2:
            i, k = idx
            f = 3 * v[:, i] * v[:, k] - (i == k)
            grad = np.zeros_like(v)
            grad[:, i] += 3 * v[:, k]
            grad[:, k] += 3 * v[:, i]
        else:
            i, j, k = idx
            f = (
                15 * v[:, i] * v[:, j] * v[:, k]
                - 3 * (i == j) * v[:, k]
                - 3 * (k == i) * v[:, j]
                - 3 * (j == k) * v[:, i]
            )
            grad = np.zeros_like(v)
            grad[:, i] += 15 * v[:, j] * v[:, k] - 3 * (j == k)
            grad[:, j] += 15 * v[:, i] * v[:, k] - 3 * (k == i)
            grad[:, k] += 15 * v[:, i] * v[:, j] - 3 * (i == j)
        return f, grad

    return [lambda v, idx=idx: harmonic(v, idx) for idx in indices]


def sphere_eigenvector_fields(points, n_modes):
    """Analytic eigenvector fields of the 1-Laplacians on the unit sphere
    evaluated at ``points``, shape (n_modes, N, 3), ordered by eigenvalue.

    Each degree contributes the fields P grad f and n x grad f for the
    harmonics f of that degree. Points are projected to the sphere first and
    every field is normalized to unit root mean square norm.
    """
    v = points / np.linalg.norm(points, axis=1, keepdims=True)
    fields = []
    degree = 1
    while len(fields) < n_modes:
        for harmonic in _harmonics(degree):
            _, grad = harmonic(v)
            tangential = grad - np.sum(grad * v, axis=1, keepdims=True) * v
            fields.append(tangential)
            fields.append(np.cross(v, grad))
        degree += 1
    fields = np.array(fields[:n_modes])
    return fields / np.sqrt(np.mean(np.sum(fields**2, axis=2), axis=1))[:, None, None]


def torus_laplace_beltrami_eigenvalues(n_modes, major=2.0, minor=1.0, n_grid=2000, max_frequency=12):
    """Semi-analytic eigenvalues of the Laplace-Beltrami operator on the
    torus, with multiplicities, by separation of variables. The angular
    problem in theta is solved with second order finite differences."""
    h = 2 * np.pi / n_grid
    theta = np.arange(n_grid) * h
    w = major + minor * np.cos(theta)
    w_half = major + minor * np.cos(theta + h / 2)
    # -(w Theta')' + m^2 / w Theta = lambda w Theta on a periodic grid,
    # scaled by the minor radius so the metric is dtheta^2 + w^2 dphi^2.
    off = -w_half / (minor**2 * h**2)
    stiffness = np.diag(-(off + np.roll(off, 1))) + np.diag(off[:-1], 1) + np.diag(off[:-1], -1)
    stiffness[0, -1] = stiffness[-1, 0] = off[-1]
    mass = np.diag(w)

    values = []
    for m in range(max_frequency + 1):
        matrix = stiffness + np.diag(m**2 / w)
        modes = eigh(matrix, mass, subset_by_index=[0, n_modes - 1], eigvals_only=True)
        multiplicity = 1 if m == 0 else 2
        values += [x for x in modes for _ in range(multiplicity)]
    values = np.sort(np.array(values))
    values[np.abs(values) < 1e-8] = 0.0
    return values[:n_modes]


def torus_hodge_eigenvalues(n_modes, **kwargs):
    """Eigenvalues of the Hodge Laplacian on the torus: two harmonic modes,
    then each nontrivial Laplace-Beltrami eigenvalue with doubled multiplicity."""
    scalar = torus_laplace_beltrami_eigenvalues(n_modes, **kwargs)
    nontrivial = scalar[scalar > 0]
    values = np.concatenate([[0.0, 0.0], np.repeat(nontrivial, 2)])
    return values[:n_modes]
