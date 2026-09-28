"""Local charts: nearest neighbors, tangent frames, GMLS polynomials and first rings."""

import numpy as np
from scipy.spatial import Delaunay, KDTree


def nearest_neighbors(data, n_neighbors):
    """Indices of the k nearest neighbors of every point, the point itself first."""
    tree = KDTree(data)
    _, indices = tree.query(data, k=n_neighbors)
    if not np.array_equal(indices[:, 0], np.arange(len(data))):
        raise ValueError("data contains duplicate points")
    return indices


def local_frames(data, knn):
    """Orthonormal frames from local PCA, shape (N, n, n).

    The first two columns span the estimated tangent space, the remaining
    columns span the estimated normal space.
    """
    neighborhoods = data[knn]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("ikp,ikq->ipq", centered, centered)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, :, ::-1]


def local_coordinates(data, knn, frames):
    """Neighbors expressed in the local frame of the base point.

    Returns the tangential coordinates (N, k, 2) and the normal coordinates
    (N, k, n - 2). The base point sits at the origin of its own chart.
    """
    relative = data[knn] - data[:, None, :]
    coordinates = np.einsum("ikn,inm->ikm", relative, frames)
    return coordinates[:, :, :2], coordinates[:, :, 2:]


def refine_frames(data, knn, frames, iterations):
    """Improve the PCA frames using the GMLS fit itself.

    The linear terms of the fitted quadratic give the tilt of the local graph
    at the base point. Rotating the frame so that this tilt vanishes and
    refitting gives a higher order estimate of the tangent space.
    """
    for _ in range(iterations):
        tangential, normal = local_coordinates(data, knn, frames)
        coefficients = fit_polynomials(tangential, normal)
        n_points, n = frames.shape[:2]
        local = np.zeros((n_points, n, n))
        local[:, 0, 0] = local[:, 1, 1] = 1.0
        local[:, 2:, 0] = coefficients[:, 1, :]
        local[:, 2:, 1] = coefficients[:, 2, :]
        local[:, 2:, 2:] = np.eye(n - 2)
        frames, _ = np.linalg.qr(frames @ local)
    return frames


def polynomial_features(v):
    """Quadratic monomials 1, x, y, x^2, xy, y^2 evaluated at v (..., 2)."""
    x, y = v[..., 0], v[..., 1]
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def fit_polynomials(tangential, normal):
    """Weighted least squares fit of the normal coordinates as quadratics in the
    tangential coordinates. The base point has weight one, all others 1/k.

    Returns coefficients (N, 6, n - 2) in the order 1, x, y, x^2, xy, y^2.
    """
    n_neighbors = tangential.shape[1]
    weights = np.full(n_neighbors, 1.0 / n_neighbors)
    weights[0] = 1.0
    features = polynomial_features(tangential)
    weighted = features.transpose(0, 2, 1) * weights
    gram = weighted @ features
    rhs = weighted @ normal
    return np.linalg.pinv(gram) @ rhs


def first_rings(tangential):
    """Delaunay triangles touching the base point of every local chart.

    Returns two integer arrays. ``owner`` (T,) gives the base point of each
    triangle and ``triangles`` (T, 3) its vertices as positions in the
    neighbor list of the owner, with the base point (position 0) first.
    """
    owner, triangles = [], []
    for i, points in enumerate(tangential):
        try:
            simplices = Delaunay(points).simplices
        except Exception as error:
            raise ValueError(f"local mesh failed at point {i}: {error}") from error
        ring = simplices[(simplices == 0).any(axis=1)]
        ring = np.array([[0] + [v for v in tri if v != 0] for tri in ring])
        owner.append(np.full(len(ring), i))
        triangles.append(ring)
    return np.concatenate(owner), np.concatenate(triangles)


class LocalCharts:
    """Everything the assembly needs, computed once from the point cloud."""

    def __init__(self, data, n_neighbors, tangent_refinements=1):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("data must have shape (N, n) with n >= 3")
        if n_neighbors < 7:
            raise ValueError("n_neighbors must be at least 7 to fit a quadratic")
        if n_neighbors > len(data):
            raise ValueError("n_neighbors exceeds the number of points")

        self.data = data
        self.n_neighbors = n_neighbors
        self.knn = nearest_neighbors(data, n_neighbors)
        self.frames = local_frames(data, self.knn)
        self.frames = refine_frames(data, self.knn, self.frames, tangent_refinements)
        self.tangential, self.normal = local_coordinates(data, self.knn, self.frames)
        self.coefficients = fit_polynomials(self.tangential, self.normal)
        self.ring_owner, self.ring_triangles = first_rings(self.tangential)

        # Every ring triangle [0, a, b] contributes to the off diagonal entries
        # (i, a) and (i, b). Store each as [0, vertex, other].
        tri = self.ring_triangles
        flipped = tri[:, [0, 2, 1]]
        self.edge_owner = np.concatenate([self.ring_owner, self.ring_owner])
        self.edge_triangles = np.concatenate([tri, flipped])
        self.edge_neighbor = self.knn[self.edge_owner, self.edge_triangles[:, 1]]

    @property
    def n_points(self):
        return len(self.data)

    @property
    def tangent_bases(self):
        """Tangent basis vectors t_1, t_2 at every point, shape (N, n, 2)."""
        return self.frames[:, :, :2]
