"""The local curved mesh method."""

import numpy as np
from scipy.sparse.linalg import eigsh

from . import kernels
from .assembly import assemble_function_matrix, assemble_vector_matrix
from .charts import LocalCharts
from .quadrature import triangle_rule


class CurvedMeshMethod:
    """Weak form estimation of Laplacians from point cloud data.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors (including the point itself) used for the
        local charts.
    quadrature : str
        Rule on the reference triangle: "vertex" (used in the paper),
        "midpoint" or "dunavant6".
    tangent_refinements : int
        Number of times the local PCA tangent spaces are corrected with the
        tilt of the GMLS fit. Zero uses plain local PCA.

    After ``fit(data)`` the operator methods return a pair of sparse matrices
    (stiffness, mass) whose generalized eigenvalue problem approximates the
    corresponding eigenvalue problem on the manifold. Vector fields are
    represented by 2N coefficients in the local tangent bases, see
    ``to_ambient`` and ``from_ambient``.
    """

    def __init__(self, n_neighbors=40, quadrature="vertex", tangent_refinements=1):
        self.n_neighbors = n_neighbors
        self.quadrature = quadrature
        self.tangent_refinements = tangent_refinements
        self.charts = None
        self._cache = {}

    def fit(self, data):
        self.charts = LocalCharts(data, self.n_neighbors, self.tangent_refinements)
        self._nodes, self._weights = triangle_rule(self.quadrature)
        self._cache = {}
        return self

    def _check_fitted(self):
        if self.charts is None:
            raise RuntimeError("call fit(data) first")

    def _function_matrix(self, name, kernel):
        self._check_fitted()
        if name not in self._cache:
            self._cache[name] = assemble_function_matrix(
                self.charts, kernel, self._nodes, self._weights
            )
        return self._cache[name]

    def _vector_matrix(self, name, kernel):
        self._check_fitted()
        if name not in self._cache:
            self._cache[name] = assemble_vector_matrix(
                self.charts, kernel, self._nodes, self._weights
            )
        return self._cache[name]

    def function_mass_matrix(self):
        return self._function_matrix("function_mass", kernels.function_mass)

    def vector_field_mass_matrix(self):
        return self._vector_matrix("vector_mass", kernels.vector_mass)

    def laplace_beltrami(self):
        """Stiffness and mass matrices of the Laplace-Beltrami operator (N x N)."""
        return self._function_matrix("laplace_beltrami", kernels.laplace_beltrami), self.function_mass_matrix()

    def bochner_laplacian(self):
        """Stiffness and mass matrices of the Bochner Laplacian (2N x 2N)."""
        return self._vector_matrix("bochner", kernels.bochner), self.vector_field_mass_matrix()

    def hodge_laplacian(self):
        """Stiffness and mass matrices of the Hodge Laplacian (2N x 2N)."""
        return self._vector_matrix("hodge", kernels.hodge), self.vector_field_mass_matrix()

    def to_ambient(self, coefficients):
        """Map tangent coefficients (2N,) or (2N, m) to ambient vectors
        (N, n) or (N, n, m)."""
        self._check_fitted()
        bases = self.charts.tangent_bases
        w = np.asarray(coefficients).reshape(self.charts.n_points, 2, -1)
        fields = np.einsum("ink,ikm->inm", bases, w)
        return fields[..., 0] if np.ndim(coefficients) == 1 else fields

    def from_ambient(self, fields):
        """Project ambient vectors (N, n) or (N, n, m) onto the tangent bases,
        returning coefficients (2N,) or (2N, m)."""
        self._check_fitted()
        bases = self.charts.tangent_bases
        v = np.asarray(fields).reshape(self.charts.n_points, bases.shape[1], -1)
        w = np.einsum("ink,inm->ikm", bases, v).reshape(2 * self.charts.n_points, -1)
        return w[:, 0] if np.ndim(fields) == 2 else w


def solve_eigenproblem(stiffness, mass, n_modes, sigma=-1.0):
    """Smallest eigenpairs of stiffness w = lambda mass w, ascending."""
    values, vectors = eigsh(stiffness, k=n_modes, M=mass, sigma=sigma, which="LM")
    order = np.argsort(values)
    return values[order], vectors[:, order]
