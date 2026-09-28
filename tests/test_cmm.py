import numpy as np
import pytest

from cmm import CurvedMeshMethod, solve_eigenproblem
from cmm.manifolds import (
    sample_sphere,
    sample_torus,
    sphere_eigenvalues,
    sphere_eigenvector_fields,
    torus_hodge_eigenvalues,
)
from cmm.metrics import eigenspaces, eigenvalue_error, eigenvector_error


@pytest.fixture(scope="module")
def sphere():
    rng = np.random.default_rng(0)
    data = sample_sphere(3000, rng)
    return data, CurvedMeshMethod(n_neighbors=40).fit(data)


def test_matrices_are_symmetric(sphere):
    _, method = sphere
    for stiffness, mass in (method.laplace_beltrami(), method.bochner_laplacian(), method.hodge_laplacian()):
        assert abs(stiffness - stiffness.T).max() < 1e-12
        assert abs(mass - mass.T).max() < 1e-12


def test_mass_matrix_measures_area(sphere):
    _, method = sphere
    mass = method.function_mass_matrix()
    assert mass.sum() == pytest.approx(4 * np.pi, rel=0.05)


def test_laplace_beltrami_spectrum(sphere):
    _, method = sphere
    values, _ = solve_eigenproblem(*method.laplace_beltrami(), n_modes=9, sigma=-0.5)
    reference = np.array([0, 2, 2, 2, 6, 6, 6, 6, 6])
    assert abs(values[0]) < 0.05
    assert np.mean(np.abs(values[1:] - reference[1:]) / reference[1:]) < 0.05


@pytest.mark.parametrize("operator", ["bochner", "hodge"])
def test_vector_laplacians_on_sphere(sphere, operator):
    data, method = sphere
    stiffness, mass = getattr(method, f"{operator}_laplacian")()
    values, vectors = solve_eigenproblem(stiffness, mass, n_modes=16)
    reference = sphere_eigenvalues(operator, 16)
    assert eigenvalue_error(reference, values) < 0.06

    fields = sphere_eigenvector_fields(data, 6)
    estimated = np.moveaxis(method.to_ambient(vectors[:, :6]), 2, 0)
    assert eigenvector_error(fields, estimated, eigenspaces(reference[:6])) < 1e-3


def test_ambient_round_trip(sphere):
    data, method = sphere
    fields = sphere_eigenvector_fields(data, 2)
    coefficients = method.from_ambient(np.moveaxis(fields, 0, 2))
    recovered = np.moveaxis(method.to_ambient(coefficients), 2, 0)
    assert np.sqrt(np.mean((recovered - fields) ** 2)) < 1e-3


def test_hodge_on_torus():
    rng = np.random.default_rng(1)
    data = sample_torus(4000, rng)
    method = CurvedMeshMethod(n_neighbors=40).fit(data)
    values, _ = solve_eigenproblem(*method.hodge_laplacian(), n_modes=12, sigma=-0.1)
    reference = torus_hodge_eigenvalues(12)
    assert np.all(np.abs(values[:2]) < 0.05)
    assert eigenvalue_error(reference[2:], values[2:]) < 0.1


def test_quadrature_rules_agree_in_the_limit():
    rng = np.random.default_rng(2)
    data = sample_sphere(2000, rng)
    reference = sphere_eigenvalues("bochner", 6)
    for rule in ("vertex", "midpoint", "dunavant6"):
        method = CurvedMeshMethod(n_neighbors=40, quadrature=rule).fit(data)
        values, _ = solve_eigenproblem(*method.bochner_laplacian(), n_modes=6)
        assert eigenvalue_error(reference, values) < 0.1
