"""Integrands of the mass and stiffness matrices, following Appendix A.

Each kernel receives the chart geometry and the two nodal basis functions
(with their gradients) and returns the integrand at every quadrature node,
including the volume factor sqrt(det g). Vector field kernels return 2x2
blocks indexed by the coordinate directions d/du_1, d/du_2.
"""

import numpy as np


def function_mass(geo, phi_i, dphi_i, phi_j, dphi_j):
    return phi_i[None, :] * phi_j[None, :] * geo.sqrt_det


def laplace_beltrami(geo, phi_i, dphi_i, phi_j, dphi_j):
    return np.einsum("k,bqkl,l->bq", dphi_i, geo.inverse, dphi_j) * geo.sqrt_det


def vector_mass(geo, phi_i, dphi_i, phi_j, dphi_j):
    return (phi_i[None, :] * phi_j[None, :] * geo.sqrt_det)[..., None, None] * geo.g


def bochner(geo, phi_i, dphi_i, phi_j, dphi_j):
    """<grad_g(phi_i d/du_p), grad_g(phi_j d/du_q)>_g with the (2,0) tensor
    inner product trace(X g Y^T g)."""
    x = geo.gradient_matrices(phi_i, dphi_i) @ geo.g[:, :, None]
    y = geo.g[:, :, None] @ geo.gradient_matrices(phi_j, dphi_j)
    return np.einsum("bqpij,bqrij->bqpr", x, y) * geo.sqrt_det[..., None, None]


def hodge(geo, phi_i, dphi_i, phi_j, dphi_j):
    """<d flat v, d flat w>_g + <d^* flat v, d^* flat w>_g for
    v = phi_i d/du_p and w = phi_j d/du_q."""
    d_i = geo.exterior_derivative(phi_i, dphi_i)
    d_j = geo.exterior_derivative(phi_j, dphi_j)
    ds_i = geo.codifferential(phi_i, dphi_i)
    ds_j = geo.codifferential(phi_j, dphi_j)
    two_forms = d_i[..., :, None] * d_j[..., None, :] / geo.det[..., None, None]
    functions = ds_i[..., :, None] * ds_j[..., None, :]
    return (two_forms + functions) * geo.sqrt_det[..., None, None]
