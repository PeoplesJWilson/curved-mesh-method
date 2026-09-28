"""Riemannian metric of the local curved mesh at quadrature points.

A curved triangle is parameterized by (u_1, u_2) -> (v, p(v)) with
v = u_1 p_1 + u_2 p_2 in the tangent plane and p the GMLS polynomial of the
base point. All quantities below are batched over triangles (B) and quadrature
nodes (Q).
"""

import numpy as np


class ChartGeometry:
    """Metric, inverse metric, derivatives and Christoffel symbols on a batch of
    curved triangles.

    Attributes have leading shape (B, Q). ``g`` and ``inverse`` are 2x2,
    ``derivative[..., m, n, k]`` is the partial derivative of g_mn with respect
    to u_k, and ``christoffel[..., i, k, l]`` is Gamma^i_kl.
    """

    def __init__(self, coefficients, p1, p2, nodes):
        # coefficients (B, 6, c) in the order 1, x, y, x^2, xy, y^2
        _, d, e, a, c, b = (coefficients[:, j, :] for j in range(6))
        basis = np.stack([p1, p2], axis=1)  # (B, 2, 2), rows p_1 and p_2

        v = nodes[None, :, 0, None] * p1[:, None, :] + nodes[None, :, 1, None] * p2[:, None, :]
        x, y = v[..., 0, None], v[..., 1, None]
        grad_x = d[:, None] + 2 * a[:, None] * x + c[:, None] * y
        grad_y = e[:, None] + c[:, None] * x + 2 * b[:, None] * y
        gradient = np.stack([grad_x, grad_y], axis=2)  # (B, Q, 2, c)

        hessian = np.empty((len(a), 2, 2, a.shape[1]))
        hessian[:, 0, 0] = 2 * a
        hessian[:, 0, 1] = hessian[:, 1, 0] = c
        hessian[:, 1, 1] = 2 * b

        # dp[..., m, :] is the directional derivative of p along p_m
        dp = np.einsum("bqjc,bmj->bqmc", gradient, basis)
        # h[b, m, k, :] = p_m^T H p_k
        h = np.einsum("bmi,bijc,bkj->bmkc", basis, hessian, basis)

        flat = np.einsum("bmj,bnj->bmn", basis, basis)
        self.g = flat[:, None] + np.einsum("bqmc,bqnc->bqmn", dp, dp)
        self.derivative = (
            np.einsum("bmkc,bqnc->bqmnk", h, dp) + np.einsum("bqmc,bnkc->bqmnk", dp, h)
        )

        g = self.g
        self.det = g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0]
        self.sqrt_det = np.sqrt(self.det)
        self.inverse = np.empty_like(g)
        self.inverse[..., 0, 0] = g[..., 1, 1] / self.det
        self.inverse[..., 0, 1] = -g[..., 0, 1] / self.det
        self.inverse[..., 1, 0] = -g[..., 1, 0] / self.det
        self.inverse[..., 1, 1] = g[..., 0, 0] / self.det

        dg = self.derivative
        # d_k sqrt(det g)
        self.sqrt_det_derivative = (
            g[..., 1, 1, None] * dg[..., 0, 0, :]
            + g[..., 0, 0, None] * dg[..., 1, 1, :]
            - 2 * g[..., 0, 1, None] * dg[..., 0, 1, :]
        ) / (2 * self.sqrt_det[..., None])

        # Gamma^i_kl = 1/2 g^im (d_l g_mk + d_k g_ml - d_m g_kl)
        symbols = dg + dg.transpose(0, 1, 2, 4, 3) - dg.transpose(0, 1, 4, 2, 3)
        self.christoffel = 0.5 * np.einsum("bqim,bqmkl->bqikl", self.inverse, symbols)

    def gradient_matrices(self, phi, dphi):
        """Matrix form of grad_g(phi d/du_m) for m = 1, 2, shape (B, Q, 2, 2, 2).

        Entry [m, i, k] before the metric inverse is d_k(phi) delta_im +
        phi Gamma^i_mk. The result is this matrix times g^{-1}.
        """
        gamma = self.christoffel.transpose(0, 1, 3, 2, 4)  # [m, i, k]
        matrices = phi[None, :, None, None, None] * gamma
        matrices = matrices + np.eye(2)[None, None, :, :, None] * dphi[None, None, None, None, :]
        return matrices @ self.inverse[:, :, None]

    def exterior_derivative(self, phi, dphi):
        """Coefficient of du_1 ^ du_2 in d(flat(phi d/du_m)), shape (B, Q, 2)."""
        g, dg = self.g, self.derivative
        return (
            dphi[0] * g[..., :, 1]
            - dphi[1] * g[..., :, 0]
            + phi[None, :, None] * (dg[..., :, 1, 0] - dg[..., :, 0, 1])
        )

    def codifferential(self, phi, dphi):
        """d^*(flat(phi d/du_m)) for m = 1, 2, shape (B, Q, 2)."""
        return -(dphi[None, None, :] + phi[None, :, None] * self.sqrt_det_derivative / self.sqrt_det[..., None])
