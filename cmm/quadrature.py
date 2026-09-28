"""Quadrature rules on the reference triangle {u_1, u_2 >= 0, u_1 + u_2 <= 1}."""

import numpy as np


def triangle_rule(name):
    """Return quadrature nodes (Q, 2) and weights (Q,) summing to 1/2."""
    if name == "vertex":
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        weights = np.full(3, 1.0 / 6.0)
    elif name == "midpoint":
        points = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        weights = np.full(3, 1.0 / 6.0)
    elif name == "dunavant6":
        a, b = 0.445948490915965, 0.091576213509771
        points = np.array(
            [[a, a], [a, 1 - 2 * a], [1 - 2 * a, a],
             [b, b], [b, 1 - 2 * b], [1 - 2 * b, b]]
        )
        weights = 0.5 * np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
    else:
        raise ValueError(f"unknown quadrature rule {name!r}")
    return points, weights


def hat_function(u):
    """Nodal basis function of the base point and its constant gradient."""
    return 1.0 - u[:, 0] - u[:, 1], np.array([-1.0, -1.0])


def edge_function(u):
    """Nodal basis function of the neighbor placed at (1, 0) and its gradient."""
    return u[:, 0].copy(), np.array([1.0, 0.0])
