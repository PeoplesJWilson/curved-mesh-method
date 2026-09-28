"""Reproduce every figure of the paper. Flags are passed on to each script,
for example ``python experiments/run_all.py --quick``.

``python experiments/run_all.py --no-refine`` reproduces the settings of the
paper: plain local PCA tangent spaces for the sphere figures, written to
figures/plain. The torus figure is always made with tangent refinement,
since plain local PCA does not converge there, and goes to figures/refined."""

import sys

import noisy_sphere
import plot_datasets
import sphere_convergence
import sphere_vectorfields
import torus_convergence
from common import log

SCRIPTS = [
    ("Figure 1, datasets", plot_datasets, True),
    ("Figure 2, sphere convergence", sphere_convergence, True),
    ("Figure 3, sphere eigenvector fields", sphere_vectorfields, True),
    ("Figure 4, torus convergence", torus_convergence, False),
    ("Figure 5, noisy sphere", noisy_sphere, True),
]


def main():
    flags = sys.argv[1:]
    for title, module, allow_plain in SCRIPTS:
        log(title)
        sys.argv = [sys.argv[0]] + (flags if allow_plain else [f for f in flags if f != "--no-refine"])
        module.main()
    log("done")


if __name__ == "__main__":
    main()
