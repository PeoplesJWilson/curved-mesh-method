"""Reproduce every figure of the paper. Flags are passed on to each script,
for example ``python experiments/run_all.py --quick``."""

import sys

import noisy_sphere
import plot_datasets
import sphere_convergence
import sphere_vectorfields
import torus_convergence
from common import log

SCRIPTS = [
    ("Figure 1, datasets", plot_datasets),
    ("Figure 2, sphere convergence", sphere_convergence),
    ("Figure 3, sphere eigenvector fields", sphere_vectorfields),
    ("Figure 4, torus convergence", torus_convergence),
    ("Figure 5, noisy sphere", noisy_sphere),
]


def main():
    for title, module in SCRIPTS:
        log(title)
        module.main()
    log("all figures written to figures/")


if __name__ == "__main__":
    main()
