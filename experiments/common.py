"""Shared helpers for the experiment scripts."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cmm import CurvedMeshMethod, solve_eigenproblem  # noqa: E402
from cmm.manifolds import sample_sphere, sphere_eigenvalues, sphere_eigenvector_fields  # noqa: E402
from cmm.metrics import eigenspaces, eigenvalue_error, eigenvector_error  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SPHERE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000]
TORUS_SIZES = [4000, 6000, 10000, 12000, 16000]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--neighbors", type=int, default=40, help="nearest neighbors per chart")
    p.add_argument("--trials", type=int, default=5, help="random trials per sample size")
    p.add_argument("--no-refine", action="store_true", help="use plain local PCA tangent spaces")
    p.add_argument("--quick", action="store_true", help="fewer and smaller trials")
    p.add_argument("--replot", action="store_true", help="only redraw from saved results")
    return p


def method_from_args(args):
    return CurvedMeshMethod(n_neighbors=args.neighbors, tangent_refinements=0 if args.no_refine else 1)


def sizes_and_trials(args, sizes):
    if args.quick:
        return sizes[:3], min(args.trials, 2)
    return sizes, args.trials


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sphere_trial(n_points, operator, method, seed, noise=0.0, n_eigenvalues=48, n_eigenvectors=6):
    """Errors of the eigenvalues and eigenvector fields on the (noisy) sphere."""
    rng = np.random.default_rng(seed)
    data = sample_sphere(n_points, rng, noise=noise)
    method.fit(data)
    stiffness, mass = getattr(method, f"{operator}_laplacian")()
    values, vectors = solve_eigenproblem(stiffness, mass, n_eigenvalues)

    reference = sphere_eigenvalues(operator, n_eigenvalues)
    fields = sphere_eigenvector_fields(data, n_eigenvectors)
    estimated = np.moveaxis(method.to_ambient(vectors[:, :n_eigenvectors]), 2, 0)
    return (
        eigenvalue_error(reference, values),
        eigenvector_error(fields, estimated, eigenspaces(reference[:n_eigenvectors])),
    )


def convergence_axes(ax, sizes, eigenvalue_errors, eigenvector_errors=None, title=None):
    """Log-log convergence plot in the style of the paper. Error arrays have
    shape (trials, sizes)."""
    sizes = np.asarray(sizes, dtype=float)
    ax.loglog(sizes, eigenvalue_errors.mean(axis=0), color="blue", label="Eigenvalues")
    ax.loglog(sizes, eigenvalue_errors.T, "x", color="blue")
    anchor = eigenvalue_errors.mean(axis=0)[0]
    ax.loglog(sizes, 0.6 * anchor * (sizes[0] / sizes) ** 0.5, ":", color="black", label=r"$O(N^{-1/2})$")
    if eigenvector_errors is not None:
        ax.loglog(sizes, eigenvector_errors.mean(axis=0), color="red", label="Eigenvectors")
        ax.loglog(sizes, eigenvector_errors.T, "x", color="red")
        anchor = eigenvector_errors.mean(axis=0)[0]
        ax.loglog(sizes, anchor * (sizes[0] / sizes), "--", color="black", label=r"$O(N^{-1})$")
    ax.set_xlabel("N")
    ax.set_ylabel("Error")
    if title:
        ax.set_title(title)
    ax.legend()
