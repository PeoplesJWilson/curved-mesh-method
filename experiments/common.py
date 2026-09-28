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

SPHERE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000]
TORUS_SIZES = [4000, 6000, 10000, 12000, 16000]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--neighbors",
        type=int,
        default=None,
        help="nearest neighbors per chart (default 40, or 0.7 sqrt(N) with --no-refine)",
    )
    p.add_argument("--trials", type=int, default=5, help="random trials per sample size")
    p.add_argument("--no-refine", action="store_true", help="use plain local PCA tangent spaces")
    p.add_argument("--quick", action="store_true", help="fewer and smaller trials")
    p.add_argument("--replot", action="store_true", help="only redraw from saved results")
    return p


def method_from_args(args, n_points):
    """Plain local PCA needs neighborhoods that grow with N to keep the
    tangent error decaying, as in Section 4; the paper's figures correspond
    to about 0.7 sqrt(N) neighbors. With refinement a fixed 40 suffices."""
    if args.neighbors:
        neighbors = args.neighbors
    elif args.no_refine:
        neighbors = int(round(0.7 * np.sqrt(n_points)))
    else:
        neighbors = 40
    return CurvedMeshMethod(n_neighbors=neighbors, tangent_refinements=0 if args.no_refine else 1)


def output_dirs(args):
    """Figures and cached results go to figures/refined or figures/plain,
    depending on whether tangent refinement is used."""
    variant = "plain" if args.no_refine else "refined"
    figures, results = ROOT / "figures" / variant, ROOT / "results" / variant
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return figures, results


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


def reference_line(ax, sizes, errors, style):
    """Draw O(N^-1) or O(N^-1/2) through the first mean error, whichever
    rate is closer to the fitted slope of the mean error."""
    mean = errors.mean(axis=0)
    slope = -np.polyfit(np.log(sizes), np.log(mean), 1)[0]
    rate = 1.0 if abs(slope - 1.0) < abs(slope - 0.5) else 0.5
    label = r"$O(N^{-1})$" if rate == 1.0 else r"$O(N^{-1/2})$"
    ax.loglog(sizes, mean[0] * (sizes[0] / sizes) ** rate, style, color="black", label=label)


def convergence_axes(ax, sizes, eigenvalue_errors, eigenvector_errors=None, title=None):
    """Log-log convergence plot in the style of the paper. Error arrays have
    shape (trials, sizes)."""
    sizes = np.asarray(sizes, dtype=float)
    ax.loglog(sizes, eigenvalue_errors.mean(axis=0), color="blue", label="Eigenvalues")
    ax.loglog(sizes, eigenvalue_errors.T, "x", color="blue")
    reference_line(ax, sizes, eigenvalue_errors, ":")
    if eigenvector_errors is not None:
        ax.loglog(sizes, eigenvector_errors.mean(axis=0), color="red", label="Eigenvectors")
        ax.loglog(sizes, eigenvector_errors.T, "x", color="red")
        reference_line(ax, sizes, eigenvector_errors, "--")
    ax.set_xlabel("N")
    ax.set_ylabel("Error")
    if title:
        ax.set_title(title)
    ax.legend()
