"""Figure 4: Hodge Laplacian on the torus. Convergence of the first ten
nontrivial eigenvalues and a mode by mode comparison for N = 16000."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    TORUS_SIZES,
    convergence_axes,
    log,
    method_from_args,
    output_dirs,
    parser,
    sizes_and_trials,
)
from cmm import solve_eigenproblem
from cmm.manifolds import sample_torus, torus_hodge_eigenvalues
from cmm.metrics import eigenvalue_error

N_NONTRIVIAL = 10
N_SHOWN = 12


def torus_trial(n_points, method, seed, n_modes):
    rng = np.random.default_rng(seed)
    data = sample_torus(n_points, rng)
    method.fit(data)
    values, _ = solve_eigenproblem(*method.hodge_laplacian(), n_modes, sigma=-0.1)
    return values


def main():
    args = parser(__doc__).parse_args()
    sizes, trials = sizes_and_trials(args, TORUS_SIZES)
    FIGURES, RESULTS = output_dirs(args)
    path = RESULTS / "torus_convergence.npz"
    reference = torus_hodge_eigenvalues(2 + N_NONTRIVIAL)

    if args.replot:
        saved = np.load(path)
        sizes, errors, spectrum = saved["sizes"], saved["errors"], saved["spectrum"]
    else:
        errors = np.zeros((trials, len(sizes)))
        spectrum = None
        for j, n in enumerate(sizes):
            for t in range(trials):
                values = torus_trial(n, method_from_args(args, n), seed=1000 * t + j, n_modes=2 + N_NONTRIVIAL)
                errors[t, j] = eigenvalue_error(reference[2:], values[2:])
                log(f"N={n} trial={t} eigenvalues={errors[t, j]:.3e}")
                if j == len(sizes) - 1 and t == 0:
                    spectrum = values[:N_SHOWN]
        np.savez(path, sizes=np.array(sizes), errors=errors, spectrum=spectrum)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    convergence_axes(axes[0], sizes, errors, title="(a)")
    modes = np.arange(1, N_SHOWN + 1)
    axes[1].plot(modes, reference[:N_SHOWN], "x-", color="blue", label="Analytic")
    axes[1].plot(modes, spectrum, "x-", color="red", label="Estimated")
    axes[1].set_xlabel("Mode")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_xticks(modes)
    axes[1].set_title("(b)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "torus_convergence.png", dpi=150)
    log(f"saved {FIGURES / 'torus_convergence.png'}")


if __name__ == "__main__":
    main()
