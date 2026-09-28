"""Figure 2: convergence of the Bochner and Hodge Laplacians on the sphere."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    SPHERE_SIZES,
    convergence_axes,
    log,
    method_from_args,
    output_dirs,
    parser,
    sizes_and_trials,
    sphere_trial,
)


def main():
    args = parser(__doc__).parse_args()
    sizes, trials = sizes_and_trials(args, SPHERE_SIZES)
    FIGURES, RESULTS = output_dirs(args)
    path = RESULTS / "sphere_convergence.npz"

    if args.replot:
        saved = np.load(path)
        sizes, errors = saved["sizes"], {k: saved[k] for k in saved.files if k != "sizes"}
    else:
        errors = {}
        for operator in ("bochner", "hodge"):
            eig = np.zeros((trials, len(sizes)))
            vec = np.zeros((trials, len(sizes)))
            for j, n in enumerate(sizes):
                for t in range(trials):
                    eig[t, j], vec[t, j] = sphere_trial(n, operator, method_from_args(args, n), seed=1000 * t + j)
                    log(f"{operator} N={n} trial={t} eigenvalues={eig[t, j]:.3e} eigenvectors={vec[t, j]:.3e}")
            errors[f"{operator}_eigenvalues"], errors[f"{operator}_eigenvectors"] = eig, vec
        np.savez(path, sizes=np.array(sizes), **errors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, operator, label in zip(axes, ("bochner", "hodge"), ("(a)", "(b)")):
        convergence_axes(
            ax, sizes, errors[f"{operator}_eigenvalues"], errors[f"{operator}_eigenvectors"], title=label
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "sphere_convergence.png", dpi=150)
    log(f"saved {FIGURES / 'sphere_convergence.png'}")


if __name__ == "__main__":
    main()
