"""Figure 5: Bochner Laplacian on the sphere with noise in the normal direction."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    SPHERE_SIZES,
    log,
    method_from_args,
    output_dirs,
    parser,
    reference_line,
    sizes_and_trials,
    sphere_trial,
)

NOISE_LEVELS = [0.001, 0.01, 0.1]
COLORS = ["blue", "red", "green"]


def main():
    args = parser(__doc__).parse_args()
    sizes, trials = sizes_and_trials(args, SPHERE_SIZES)
    FIGURES, RESULTS = output_dirs(args)
    path = RESULTS / "noisy_sphere.npz"

    if args.replot:
        saved = np.load(path)
        sizes, eig, vec = saved["sizes"], saved["eigenvalues"], saved["eigenvectors"]
    else:
        eig = np.zeros((len(NOISE_LEVELS), trials, len(sizes)))
        vec = np.zeros_like(eig)
        for i, eta in enumerate(NOISE_LEVELS):
            for j, n in enumerate(sizes):
                for t in range(trials):
                    eig[i, t, j], vec[i, t, j] = sphere_trial(
                        n, "bochner", method_from_args(args, n), seed=1000 * t + j, noise=eta
                    )
                    log(f"eta={eta} N={n} trial={t} eigenvalues={eig[i, t, j]:.3e} eigenvectors={vec[i, t, j]:.3e}")
        np.savez(path, sizes=np.array(sizes), eigenvalues=eig, eigenvectors=vec)

    sizes = np.asarray(sizes, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, errors, label, style in zip(axes, (eig, vec), ("(a)", "(b)"), (":", "--")):
        for i, eta in enumerate(NOISE_LEVELS):
            ax.loglog(sizes, errors[i].mean(axis=0), color=COLORS[i], label=rf"$\eta = {eta}$")
            ax.loglog(sizes, errors[i].T, "x", color=COLORS[i])
        reference_line(ax, sizes, errors[0], style)
        ax.set_xlabel("N")
        ax.set_ylabel("Error")
        ax.set_title(label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "noisy_sphere.png", dpi=150)
    log(f"saved {FIGURES / 'noisy_sphere.png'}")


if __name__ == "__main__":
    main()
