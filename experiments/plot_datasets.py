"""Figure 1: example datasets with N = 4000 points."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, log
from cmm.manifolds import sample_sphere, sample_torus


def main():
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    datasets = [
        ("(a)", sample_sphere(4000, rng)),
        ("(b)", sample_torus(4000, rng)),
        ("(c)", sample_sphere(4000, rng, noise=0.1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), subplot_kw={"projection": "3d"})
    for ax, (title, points) in zip(axes, datasets):
        ax.scatter(*points.T, s=1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect(np.ptp(points, axis=0))
    fig.tight_layout()
    fig.savefig(FIGURES / "datasets.png", dpi=150)
    log(f"saved {FIGURES / 'datasets.png'}")


if __name__ == "__main__":
    main()
