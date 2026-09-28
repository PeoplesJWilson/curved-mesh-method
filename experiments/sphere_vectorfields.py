"""Figure 3: the first four Bochner eigenvector fields on the sphere,
estimated from N = 6000 points, next to the analytic fields."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, log, method_from_args, parser
from cmm import solve_eigenproblem
from cmm.manifolds import sample_sphere, sphere_eigenvalues, sphere_eigenvector_fields
from cmm.metrics import eigenspaces

N_POINTS = 6000
N_SHOWN = 4
N_ARROWS = 2000


def matched_estimates(fields, estimated, group_sizes):
    """Least squares fit of every analytic field by the estimated eigenspace."""
    matched, start = [], 0
    for size in group_sizes:
        basis = estimated[start:start + size].reshape(size, -1).T
        for field in fields[start:start + size]:
            coefficients, *_ = np.linalg.lstsq(basis, field.ravel(), rcond=None)
            matched.append((basis @ coefficients).reshape(field.shape))
        start += size
    return np.array(matched)


def quiver(ax, points, field, title):
    norm = np.linalg.norm(field, axis=1)
    ax.quiver(*points.T, *field.T, length=0.08, normalize=False, colors=plt.cm.viridis(norm / norm.max()))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


def main():
    args = parser(__doc__).parse_args()
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    data = sample_sphere(N_POINTS, rng)
    method = method_from_args(args).fit(data)
    n_modes = 6
    values, vectors = solve_eigenproblem(*method.bochner_laplacian(), n_modes)
    log(f"estimated eigenvalues {np.round(values, 3)}")

    fields = sphere_eigenvector_fields(data, n_modes)
    estimated = np.moveaxis(method.to_ambient(vectors), 2, 0)
    groups = eigenspaces(sphere_eigenvalues("bochner", n_modes))
    matched = matched_estimates(fields, estimated, groups)

    subset = rng.choice(N_POINTS, N_ARROWS, replace=False)
    fig, axes = plt.subplots(2, N_SHOWN, figsize=(3.2 * N_SHOWN, 6.4), subplot_kw={"projection": "3d"})
    for m in range(N_SHOWN):
        quiver(axes[0, m], data[subset], matched[m][subset], f"CMM, mode={m + 1}")
        quiver(axes[1, m], data[subset], fields[m][subset], f"truth, mode={m + 1}")
    fig.tight_layout()
    fig.savefig(FIGURES / "sphere_vectorfields.png", dpi=150)
    log(f"saved {FIGURES / 'sphere_vectorfields.png'}")


if __name__ == "__main__":
    main()
