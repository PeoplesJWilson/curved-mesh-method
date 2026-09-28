# A Higher Order Local Mesh Method for Approximating 1-Laplacians on Unknown Manifolds

This repository contains the implementation described in our paper (https://arxiv.org/abs/2405.15735). The project focuses on approximating vector Laplacians from point cloud data using a higher-order curved mesh method.

If you use this code in your work, please cite:

```
@article{peoples2024higher,
  title={A Higher Order Local Mesh Method for Approximating 1-Laplacians on Unknown Manifolds},
  author={Peoples, John Wilson and Harlim, John},
  journal={arXiv preprint arXiv:2405.15735},
  year={2024}
}
```

## Overview

Given points sampled from a two dimensional manifold embedded in R^n, the curved mesh method (CMM) builds a local chart at every point: the k nearest neighbors are projected onto an estimated tangent plane, a Delaunay triangulation of the projected neighbors gives the first ring of triangles around the point, and a quadratic fitted by generalized moving least squares lifts each triangle to a curved patch. Mass and stiffness matrices of the weak eigenvalue problem are then assembled from integrals over these curved triangles, exactly as in Section 3 and Appendix A of the paper. The package provides

- the Laplace-Beltrami operator on functions (N x N matrices),
- the Bochner Laplacian on vector fields (2N x 2N matrices),
- the Hodge Laplacian on vector fields (2N x 2N matrices).

Vector fields are represented by two coefficients per point in the local tangent basis, see Equation (vecW) of the paper. `to_ambient` and `from_ambient` convert between this representation and ambient vectors.

## Installation

The code needs Python 3.9 or newer with numpy and scipy. The experiments also need matplotlib. With [uv](https://docs.astral.sh/uv/):

```
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

With pyenv or any other Python, the equivalent is

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally install the package itself with `pip install -e .` so that `import cmm` works from anywhere. The scripts below already add the repository root to the path, so this is not required.

## Usage

```python
import numpy as np
from cmm import CurvedMeshMethod, solve_eigenproblem
from cmm.manifolds import sample_sphere

data = sample_sphere(4000, np.random.default_rng(0))      # (N, 3) point cloud

method = CurvedMeshMethod(n_neighbors=40).fit(data)
stiffness, mass = method.bochner_laplacian()              # sparse 2N x 2N matrices
values, vectors = solve_eigenproblem(stiffness, mass, n_modes=16)

fields = method.to_ambient(vectors)                        # (N, 3, 16) eigenvector fields
```

`method.hodge_laplacian()` and `method.laplace_beltrami()` work the same way. The generalized eigenvalue problem is solved with shift and invert ARPACK; pass `sigma` to `solve_eigenproblem` to change the shift (the default of -1 targets the smallest eigenvalues).

Parameters of `CurvedMeshMethod`:

- `n_neighbors`: size of the local neighborhoods. Values around 30 to 50 work well for the examples in the paper. Too few neighbors makes the local meshes of neighboring points inconsistent, which shows up as spurious negative eigenvalues.
- `quadrature`: rule on the reference triangle, `"vertex"` (first order, used in the paper), `"midpoint"` or `"dunavant6"`.
- `tangent_refinements`: number of times the local PCA tangent space is corrected using the GMLS fit, see below. The default of 1 gives a higher order tangent estimate; set it to 0 for plain local PCA as described in Section 2.1.

### Tangent space refinement

The GMLS polynomial p(v1, v2) = a v1^2 + b v2^2 + c v1 v2 + d v1 + e v2 + f is fitted in the frame given by local PCA. If that frame were exactly tangent to the manifold at the base point, the linear coefficients d and e would vanish, since the manifold is the graph of a function with zero slope over its own tangent plane. Nonzero d and e therefore measure the tilt of the PCA plane. The refinement takes the vectors (1, 0, d) and (0, 1, e), which span the tangent plane of the fitted graph at the base point, maps them back to ambient coordinates, orthonormalizes the resulting frame, and refits the polynomial. This is a one step correction in the spirit of the second order local SVD of Harlim, Jiang and Peoples (2023), reusing the fit the method already computes. In practice one pass reduces the tangent error by two orders of magnitude, and the eigenvalue errors of the estimated Laplacians improve accordingly, because the theory of Section 4 assumes exact tangent spaces.

## Reproducing the figures

Run the scripts from the repository root with the environment activated. Each script in `experiments/` writes one figure and caches its raw results in `results/`.

### Results of the paper

```
python experiments/run_all.py --no-refine
```

This uses plain local PCA tangent spaces with about 0.7 sqrt(N) nearest neighbors, the setting that matches the paper, and writes the sphere figures (Figures 1, 2, 3 and 5) to `figures/plain/`. The neighborhoods have to grow with N because the tangent error of local PCA with a fixed number of neighbors does not decay, see Section 4. The torus figure (Figure 4) is an exception: plain local PCA does not converge on the torus with this implementation, so that figure is always produced with tangent refinement and 40 neighbors and lands in `figures/refined/`.

### Results with tangent refinement

```
python experiments/run_all.py
```

This uses the default settings of `CurvedMeshMethod` (tangent refinement, 40 nearest neighbors) and writes all five figures to `figures/refined/`. On clean data the errors are several times smaller than in the paper and decay closer to O(N^-1). On noisy data the refinement is more sensitive than plain local PCA, since it estimates the tangent tilt from the slope of the local fit.

### Individual figures and options

```
python experiments/plot_datasets.py         # Figure 1, example datasets
python experiments/sphere_convergence.py    # Figure 2, Bochner and Hodge convergence on the sphere
python experiments/sphere_vectorfields.py   # Figure 3, estimated and analytic eigenvector fields
python experiments/torus_convergence.py     # Figure 4, Hodge Laplacian on the torus
python experiments/noisy_sphere.py          # Figure 5, robustness to noise
```

All scripts accept `--no-refine` (plain local PCA tangent spaces), `--neighbors` (40 by default, 0.7 sqrt(N) with `--no-refine`), `--trials`, `--quick` (three sample sizes, two trials) and `--replot` (redraw from cached results). The reference lines in the convergence plots show O(N^-1) or O(N^-1/2), whichever is closer to the fitted rate of the mean error. The full runs take a few minutes each on a laptop; the largest single problem (N = 16000) takes about 20 seconds with 40 neighbors.

The reference solutions used for the error metrics are in `cmm/manifolds.py`: analytic eigenvalues and eigenvector fields of the 1-Laplacians on the sphere, and semi-analytic Hodge eigenvalues on the torus obtained from the Laplace-Beltrami spectrum by separation of variables.

## Layout

```
cmm/
  charts.py       nearest neighbors, tangent frames, GMLS fit, first ring meshes
  quadrature.py   rules on the reference triangle and the nodal basis functions
  geometry.py     metric, its derivatives and Christoffel symbols on curved triangles
  kernels.py      integrands of the mass and stiffness matrices (Appendix A)
  assembly.py     sparse assembly of the symmetrized matrices
  method.py       the CurvedMeshMethod class and the eigensolver
  manifolds.py    sampling and reference eigensolutions for the sphere and torus
  metrics.py      error metrics of Section 5
experiments/      scripts that reproduce the figures of the paper
tests/            pytest suite, run with `pytest tests`
```

## Tests

```
pytest tests
```

The tests check symmetry of the matrices, the area computed by the mass matrix, and the spectra of all three operators against the analytic values on the sphere and the semi-analytic values on the torus.
