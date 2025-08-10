# ABIC LiNGAM — Differentiable Causal Discovery under Latent Confounding

**ABIC LiNGAM** learns **acyclic directed mixed graphs (ADMGs)** with both directed edges and bidirected (latent‑confounding) edges via **continuous optimization**. It extends ABIC to **non‑Gaussian** errors using the **Multivariate Generalized Gaussian Distribution (MGGD)** with shape parameter **β**.

- **β = 1 (Gaussian)** → reduces to an ABIC‑style score.  
- **β ≠ 1 (non‑Gaussian)** → higher‑order information often improves **edge orientation**.

**Clean‑room implementation**: autograd‑only gradients (no custom VJPs), alternative smooth acyclicity/bow‑free penalties, and pseudo‑variable construction via linear solves.

> Inspiration (not a copy):  
> • R. Bhattacharya, T. Nagarajan, D. Malinsky, I. Shpitser, *Differentiable causal discovery under unmeasured confounding*, AISTATS 2021.  
> • dcd repository (ABIC, Gaussian): https://gitlab.com/rbhatta8/dcd  
> Related paper for this repo (TMLR, Aug 2025): https://openreview.net/forum?id=4056

---

## Install

```bash
pip install numpy autograd scipy
# Optional if you use the plotting option in the beta estimator: matplotlib
````

---

## Quick start

```python
import numpy as np
from abic_lingam import ABIC_LiNGAM  # module from this repo

# X: (n, d) with rows = samples, columns = variables; center columns to mean 0
rng = np.random.default_rng(0)
n, d = 1000, 5
X = rng.normal(size=(n, d))
X -= X.mean(axis=0, keepdims=True)

# Non-Gaussian (try beta=3). Use beta=1 for the Gaussian/ABIC case.
model = ABIC_LiNGAM(X, beta=3.0, lam=0.05, rng=0)

# Optional prior knowledge
levels = [["x0","x1"], ["x2","x3","x4"]]  # forbid edges from later tier to earlier tier
exogenous = {"x0"}                        # forbid bidirected edges touching x0

B, Omega = model.fit(levels=levels, exogenous=exogenous, verbose=False)

print("Directed B:\n", B)         # directed coefficients
print("Bidirected Omega:\n", Omega)  # bidirected/noise covariance (diag = noise)
```

Notes:

* Please **center** `X` (mean 0 per column).
* Variable names inside constraints are `"x0" ... "x{d-1}"` following column order.
* Small weights are thresholded to zero at the end (`w_threshold`).

---

## Estimating **β** (MGGD) from your data

This repo includes a simple MGGD estimator (Code 2) that alternates Σ updates and solves a scalar equation for β with `scipy.optimize.root_scalar`.

**Global β from multivariate X:**

```python
# assuming Code 2 is saved as mggd.py in your repo
from mggd import EstMGGD

est = EstMGGD(X, display=False, plot=False)  # X: (n, d)
res = est.estimate()
beta_hat = float(res["beta"])

model = ABIC_LiNGAM(X, beta=beta_hat, lam=0.05, rng=0)
B, Omega = model.fit()
```

**Per‑variable β (optional, robust summary):**

```python
from mggd import EstMGGD
bets = []
for j in range(X.shape[1]):
    r = EstMGGD(X[:, j], display=False).estimate()  # 1-D allowed
    bets.append(float(r["beta"]))
beta_hat = np.median(bets)  # or max(bets)

model = ABIC_LiNGAM(X, beta=beta_hat).fit()
```

---

## What’s inside (brief)

* **Smooth acyclicity**: truncated series $\sum_{k=1}^K \mathrm{tr}((B\circ B)^k)/k!$ (Hadamard‑squared).
* **Bow‑free surrogate**: penalizes overlap between directed and bidirected supports.
* **Pseudo variables**: build $Z$ via linear solves with $\Omega_{-j,-j}$ (avoid explicit inversion).
* **Augmented Lagrangian** with L‑BFGS‑B (autograd gradients only).

---

## Citation

If you use this repository, please cite:

```
@article{MorinishiShimizu2025ABICLiNGAM,
  title   = {Differentiable Causal Discovery of Linear Non-Gaussian Acyclic Models Under Unmeasured Confounding},
  author  = {Yoshimitsu Morinishi and Shohei Shimizu},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  month   = {August},
  url     = {https://openreview.net/forum?id=4056}
}
```

Inspiration:

```
@inproceedings{Bhattacharya2021ABIC,
  title     = {Differentiable causal discovery under unmeasured confounding},
  author    = {Bhattacharya, R. and Nagarajan, T. and Malinsky, D. and Shpitser, I.},
  booktitle = {AISTATS},
  pages     = {2314--2322},
  year      = {2021},
  month     = {March}
}
```

---

## License

**MIT License** (see `LICENSE`).
This is a **clean‑room** implementation. Any third‑party resources (e.g., dcd) remain under their own licenses.

---

## Acknowledgements

We thank Bhattacharya et al. (2021) and the dcd project for foundational ideas on differentiable ADMG constraints. This repository re‑implements the approach with a non‑Gaussian score and alternative penalties—**not** a copy of existing code.

```
```
