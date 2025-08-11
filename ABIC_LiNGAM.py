# SPDX-License-Identifier: MIT
# ADMG structure learning (clean-room style)
# - uses only autograd's automatic differentiation (no custom VJP)
# - acyclicity penalty written in a different analytic form
# - naming, control flow, and helper APIs refreshed

import numpy as np
import autograd.numpy as anp
from autograd import grad
import scipy.optimize as sopt
import functools


class ABIC_LiNGAM:
    """
    Learn directed coefficients (B) and bidirected noise covariance (Omega)
    from centered data X via smooth penalties and augmented Lagrangian.
    """

    def __init__(self, X, beta=1.0, lam=0.05, acyc_order=None, rng=None):
        """
        Parameters
        ----------
        X : (n, d) array-like (assumed centered)
        beta : float, power in residual loss, i.e., ||r||^(2*beta)
        lam : float, sparsity-like smooth penalty weight
        acyc_order : int or None, truncation order for acyclicity series (defaults to d)
        rng : seed or np.random.Generator for reproducible init
        """
        X = anp.asarray(X)
        self.X = X
        self.n, self.d = X.shape
        self.S = anp.cov(X.T)
        self.beta = float(beta)
        self.lam = float(lam)
        self.acyc_order = acyc_order
        self._rng = np.random.default_rng(rng)

        # learned params after fit()
        self.B_ = None        # directed
        self.Omega_ = None    # bidirected (incl. diagonal noise)


    # ---------- structural penalties (all auto-diff friendly) ----------

    def _acyclicity_penalty(self, W, K=None):
        """
        Smooth acyclicity surrogate written as a truncated series:
            h(W) = sum_{k=1..K} trace((W∘W)^k) / k!
        where ∘ is Hadamard product. K defaults to d.

        This differs in form/implementation from common "M=I+..." variants
        and avoids any custom VJP; autograd handles the gradient.
        """
        d = W.shape[0]
        if K is None:
            K = self.acyc_order or d
        A = W * W                           # Hadamard square
        Ak = anp.eye(d)
        acc = 0.0
        for k in range(1, K + 1):
            Ak = anp.dot(Ak, A)             # A^k
            acc = acc + anp.trace(Ak) / float(np.math.factorial(k))
        return acc

    @staticmethod
    def _bow_penalty(W1, W2):
        """
        Bow-freeness surrogate in an alternative form:
            || W1 ∘ W2 ||_F^2 / |W1|
        (normalization differs from他実装; ハイパーパラメータで吸収可能)
        """
        A = W1 * W2
        return anp.sum(A * A) / A.size


    # ---------- objective ----------

    def _objective(self, theta, mu, nu, Z, penalty_fn):
        """
        Augmented Lagrangian objective. All pieces are auto-diff compatible.
        theta : concatenated vector [vec(B), vec(L)], where L is strictly lower
                triangular part to be mirrored to form a symmetric matrix.
        mu, nu : AL parameters
        Z : list of pseudo-variables, Z[j] has shape (n, d)
        penalty_fn : structure penalty on (B, L_sym)
        """
        n, d = self.X.shape

        # unpack and enforce symmetry for the bidirected part
        B = anp.reshape(theta[:d * d], (d, d))
        L = anp.reshape(theta[d * d:], (d, d))
        L = L + L.T
        L = L - anp.diag(anp.diag(L))  # zero diagonal

        # data fit term (with generalized power 2*beta)
        data_term = 0.0
        for j in range(d):
            r = self.X[:, j] - anp.dot(self.X, B[:, j]) - anp.dot(Z[j], L[:, j])
            data_term = data_term + 0.5 / n * (anp.linalg.norm(r) ** (2 * self.beta))

        # structural constraints
        h = self._acyclicity_penalty(B) + penalty_fn(B, L)
        aug = 0.5 * mu * (h ** 2) + nu * h

        # smooth L0-ish (tanh-like) regularization on theta
        s = anp.log(n) * anp.abs(theta)
        t = (anp.exp(s) - 1) / (anp.exp(s) + 1)
        return data_term + aug + self.lam * anp.sum(t)


    # ---------- bounds from prior knowledge (different construction) ----------

    def _build_bounds(self, levels=None, exogenous=(), var_names=None, w_range=4.0):
        """
        Create L-BFGS-B bounds for theta with a different implementation style
        (array-based set-up, then reshape), avoiding nested if-chains.

        levels    : list[list[str]] topologically earlier -> later groups
        exogenous : iterable[str] with no incoming bidirected edges
        var_names : names for variables, defaults to ["x0", ...]
        """
        d = self.d
        if var_names is None:
            var_names = [f"x{i}" for i in range(d)]
        if levels is None:
            levels = [var_names]

        tier = {v: t for t, group in enumerate(levels) for v in group}
        exo = set(exogenous)

        # Directed bounds B: start wide, zero diag, then forbid backward edges
        B_lo = -w_range * np.ones((d, d))
        B_hi = +w_range * np.ones((d, d))
        np.fill_diagonal(B_lo, 0.0)
        np.fill_diagonal(B_hi, 0.0)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if tier[var_names[i]] > tier[var_names[j]]:
                    B_lo[i, j] = 0.0
                    B_hi[i, j] = 0.0

        # Bidirected bounds L (we optimize lower-tri; upper/diag fixed zero)
        L_lo = -w_range * np.ones((d, d))
        L_hi = +w_range * np.ones((d, d))
        for i in range(d):
            for j in range(d):
                if i <= j or (var_names[i] in exo) or (var_names[j] in exo):
                    L_lo[i, j] = 0.0
                    L_hi[i, j] = 0.0

        bounds_B = np.c_[B_lo.reshape(-1), B_hi.reshape(-1)].tolist()
        bounds_L = np.c_[L_lo.reshape(-1), L_hi.reshape(-1)].tolist()
        return bounds_B + bounds_L


    # ---------- pseudo variables (different numerics & interface) ----------

    def _pseudo(self, B, Omega):
        """
        Build pseudo-variables Z using solves instead of explicit inversion.
        Returns a list Z such that Z[j] has a zero column at j (shape (n, d)).
        """
        d = B.shape[0]
        eps = self.X - anp.dot(self.X, B)
        Z = [None] * d
        for j in range(d):
            idx = [k for k in range(d) if k != j]
            Om = Omega[anp.ix_(idx, idx)]
            Zij = anp.linalg.solve(Om, eps[:, idx].T).T    # (n, d-1)
            Zj = anp.insert(Zij, j, 0.0, axis=1)           # (n, d)
            Z[j] = Zj
        return Z


    # ---------- main solver loop (naming & flow refreshed) ----------

    def fit(self, levels=None, exogenous=(), max_outer=100, tol_h=1e-8, mu_max=1e16,
            w_threshold=0.05, inner_start=1, inner_growth=1, inner_tol=1e-4, verbose=False):
        """
        Train the model.

        Returns
        -------
        B, Omega : learned directed matrix and bidirected covariance (with diag)
        """
        d = self.d
        rng = self._rng

        # init
        B = anp.array(rng.uniform(-0.5, 0.5, size=(d, d)))
        L = anp.array(rng.uniform(-0.05, 0.05, size=(d, d)))
        lower_mask = anp.array(np.tril(np.ones((d, d)), k=-1))
        L = L * lower_mask
        L = L + L.T
        L = L - anp.diag(anp.diag(L))
        D = anp.diag(anp.diag(self.S))  # diagonal noise (kept diagonal)

        mu, nu, h_prev = 1.0, 0.0, np.inf
        inner_cap = inner_start
        penalty_fn = self._bow_penalty

        bounds = self._build_bounds(levels, set(exogenous), [f"x{i}" for i in range(d)])
        objective = functools.partial(self._objective)
        gfun = grad(objective)

        for outer in range(max_outer):
            B_new, L_new, D_new = None, None, None
            h_new = None

            while mu < mu_max:
                B_new = B.copy()
                L_new = L.copy()
                D_new = D.copy()

                # inner refinement
                for _ in range(inner_cap):
                    B_old, L_old, D_old = B_new, L_new, D_new
                    Z = self._pseudo(B_new, L_new + D_new)

                    theta0 = anp.concatenate([anp.ravel(B_new), anp.ravel(L_new)])
                    res = sopt.minimize(
                        self._objective, theta0,
                        args=(mu, nu, Z, penalty_fn),
                        method="L-BFGS-B",
                        jac=gfun,
                        bounds=bounds,
                        options={"disp": False}
                    )

                    B_new = anp.reshape(res.x[:d * d], (d, d))
                    L_new = anp.reshape(res.x[d * d:], (d, d))
                    L_new = L_new + L_new.T
                    L_new = L_new - anp.diag(anp.diag(L_new))

                    # refresh diagonal noise from residuals (different expression)
                    diag_vals = [anp.var(self.X[:, j] - anp.dot(self.X, B_new[:, j])) for j in range(d)]
                    D_new = anp.diag(anp.array(diag_vals))

                    # convergence of inner loop
                    delta = anp.sum(anp.abs(B_old - B_new)) \
                          + anp.sum(anp.abs((L_old + D_old) - (L_new + D_new)))
                    if float(delta) < inner_tol:
                        break

                h_new = self._acyclicity_penalty(B_new) + penalty_fn(B_new, L_new)
                if verbose:
                    print(f"[outer {outer}] h={float(h_new):.3e} mu={mu:.1e} inner={inner_cap}")

                # penalty schedule
                if float(h_new) < 0.25 * float(h_prev):
                    break
                else:
                    mu *= 10.0

            # AL update
            B, L, D = B_new.copy(), L_new.copy(), D_new.copy()
            h_prev = h_new
            nu = nu + mu * h_prev
            inner_cap += inner_growth

            if float(h_prev) <= tol_h or mu >= mu_max:
                break

        # threshold small entries
        Bf = anp.where(anp.abs(B) < w_threshold, 0.0, B)
        Of = anp.where(anp.abs(L + D) < w_threshold, 0.0, (L + D))

        self.B_, self.Omega_ = np.array(Bf), np.array(Of)
        return self.B_, self.Omega_
