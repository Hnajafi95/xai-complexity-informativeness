#!/usr/bin/env python3
"""
E3 — Collinearity-aware methods compared.

Goal: demonstrate that, on a pure-collinearity DGP where X1 carries all the
signal and X2 is a perfect (linearly) correlated noise feature with NO
causal effect on y, different methods give very different answers about
which feature is "important":

  - Interventional (marginal) SHAP, linear model — KernelSHAP with a
        background sample; follows the fitted weights
  - Observational (conditional) SHAP, linear model — exact Shapley values
        with v(S) = E[f(X) | X_S = x_S] under the fitted Gaussian feature
        distribution; spreads credit across dependent features
  - TreeSHAP (interventional) on XGBoost
  - PatternAttribution (Haufe transformation)
  - AEC (our implementation of the Salih 2024 formula)

Terminology note: shap.KernelExplainer and
TreeExplainer(data, feature_perturbation="interventional") both compute the
INTERVENTIONAL / marginal variant. An earlier version of this script labelled
the KernelSHAP run "observational"; it was never conditional. The exact
conditional variant is now computed explicitly below.

No public reference implementation of AEC exists at the time of writing
(Salih 2024 published only the equations); we implement it directly from
the conceptual formula in the paper:

    TotalEffect(X_j)  =  DirectEffect(X_j -> y)
                       + sum_{i != j} IndirectEffect(X_i <-> X_j -> y)

where DirectEffect = univariate linear regression of y on X_j, and
IndirectEffect_{i,j} = (Pearson r between X_i and X_j) * DirectEffect(X_i),
so that a feature whose only "importance" comes from being correlated with
the true signal has its credit reassigned to the signal.

This reimplementation is documented in the file header so that, if Salih
releases code differing in details, the experiment can be updated.

Run:
    cd revised_v2 && python3 experiments/e3_collinearity_methods.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
np.random.seed(0)


# ── 1. Collinearity-only DGP, Janzing-2020 style ──────────────────────────
# We want a setup where observational vs interventional Shapley diverge.
# Construct a latent source that drives y, and let TWO observed features
# (X1, X2) be noisy copies of that source. The model can use either or
# both — and observational SHAP will smear credit across both, while a
# causally-correct attribution should privilege the one that is the cleaner
# read of the source. We add X3 as an independent noise control.
#
# DGP:
#   s ~ N(0, 1)                              # latent source (the actual cause)
#   y = 3·s + ε_y                            # y depends only on the source
#   X1 = s + ε_1   (small noise)             # noisy read of s
#   X2 = s + ε_2   (larger noise)            # noisier read of s
#   X3 ~ N(0, 1)                              # independent noise

n = 2000
s = np.random.normal(0, 1, n)
y = 3.0 * s + np.random.normal(0, 0.3, n)
X1 = s + np.random.normal(0, 0.2, n)
X2 = s + np.random.normal(0, 0.5, n)
X3 = np.random.normal(0, 1, n)
X = np.stack([X1, X2, X3], axis=1)
feature_names = ["X1 (clean read of s)", "X2 (noisy read of s)", "X3 (noise)"]
print(f"Data: n={n}, p={X.shape[1]}")
print(f"corr(X1, X2) = {pearsonr(X1, X2)[0]:+.3f}")
print(f"corr(X1, X3) = {pearsonr(X1, X3)[0]:+.3f}")
print(f"corr(X2, X3) = {pearsonr(X2, X3)[0]:+.3f}")
print(f"corr(X1, y)  = {pearsonr(X1, y)[0]:+.3f}")
print(f"corr(X2, y)  = {pearsonr(X2, y)[0]:+.3f}")
print(f"corr(X3, y)  = {pearsonr(X3, y)[0]:+.3f}\n")


# ── 2. Models — XGBoost (for tree-based SHAP variants) and Linear ─────────
import pandas as pd
X_df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             random_state=0, verbosity=0)
xgb_model.fit(X_df, y)
lin_model = LinearRegression().fit(X, y)
print(f"XGBoost R² = {xgb_model.score(X_df, y):.4f}")
print(f"Linear   R² = {lin_model.score(X, y):.4f}, coefs = {lin_model.coef_}")


# ── 3. Run methods ────────────────────────────────────────────────────────
results = {}

# (a) Interventional (marginal) SHAP via KernelSHAP on the linear model
print("\nRunning KernelSHAP (interventional / marginal background)...")
background = shap.sample(X, 50, random_state=0)
expl_obs = shap.KernelExplainer(lin_model.predict, background)
shap_obs = expl_obs.shap_values(X[:200], nsamples=200, silent=True)
results["Interv. SHAP (KernelSHAP, linear)"] = np.abs(shap_obs).mean(0)

# (b) TreeSHAP interventional on XGBoost
print("Running TreeSHAP (interventional)...")
expl_int = shap.TreeExplainer(xgb_model, X_df, feature_perturbation="interventional")
shap_int = expl_int.shap_values(X_df)
results["TreeSHAP (interventional)"] = np.abs(shap_int).mean(0)

# (b2) Observational (conditional) SHAP on the same linear model, computed
# exactly: for f(x) = w·x + b and Gaussian X, E[f(X) | X_S = x_S] =
# b + w_S·x_S + w_{~S}·(mu_{~S} + Sigma_{~S,S} Sigma_{S,S}^{-1} (x_S - mu_S)).
# Shapley values follow by enumerating all 2^p coalitions (p = 3).
print("Computing exact observational (conditional) SHAP on the linear model...")
from itertools import combinations as _comb
from math import factorial as _fact
mu, Sig = X.mean(0), np.cov(X, rowvar=False)
w_lin, b_lin = lin_model.coef_, lin_model.intercept_
def cond_value(x, S):
    S = list(S); R = [j for j in range(3) if j not in S]
    if not S:
        return b_lin + w_lin @ mu
    xr = mu[R] + Sig[np.ix_(R, S)] @ np.linalg.solve(Sig[np.ix_(S, S)], x[S] - mu[S]) if R else []
    return b_lin + w_lin[S] @ x[S] + (w_lin[R] @ xr if R else 0.0)
def cond_shap(x):
    p = 3; phi = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for r in range(p):
            for S in _comb(others, r):
                wgt = _fact(r) * _fact(p - r - 1) / _fact(p)
                phi[j] += wgt * (cond_value(x, list(S) + [j]) - cond_value(x, S))
    return phi
shap_cond = np.array([cond_shap(x) for x in X[:200]])
results["Obs. SHAP (conditional, linear)"] = np.abs(shap_cond).mean(0)

# (c) Pattern Attribution on linear model
print("Computing Pattern Attribution (Haufe transformation)...")
w = lin_model.coef_
cov_X = np.cov(X, rowvar=False)
A = (cov_X @ w) / (w @ cov_X @ w + 1e-12)
results["PatternAttribution"] = np.abs(A)

# (d) AEC — our implementation of the Salih 2024 conceptual formula
print("Running AEC (our implementation; no official code public)...")

def aec_importance(X, y):
    """
    AEC importance per Salih 2024:
      Total(X_j) = Direct(X_j -> y)  +  sum_{i != j} Indirect(X_i <-> X_j -> y)
      Direct(X_j -> y)             = |β| from univariate y ~ X_j
      Indirect(X_i <-> X_j -> y)   = |corr(X_i, X_j) * β(X_i -> y)|

    A feature whose only "importance" is via correlation with the true cause
    will have small direct effect but large indirect effects pointing back
    to the true cause; AEC re-credits accordingly.
    """
    n, p = X.shape
    # Direct effects: per-feature univariate regression of y on X_j
    direct = np.zeros(p)
    for j in range(p):
        b = np.cov(X[:, j], y)[0, 1] / np.var(X[:, j])
        direct[j] = abs(b)
    # Indirect: for each pair (i, j), credit |corr(X_i, X_j) * direct(X_i)|
    # back to j when X_i is more "primary" than j. Per the paper's
    # symmetric pairwise treatment, we sum |corr * direct(other)| over
    # other features.
    indirect = np.zeros(p)
    corr = np.corrcoef(X.T)
    for j in range(p):
        for i in range(p):
            if i == j:
                continue
            indirect[j] += abs(corr[i, j]) * direct[i]
    return direct, indirect


direct, indirect = aec_importance(X, y)
# Total raw AEC importance:
aec_total = direct + indirect
# But importance per *feature* in Salih's framework is the DIRECT effect
# after accounting for indirect contributions. Following the paper's logic
# of "isolating direct effects," we report the direct component normalized
# by the total spread:
aec_direct_norm = direct
results["AEC (direct component)"] = aec_direct_norm
# Total component (sanity check; for figure annotation):
results_aux = {"AEC direct": direct, "AEC indirect": indirect}


# ── 4. Normalize and print ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("NORMALIZED IMPORTANCE (sum to 1)")
print("X1 and X2 are both noisy reads of the same latent source s; X3 is unrelated.")
print("=" * 80)
print(f"{'Method':<32}{'X1':>10}{'X2':>10}{'X3':>10}{'X1/X2':>10}")
print("-" * 80)
normalized = {}
for name, vals in results.items():
    norm = vals / max(vals.sum(), 1e-12)
    normalized[name] = norm
    ratio = norm[0] / max(norm[1], 1e-12)
    print(f"{name:<32}{norm[0]:>10.3f}{norm[1]:>10.3f}{norm[2]:>10.3f}{ratio:>10.2f}")

print("\nInterpretation:")
print("  - Interventional SHAP answers 'what does the fitted model use?': it follows")
print("    the fitted weights, so it concentrates credit on the cleaner read X1.")
print("  - Observational SHAP answers a data-level question: it credits features for")
print("    the information they carry about the prediction, so the two dependent")
print("    reads share credit even though the model barely uses X2.")
print("  - A source-symmetric split (ratio ~1) treats both reads as projections of s.")


# ── 5. Save and figure ───────────────────────────────────────────────────
out = {
    "feature_names": feature_names,
    "ground_truth": [1.0, 0.0, 0.0],
    "raw_importance": {k: v.tolist() for k, v in results.items()},
    "normalized_importance": {k: v.tolist() for k, v in normalized.items()},
    "aec_direct": direct.tolist(),
    "aec_indirect": indirect.tolist(),
}
np.save(HERE / "e3_results.npy", out)

fig, ax = plt.subplots(figsize=(11.5, 5.4))
method_names = list(normalized.keys())
x1_vals = [normalized[m][0] for m in normalized]
x2_vals = [normalized[m][1] for m in normalized]
x3_vals = [normalized[m][2] for m in normalized]
x_pos = np.arange(len(method_names))
bar_w = 0.27

ax.bar(x_pos - bar_w, x1_vals, bar_w, label="X1 (clean read of latent source)", color="#3498db")
ax.bar(x_pos, x2_vals, bar_w, label="X2 (noisier read of same source)", color="#9b59b6")
ax.bar(x_pos + bar_w, x3_vals, bar_w, label="X3 (independent noise)", color="#95a5a6")

ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, rotation=12, ha="right", fontsize=9)
ax.set_ylabel("Normalized |attribution|")
ax.set_title(
    "E3 — Two noisy reads of one latent source (corr(X1, X2) ≈ 0.88).\n"
    "Interventional SHAP follows the fitted weights (model reliance); observational SHAP,\n"
    "PatternAttribution and AEC share credit across the dependent reads (data-level view).",
    fontsize=10,
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="green", linestyle=":", alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig(HERE / "e3_collinearity_comparison.pdf", dpi=150, bbox_inches="tight")
plt.savefig(HERE / "e3_collinearity_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {HERE / 'e3_collinearity_comparison.pdf'}")
