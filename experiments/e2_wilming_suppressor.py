#!/usr/bin/env python3
"""
E2 — Wilming et al. (2021, 2023) suppressor benchmark, reproduced.

Reference: braindatalab/scrutinizing-xai (Wilming et al. 2021), and the
analytical setup in Wilming et al. 2023 (Theoretical behavior of XAI methods
in the presence of suppressor variables).

Setup: a 2-feature linear ground-truth model where X1 carries the signal
and X2 is a suppressor variable. X2 is correlated with X1 (because it
shares the noise structure) but has *no direct effect* on Y. A linear
classifier trained on (X1, X2) assigns a non-zero weight to X2 because
including it lets the model denoise X1's signal. The XAI question is
whether attribution methods correctly say "X2 is uninformative."

Methods compared:
  - KernelSHAP            (additive, observational; SHAP package)
  - LIME                  (local linear surrogate)
  - Permutation importance (PFI)
  - Gradient * Input      (raw saliency for linear models)
  - PatternAttribution    (Haufe transformation A = Σ_X · w, implemented
                           directly from the equation in Haufe et al. 2014;
                           innvestigate provides the same on top of NNs)

Expected outcome (from the paper): the first four methods all assign non-
zero importance to the suppressor X2; only PatternAttribution correctly
attributes zero importance to X2.

Run:
    cd revised_v2 && python3 experiments/e2_wilming_suppressor.py
    cd revised_v2 && python3 experiments/e2_wilming_suppressor.py --plot-only
(--plot-only redraws e2_suppressor_comparison.{pdf,png} from the saved
e2_results.npy without rerunning the experiment.)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_style import apply_style, fig_size

HERE = Path(__file__).resolve().parent


def make_figure(results):
    """Draw e2_suppressor_comparison.{pdf,png} from the saved e2_results.npy dict."""
    apply_style()
    normalized = results["normalized_importance"]
    fig, ax = plt.subplots(figsize=fig_size(0.62, 0.50))
    method_names = list(normalized.keys())
    x1_vals = [normalized[m][0] for m in method_names]
    x2_vals = [normalized[m][1] for m in method_names]
    x_pos = np.arange(len(method_names))
    bar_w = 0.38

    ax.bar(x_pos - bar_w / 2, x1_vals, bar_w, label="X1 (signal)", color="#3498db")
    ax.bar(x_pos + bar_w / 2, x2_vals, bar_w, label="X2 (suppressor)", color="#e74c3c")
    # X2's data-generating relevance is 0; that reference is stated in the paper caption.
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Normalized |attribution|")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, len(method_names) - 0.45)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Annotate the failure pattern
    for i, (m, v2) in enumerate(zip(method_names, x2_vals)):
        if v2 > 0.1:
            ax.annotate(f"{v2:.2f}", (i + bar_w / 2 + 0.03, v2 + 0.02), ha="center", fontsize=7, color="#c0392b")
    # Note on PatternAttribution, placed at the side below the legend
    pa = len(method_names) - 1
    ax.annotate(
        "Haufe pattern\n(PatternAttribution)\nattributes ~0 to the\nsuppressor: the\ndata-generating answer.",
        xy=(pa + bar_w / 2, x2_vals[-1] + 0.01), xycoords="data",
        xytext=(1.02, 0.08), textcoords="axes fraction",
        fontsize=7, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", alpha=0.95, linewidth=0.6),
        arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.8, relpos=(0.0, 0.3)),
    )

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"e2_suppressor_comparison.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figure saved to {HERE / 'e2_suppressor_comparison.pdf'}")


_ap = argparse.ArgumentParser()
_ap.add_argument("--plot-only", action="store_true",
                 help="redraw the figure from saved e2_results.npy; do not rerun the experiment")
if _ap.parse_args().plot_only:
    make_figure(np.load(HERE / "e2_results.npy", allow_pickle=True).item())
    raise SystemExit(0)

# Experiment-only dependencies (not needed for --plot-only)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular

np.random.seed(0)


# ── 1. Data-generating process (Wilming 2D linear setup) ──────────────────
# Two classes; the signal feature X_signal separates them on the y-axis of
# its native scale. A suppressor X_supp is generated to be correlated with
# X_signal but to carry NO information about the class label conditional on
# X_signal. In Wilming's terminology, X_supp is part of the "distractor"
# pattern that the classifier must subtract out to clean the signal.

n_per_class = 500
n_total = 2 * n_per_class

# Ground-truth source variable (signal): determines class
s = np.concatenate([
    np.random.normal(loc=-1.0, scale=1.0, size=n_per_class),  # class 0
    np.random.normal(loc=+1.0, scale=1.0, size=n_per_class),  # class 1
])
y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

# Suppressor source (distractor): zero-mean noise that is NOT a function of y
d = np.random.normal(loc=0.0, scale=1.0, size=n_total)

# Observed features: a 2×2 mixing matrix combines (s, d) into (X1, X2).
# X1 receives the signal but also picks up some of the distractor.
# X2 receives only the distractor (it is informationally null about y).
# The classifier can recover signal by computing X1 - X2.
X1 = s + 0.9 * d
X2 = d  # pure suppressor — independent of y given X1
X = np.stack([X1, X2], axis=1)

# Sanity: zero-order vs partial correlation of X2 with y
from scipy.stats import pearsonr
r_zero = pearsonr(X2, y)[0]
# Partial correlation of X2 with y given X1
res_x2 = X2 - np.linalg.lstsq(X1.reshape(-1, 1), X2, rcond=None)[0][0] * X1
res_y = y - np.linalg.lstsq(X1.reshape(-1, 1), y.astype(float), rcond=None)[0][0] * X1
r_partial = pearsonr(res_x2, res_y)[0]
print(f"Data: n={n_total}, classes=[{(y==0).sum()}, {(y==1).sum()}]")
print(f"corr(X1, y)  = {pearsonr(X1, y)[0]:+.3f}")
print(f"corr(X2, y)  = {r_zero:+.3f}        (zero-order; small but non-zero from noise)")
print(f"corr(X2, y | X1) = {r_partial:+.3f}    (partial; this is what matters)")


# ── 2. Train a linear classifier ──────────────────────────────────────────
clf = LogisticRegression(C=1e6).fit(X, y)
w = clf.coef_.flatten()
b = clf.intercept_[0]
acc = clf.score(X, y)
print(f"\nLinear classifier: weights w = {w}, accuracy = {acc:.3f}")
print(f"Sign pattern: w[X1]={w[0]:+.3f}, w[X2]={w[1]:+.3f}")
print("Both weights are nonzero — the classifier uses X2 to denoise X1.\n")


# ── 3. Apply XAI methods ──────────────────────────────────────────────────
# We report absolute importance per feature, averaged across all training
# samples. For binary classifiers, we explain the positive class.

methods = {}

# --- KernelSHAP (observational, additive)
print("Running KernelSHAP...")
background = shap.sample(X, 50, random_state=0)
explainer_shap = shap.KernelExplainer(lambda Z: clf.predict_proba(Z)[:, 1], background)
shap_vals = explainer_shap.shap_values(X[:200], nsamples=100, silent=True)
methods["KernelSHAP"] = np.abs(shap_vals).mean(0)

# --- LIME
print("Running LIME...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X, feature_names=["X1", "X2"], discretize_continuous=False, class_names=["0", "1"]
)
lime_imp = np.zeros(2)
n_lime = 100
for i in range(n_lime):
    exp = lime_explainer.explain_instance(
        X[i], lambda Z: clf.predict_proba(Z), num_features=2, num_samples=200
    )
    coefs = dict(exp.as_map()[1])
    lime_imp += np.abs([coefs.get(0, 0.0), coefs.get(1, 0.0)])
methods["LIME"] = lime_imp / n_lime

# --- Permutation Feature Importance
print("Running PFI...")
pfi = permutation_importance(clf, X, y, n_repeats=20, random_state=0)
methods["PFI"] = np.abs(pfi.importances_mean)

# --- Gradient × Input (for a linear model: w * x, averaged over instances)
# Per-sample gradient = w; we report |w_j * E[|X_j|]| as a magnitude
# that respects the standard Gradient×Input convention.
methods["GradientxInput"] = np.abs(w) * np.abs(X).mean(0)

# --- Pattern Attribution (Haufe 2014 forward transformation)
# A = Σ_X · w / (w' Σ_X w)
# Σ_X is the data covariance matrix; for a well-specified linear model on
# the underlying source (s), the Haufe transformation recovers a pattern
# that is zero in the components that have no causal effect on y.
cov_X = np.cov(X, rowvar=False)
A = (cov_X @ w) / (w @ cov_X @ w)
methods["PatternAttribution"] = np.abs(A)

# Print
print("\n" + "=" * 60)
print("ATTRIBUTION RESULTS (raw |importance| per feature)")
print("=" * 60)
print(f"{'Method':<22}{'X1 (signal)':>15}{'X2 (suppressor)':>20}")
print("-" * 60)
for name, vals in methods.items():
    print(f"{name:<22}{vals[0]:>15.4f}{vals[1]:>20.4f}")

# Normalized
print("\n" + "=" * 60)
print("NORMALIZED IMPORTANCE (sum to 1)")
print("=" * 60)
print(f"{'Method':<22}{'X1 (signal)':>15}{'X2 (suppressor)':>20}{'X2 ratio':>14}")
print("-" * 60)
normalized = {}
for name, vals in methods.items():
    norm = vals / max(vals.sum(), 1e-12)
    normalized[name] = norm
    print(f"{name:<22}{norm[0]:>15.3f}{norm[1]:>20.3f}{norm[1]:>14.3f}")


# ── 4. Save and figure ───────────────────────────────────────────────────
results = {
    "ground_truth": {"X1": "signal", "X2": "suppressor (uninformative)"},
    "classifier_weights": {"X1": float(w[0]), "X2": float(w[1])},
    "accuracy": float(acc),
    "r_zero_order_X2_y": float(r_zero),
    "r_partial_X2_y_given_X1": float(r_partial),
    "raw_importance": {k: v.tolist() for k, v in methods.items()},
    "normalized_importance": {k: v.tolist() for k, v in normalized.items()},
}
np.save(HERE / "e2_results.npy", results)
make_figure(results)
