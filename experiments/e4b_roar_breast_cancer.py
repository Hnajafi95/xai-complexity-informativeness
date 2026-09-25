#!/usr/bin/env python3
"""
E4b — ROAR (Remove And Retrain) faithfulness benchmark on a SECOND real
dataset with documented collinearity.

Motivation: E4 runs ROAR on UCI Adult. Evidence from synthetic data plus one
real dataset is thin, and interaction-/collinearity-aware methods are largely
untested on real high-dimensional collinear tabular data. E4b hardens the
empirical claim on a second, deliberately collinear real dataset.

Dataset: Breast Cancer Wisconsin (Diagnostic), ships with scikit-learn
(no download required). n = 569, p = 30 features. The feature set is
famously collinear: each of ten base measurements (radius, texture,
perimeter, area, ...) appears three times as a "mean", "standard error",
and "worst" column, and several measurements are geometrically dependent
(area ~ radius^2, perimeter ~ radius). This makes it an ideal real-data
stress test for the collinearity story in Section 2 and the ROAR-redundancy
caveat surfaced in E4.

Methods compared (same as E4 for comparability):
  - KernelSHAP (interventional: marginal background sample)
  - TreeSHAP (interventional)
  - Permutation Feature Importance
  - Random ablation (baseline)

Protocol: rank features by global importance, remove the top-k, retrain
XGBoost from scratch on the reduced set, measure test accuracy. A faithful
method's top-k should produce a steeper drop than random removal.

Run:
    cd revised_v2 && python3 experiments/e4b_roar_breast_cancer.py
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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
np.random.seed(0)


# ── 1. Load Breast Cancer Wisconsin ───────────────────────────────────────
print("Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
data = load_breast_cancer(as_frame=True)
X_df = data.data.copy()
# target: sklearn encodes malignant=0, benign=1. We predict malignant (1).
y = (data.target == 0).astype(int).values
features = list(X_df.columns)
print(f"  shape={X_df.shape}, n_features={len(features)}")
print(f"  class balance (malignant): {y.mean():.3f}")


# ── 1b. Document the collinearity (this is the whole point of E4b) ────────
corr = X_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_pairs = [
    (i, j, upper.loc[i, j])
    for i in upper.index
    for j in upper.columns
    if pd.notna(upper.loc[i, j]) and upper.loc[i, j] > 0.9
]
high_pairs.sort(key=lambda t: -t[2])
print(f"\nCollinearity diagnostic: {len(high_pairs)} feature pairs with |r| > 0.9 "
      f"(out of {len(features) * (len(features) - 1) // 2} pairs).")
print("  Strongest collinear pairs:")
for i, j, r in high_pairs[:6]:
    print(f"    {r:.3f}  {i}  <->  {j}")

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42, stratify=y
)


# ── 2. Baseline model ─────────────────────────────────────────────────────
def fit_model(X_tr, y_tr):
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=0, n_jobs=4, verbosity=0,
        eval_metric="logloss", use_label_encoder=False,
        subsample=0.8, colsample_bytree=0.8,
    )
    m.fit(X_tr, y_tr)
    return m


print("\nTraining baseline model on full feature set...")
baseline_model = fit_model(X_train, y_train)
baseline_acc = baseline_model.score(X_test, y_test)
print(f"  baseline test accuracy = {baseline_acc:.4f}")


# ── 3. Compute global feature importance per method ──────────────────────
importance = {}

print("Computing PFI...")
pfi = permutation_importance(baseline_model, X_test, y_test, n_repeats=10,
                             random_state=0, n_jobs=4)
importance["PFI"] = dict(zip(features, np.abs(pfi.importances_mean)))

print("Computing TreeSHAP (interventional)...")
expl_tree = shap.TreeExplainer(baseline_model, X_train,
                               feature_perturbation="interventional")
shap_tree = expl_tree.shap_values(X_test)
importance["TreeSHAP_interventional"] = dict(zip(features, np.abs(shap_tree).mean(0)))

print("Computing KernelSHAP (interventional/marginal background, subsampled)...")
background = shap.kmeans(X_train, 30)
expl_kern = shap.KernelExplainer(
    lambda Z: baseline_model.predict_proba(pd.DataFrame(Z, columns=features))[:, 1],
    background,
)
shap_kern = expl_kern.shap_values(X_test.iloc[:80], nsamples=120, silent=True)
importance["KernelSHAP_interventional"] = dict(zip(features, np.abs(shap_kern).mean(0)))

rng = np.random.RandomState(0)
importance["Random"] = {f: rng.uniform(0, 1) for f in features}

print("\nTop-5 features per method:")
for name, imp in importance.items():
    top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
    print(f"  {name}: {[f for f, _ in top5]}")


# ── 4. ROAR: remove top-k, retrain, measure accuracy ─────────────────────
k_values = [1, 2, 3, 5, 8, 12]
roar_results = {}

print("\nRunning ROAR (remove top-k features, retrain)...")
for method_name, imp in importance.items():
    sorted_features = sorted(features, key=lambda f: -imp[f])
    accs = [baseline_acc]
    for k in k_values:
        removed = sorted_features[:k]
        kept = [f for f in features if f not in removed]
        m = fit_model(X_train[kept], y_train)
        acc = m.score(X_test[kept], y_test)
        accs.append(acc)
        print(f"  {method_name:<28} k={k:<2} retrained acc={acc:.4f}")
    roar_results[method_name] = accs


# ── 5. Save & figure ─────────────────────────────────────────────────────
out = {
    "features": features,
    "baseline_acc": float(baseline_acc),
    "k_values": [0] + k_values,
    "n_high_corr_pairs": len(high_pairs),
    "strongest_pairs": [(i, j, float(r)) for i, j, r in high_pairs[:6]],
    "importance": {k: list(v.items()) for k, v in importance.items()},
    "roar_curves": {k: list(map(float, v)) for k, v in roar_results.items()},
}
np.save(HERE / "e4b_results.npy", out)

fig, ax = plt.subplots(figsize=(9, 5.4))
x_axis = [0] + k_values

method_colors = {
    "KernelSHAP_interventional": "#e74c3c",
    "TreeSHAP_interventional": "#27ae60",
    "PFI": "#f39c12",
    "Random": "#7f8c8d",
}
labels = {
    "KernelSHAP_interventional": "KernelSHAP (interventional)",
    "TreeSHAP_interventional": "TreeSHAP (interventional)",
    "PFI": "Permutation Importance",
    "Random": "Random ablation (baseline)",
}

for method, curve in roar_results.items():
    ax.plot(
        x_axis, curve, marker="o",
        label=labels.get(method, method),
        color=method_colors.get(method, "black"),
        linewidth=2 if method != "Random" else 1.5,
        linestyle="--" if method == "Random" else "-",
    )

ax.axhline(baseline_acc, color="black", linestyle=":", linewidth=1, alpha=0.5,
           label=f"Baseline ({baseline_acc:.3f})")
ax.set_xlabel("Number of top-k features removed before retraining")
ax.set_ylabel("Test accuracy")
ax.set_title(
    "E4b — ROAR on Breast Cancer Wisconsin (30 features, highly collinear)\n"
    f"{len(high_pairs)} feature pairs with |r| > 0.9; collinearity blunts the ROAR drop.",
    fontsize=10.5,
)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3, linestyle="--")
ax.set_xticks(x_axis)

plt.tight_layout()
plt.savefig(HERE / "e4b_roar_curves.pdf", dpi=150, bbox_inches="tight")
plt.savefig(HERE / "e4b_roar_curves.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {HERE / 'e4b_roar_curves.pdf'}")
