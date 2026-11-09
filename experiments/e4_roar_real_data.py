#!/usr/bin/env python3
"""
E4 — ROAR (Remove And Retrain) faithfulness benchmark on real tabular data.

Reference: Hooker, Erhan, Kindermans, Kim (NeurIPS 2019), "A Benchmark for
Interpretability Methods in Deep Neural Networks". The ROAR protocol asks:
if we remove the top-k features identified by an attribution method and
retrain the model, how much does accuracy drop? A *faithful* method's top-k
features should cause a steeper accuracy drop than random removal.

Dataset: UCI Adult (income prediction). Standard tabular benchmark with a
documented mix of collinearity (e.g., education vs education_num,
relationship vs marital_status) and interaction structure.

Methods compared:
  - KernelSHAP (interventional: marginal background sample)
  - TreeSHAP (interventional)
  - Permutation Feature Importance
  - Random ablation (baseline)

For each method we compute global feature importance, then for k = 1..5 we
retrain XGBoost after removing the top-k features and measure test accuracy.

Run:
    cd revised_v2 && python3 experiments/e4_roar_real_data.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import apply_style, fig_size

# Printed at 0.5 x \textwidth; identical in E4 and E4b so the companion figures match.
FIG_FRAC, FIG_ASPECT = 0.54, 0.6  # axes canvas; the side legend adds width (PDF ~0.74 x \textwidth)


def make_figure(res, outdir=HERE):
    """ROAR curves (test accuracy vs. # top-k features removed and retrained).

    Uses only the saved results dict (e4_results.npy). Shared layout with the
    companion E4/E4b script so the two ROAR figures match."""
    x_axis = list(res["k_values"])
    baseline_acc = res["baseline_acc"]
    roar_results = res["roar_curves"]

    method_colors = {
        "KernelSHAP_interventional": "#e74c3c",
        "TreeSHAP_interventional": "#27ae60",
        "PFI": "#f39c12",
        "Random": "#7f8c8d",
    }
    # Short labels keep the side legend narrow.
    labels = {
        "KernelSHAP_interventional": "KernelSHAP (interv.)",
        "TreeSHAP_interventional": "TreeSHAP (interv.)",
        "PFI": "Permutation importance",
        "Random": "Random ablation",
    }

    apply_style()
    fig, ax = plt.subplots(figsize=fig_size(FIG_FRAC, FIG_ASPECT))
    for method, curve in roar_results.items():
        ax.plot(
            x_axis, curve, marker="o",
            label=labels.get(method, method),
            color=method_colors.get(method, "black"),
            linewidth=1.3 if method != "Random" else 1.0,
            linestyle="--" if method == "Random" else "-",
        )
    ax.axhline(baseline_acc, color="black", linestyle=":", linewidth=0.7, alpha=0.5,
               label=f"Baseline ({baseline_acc:.3f})")
    ax.set_xlabel("Top-k features removed before retraining")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xticks(x_axis)
    # Legend to the right of the axes: wide, short figure; keys never cover the curves.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, borderaxespad=0.0)
    fig.savefig(Path(outdir) / "e4_roar_curves.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(outdir) / "e4_roar_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figure saved to {Path(outdir) / 'e4_roar_curves.pdf'}")


def run_experiment():
    import pandas as pd
    import shap
    import xgboost as xgb
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance

    np.random.seed(0)


    # ── 1. Load UCI Adult ─────────────────────────────────────────────────────
    print("Loading UCI Adult dataset...")
    data = fetch_openml("adult", version=2, as_frame=True)
    X_df, y_ser = data.data, data.target
    # Encode categoricals as integer codes (XGBoost can handle this; for ROAR
    # we just need feature columns that can be dropped)
    X_df = X_df.copy()
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    for c in cat_cols:
        X_df[c] = X_df[c].astype("category").cat.codes
    y = (y_ser == ">50K").astype(int).values
    features = list(X_df.columns)
    print(f"  shape={X_df.shape}, features={features}")
    print(f"  class balance: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )


    # ── 2. Baseline model ─────────────────────────────────────────────────────
    def fit_model(X_tr, y_tr):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=0, n_jobs=4, verbosity=0,
            eval_metric="logloss", use_label_encoder=False,
        )
        m.fit(X_tr, y_tr)
        return m


    print("\nTraining baseline model on full feature set...")
    baseline_model = fit_model(X_train, y_train)
    baseline_acc = baseline_model.score(X_test, y_test)
    print(f"  baseline test accuracy = {baseline_acc:.4f}")


    # ── 3. Compute global feature importance per method ──────────────────────
    importance = {}

    # (a) Permutation importance
    print("Computing PFI...")
    pfi = permutation_importance(baseline_model, X_test, y_test, n_repeats=10,
                                 random_state=0, n_jobs=4)
    importance["PFI"] = dict(zip(features, np.abs(pfi.importances_mean)))

    # (b) TreeSHAP (interventional)
    print("Computing TreeSHAP (interventional)...")
    expl_tree = shap.TreeExplainer(baseline_model, X_train, feature_perturbation="interventional")
    shap_tree = expl_tree.shap_values(X_test.iloc[:500])
    importance["TreeSHAP_interventional"] = dict(zip(features, np.abs(shap_tree).mean(0)))

    # (c) KernelSHAP (interventional; marginal background) — slow, so subsample
    print("Computing KernelSHAP (interventional/marginal background, subsampled)...")
    background = shap.kmeans(X_train, 30)
    expl_kern = shap.KernelExplainer(
        lambda Z: baseline_model.predict_proba(pd.DataFrame(Z, columns=features))[:, 1],
        background,
    )
    shap_kern = expl_kern.shap_values(X_test.iloc[:80], nsamples=80, silent=True)
    importance["KernelSHAP_interventional"] = dict(zip(features, np.abs(shap_kern).mean(0)))

    # (d) Random baseline (permutation of feature order)
    rng = np.random.RandomState(0)
    importance["Random"] = {f: rng.uniform(0, 1) for f in features}

    # Print top-5 per method
    print("\nTop-5 features per method:")
    for name, imp in importance.items():
        top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
        print(f"  {name}: {[f for f, _ in top5]}")


    # ── 4. ROAR: remove top-k, retrain, measure accuracy ─────────────────────
    k_values = [1, 2, 3, 5, 8]
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
            print(f"  {method_name:<28} k={k}  removed={removed[:3]}...  retrained acc={acc:.4f}")
        roar_results[method_name] = accs


    # ── 5. Save & figure ─────────────────────────────────────────────────────
    out = {
        "features": features,
        "baseline_acc": float(baseline_acc),
        "k_values": [0] + k_values,
        "importance": {k: list(v.items()) for k, v in importance.items()},
        "roar_curves": {k: list(map(float, v)) for k, v in roar_results.items()},
    }
    np.save(HERE / "e4_results.npy", out)
    make_figure(np.load(HERE / "e4_results.npy", allow_pickle=True).item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from e4_results.npy without rerunning")
    args = ap.parse_args()
    if args.plot_only:
        make_figure(np.load(HERE / "e4_results.npy", allow_pickle=True).item())
    else:
        run_experiment()
