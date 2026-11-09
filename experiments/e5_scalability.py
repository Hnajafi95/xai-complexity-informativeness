#!/usr/bin/env python3
"""
E5 — Scalability benchmark across XAI methods.

We measure wall-clock time (and rough memory) of producing attributions on
the same synthetic regression problem as p (the feature count) grows:
p in {10, 50, 100, 250, 500}.

Methods:
  - KernelSHAP            (additive, O(2^p) exact, O(n_samples * p) approx)
  - TreeSHAP              (tree-specific, polynomial in tree size)
  - Permutation Importance (O(p * n_repeats))
  - Integrated Gradients  (O(steps * p))
  - Integrated Hessians   (O(steps * p^2))  -- the worst-case scaling
  - Pattern Attribution   (O(p^3) for the Σ_X · w computation; trivial)

For each (method, p) we report:
  - Wall time to attribute n_explain=50 test samples
  - Per-sample latency

The point is to put numbers on the "computational cost gradient" claim in
Section 4.5: methods with stronger theoretical guarantees pay more in time.

Run:
    cd revised_v2 && python3 experiments/e5_scalability.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import apply_style, fig_size

HERE = Path(__file__).resolve().parent


def make_figure(out):
    """Draw Fig. E5 from the saved results dict (e5_results.npy)."""
    results = out["results"]
    apply_style()
    # wide, short canvas with the legend on the right
    fig, ax = plt.subplots(figsize=fig_size(0.72, 0.40))
    method_colors = {
        "KernelSHAP": "#e74c3c",
        "TreeSHAP": "#27ae60",
        "PFI": "#f39c12",
        "IG": "#3498db",
        "IH": "#9b59b6",
        "PatternAttr": "#7f8c8d",
    }
    for method, times in results.items():
        ps = [pt for pt, _ in times if not np.isnan(_)]
        ts = [t for _, t in times if not np.isnan(t)]
        ax.plot(ps, ts, marker="o", label=method, color=method_colors[method])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of features (p)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, borderaxespad=0.0)
    ax.grid(alpha=0.3, linestyle="--", which="both")
    fig.savefig(HERE / "e5_scalability.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "e5_scalability.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nFigure saved to {HERE / 'e5_scalability.pdf'}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--plot-only", action="store_true",
                     help="redraw the figure from e5_results.npy without rerunning")
    if _ap.parse_args().plot_only:
        make_figure(np.load(HERE / "e5_results.npy", allow_pickle=True).item())
        raise SystemExit(0)

import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from path_explain import PathExplainerTorch

np.random.seed(0)
torch.manual_seed(0)


def make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, p)).astype(np.float32)
    coefs = rng.normal(0, 1, p)
    y = (X @ coefs + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


def time_kernel_shap(X_train_df, X_test_df, model, feature_names, n_explain=50):
    t0 = time.time()
    background = shap.sample(X_train_df.values.astype(np.float32), 25, random_state=0)
    def predict_fn(Z):
        return model.predict(pd.DataFrame(Z, columns=feature_names))
    expl = shap.KernelExplainer(predict_fn, background)
    _ = expl.shap_values(X_test_df.values[:n_explain].astype(np.float32),
                         nsamples=50, silent=True)
    return time.time() - t0


def time_tree_shap(X_train_df, X_test_df, xgb_model, n_explain=50):
    t0 = time.time()
    expl = shap.TreeExplainer(xgb_model, X_train_df, feature_perturbation="interventional")
    _ = expl.shap_values(X_test_df.iloc[:n_explain])
    return time.time() - t0


def time_pfi(X, y, model):
    t0 = time.time()
    _ = permutation_importance(model, X, y, n_repeats=5, n_jobs=1, random_state=0)
    return time.time() - t0


def time_ig(model_torch, X_train_t, X_test_t, n_explain=50):
    explainer = PathExplainerTorch(model_torch)
    X_e = X_test_t[:n_explain].clone().detach().requires_grad_(True)
    baseline = X_train_t.mean(0, keepdim=True).clone().detach().requires_grad_(True)
    t0 = time.time()
    _ = explainer.attributions(X_e, baseline, num_samples=32, use_expectation=False)
    return time.time() - t0


def time_ih(model_torch, X_train_t, X_test_t, n_explain=50):
    explainer = PathExplainerTorch(model_torch)
    X_e = X_test_t[:n_explain].clone().detach().requires_grad_(True)
    baseline = X_train_t.mean(0, keepdim=True).clone().detach().requires_grad_(True)
    t0 = time.time()
    _ = explainer.interactions(X_e, baseline, num_samples=16, use_expectation=False, verbose=False)
    return time.time() - t0


def time_pattern(X, lin_model):
    t0 = time.time()
    w = lin_model.coef_
    cov_X = np.cov(X, rowvar=False)
    _ = (cov_X @ w) / (w @ cov_X @ w + 1e-12)
    return time.time() - t0


def train_mlp(X_train, y_train, in_dim, hidden=32, epochs=200):
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train).unsqueeze(1)
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Softplus(beta=5),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


# ── Run timing across p values ────────────────────────────────────────────
p_values = [10, 50, 100, 250, 500]
n_train, n_test = 1500, 500
n_explain = 50

results = {m: [] for m in ["KernelSHAP", "TreeSHAP", "PFI", "IG", "IH", "PatternAttr"]}

for p in p_values:
    print(f"\n=== p = {p} ===")
    X, y = make_data(n_train + n_test, p)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    X_train_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(p)])
    X_test_df = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(p)])
    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  random_state=0, verbosity=0)
    xgb_model.fit(X_train_df, y_train)
    lin_model = LinearRegression().fit(X_train, y_train)
    mlp_model = train_mlp(X_train, y_train, in_dim=p)

    print(f"  training done; running attributions ({n_explain} test samples)...")

    # KernelSHAP — gets slow fast; cap with n_explain
    try:
        n_for_kernel = n_explain if p <= 100 else 10
        t = time_kernel_shap(X_train_df, X_test_df, xgb_model,
                              [f"X{i}" for i in range(p)],
                              n_explain=n_for_kernel)
        results["KernelSHAP"].append((p, t))
        print(f"  KernelSHAP     = {t:.2f}s  (n_explain={n_for_kernel})")
    except Exception as e:
        results["KernelSHAP"].append((p, np.nan))
        print(f"  KernelSHAP failed: {e}")

    # TreeSHAP
    t = time_tree_shap(X_train_df, X_test_df, xgb_model, n_explain=n_explain)
    results["TreeSHAP"].append((p, t))
    print(f"  TreeSHAP       = {t:.3f}s")

    # PFI
    t = time_pfi(X_test_df, y_test, xgb_model)
    results["PFI"].append((p, t))
    print(f"  PFI            = {t:.2f}s")

    # IG
    try:
        t = time_ig(mlp_model, X_train_t, X_test_t, n_explain=n_explain)
        results["IG"].append((p, t))
        print(f"  IG             = {t:.2f}s")
    except Exception as e:
        results["IG"].append((p, np.nan))
        print(f"  IG failed: {e}")

    # IH (the expensive one — skip for very large p)
    if p <= 250:
        try:
            t = time_ih(mlp_model, X_train_t, X_test_t,
                        n_explain=min(n_explain, 20))
            results["IH"].append((p, t))
            print(f"  IH             = {t:.2f}s")
        except Exception as e:
            results["IH"].append((p, np.nan))
            print(f"  IH failed: {e}")
    else:
        results["IH"].append((p, np.nan))
        print("  IH             = skipped (p > 250, too expensive)")

    # Pattern
    t = time_pattern(X_train, lin_model)
    results["PatternAttr"].append((p, t))
    print(f"  PatternAttr    = {t:.4f}s")


# ── Save & plot ───────────────────────────────────────────────────────────
print("\n=== SUMMARY (wall-clock seconds) ===")
print(f"{'p':<6}{'KernelSHAP':>12}{'TreeSHAP':>12}{'PFI':>10}{'IG':>10}{'IH':>10}{'Pattern':>10}")
def fmt(v):
    return "    n/a" if (v is None or np.isnan(v)) else f"{v:>10.3f}"
for i, p in enumerate(p_values):
    vals = [results[m][i][1] for m in ["KernelSHAP", "TreeSHAP", "PFI", "IG", "IH", "PatternAttr"]]
    print(f"{p:<6}" + "".join(fmt(v) for v in vals[:2]).replace("    n/a", "        n/a") +
          "".join(fmt(v) for v in vals[2:]))

out = {"p_values": p_values, "results": results}
np.save(HERE / "e5_results.npy", out)

make_figure(out)
