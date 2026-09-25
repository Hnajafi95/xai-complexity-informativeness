#!/usr/bin/env python3
"""
E6 — Multi-dimensional complexity factorial.

Tests Hypothesis H5 from the Future Directions section: no single attribution
method dominates across joint variations of collinearity and interaction
strength. We sweep a 3 × 3 grid:

  ρ (collinearity)        ∈ {0.3, 0.6, 0.9}    # X1, X2 correlation
  β_int (interaction)     ∈ {0, 2.5, 5.0}      # coefficient of X3*X4

For each cell we generate a DGP, train an XGBoost regressor, and compute
attributions with three methods:
  - KernelSHAP             (additive; interventional, 30-point background)
  - TreeSHAP-interaction   (path-dependent; shap computes interaction values
                            only in tree_path_dependent mode)
  - Integrated Hessians    (second-order; main effect = diagonal of the IH
                            matrix, on an MLP trained on the same data)

The metric is Kendall's τ between each method's ranking and the
ground-truth ranking derived from the known DGP coefficients.

Output: a 3 × 3 heatmap per method, plus a "best method per cell" summary.

Run:
    cd revised_v2 && python3 experiments/e6_multidim_factorial.py
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
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import kendalltau

from path_explain import PathExplainerTorch

HERE = Path(__file__).resolve().parent
np.random.seed(0)
torch.manual_seed(0)


def make_dgp(rho, beta_int, n=1500, seed=0):
    """y = 2 X1 + 2 X2 + 0 X3 + 0 X4 + beta_int * X3 X4 + 0.5 X5 + ε
       with corr(X1, X2) = rho.
    """
    rng = np.random.RandomState(seed)
    Sigma = np.eye(9)
    Sigma[0, 1] = Sigma[1, 0] = rho
    X = rng.multivariate_normal(np.zeros(9), Sigma, size=n).astype(np.float32)
    y = (2.0 * X[:, 0] + 2.0 * X[:, 1]
         + beta_int * X[:, 2] * X[:, 3]
         + 0.5 * X[:, 4]
         + rng.normal(0, 0.5, n).astype(np.float32))
    return X, y.astype(np.float32)


def ground_truth_ranking(X, beta_int):
    """Rank features by the expected magnitude of their per-instance contribution."""
    gt = np.zeros(X.shape[1])
    gt[0] = np.abs(2.0 * X[:, 0]).mean()
    gt[1] = np.abs(2.0 * X[:, 1]).mean()
    gt[4] = np.abs(0.5 * X[:, 4]).mean()
    # X3, X4 ground-truth main effect is 0; the interaction is separate.
    return gt


def kernel_shap_importance(model, X_train, X_test, feature_names, n_explain=100):
    background = shap.sample(X_train, 30, random_state=0)
    def predict_fn(Z):
        return model.predict(pd.DataFrame(Z, columns=feature_names))
    expl = shap.KernelExplainer(predict_fn, background)
    sv = expl.shap_values(X_test[:n_explain], nsamples=80, silent=True)
    return np.abs(sv).mean(0)


def tree_shap_interaction_importance(model, X_test_df):
    # shap supports interaction values only with the path-dependent algorithm
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    si = expl.shap_interaction_values(X_test_df)
    p = si.shape[1]
    diag = np.zeros(p)
    for j in range(p):
        diag[j] = np.abs(si[:, j, j]).mean()
    return diag


def ih_importance(X_train, X_test, in_dim, n_explain=80):
    """Train MLP and run Integrated Hessians; return |diagonal| (main-effect) magnitudes."""
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train).unsqueeze(1)  # captured below
    model_torch = nn.Sequential(
        nn.Linear(in_dim, 64), nn.Softplus(beta=5),
        nn.Linear(64, 64), nn.Softplus(beta=5),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model_torch.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = ((model_torch(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    model_torch.eval()
    explainer = PathExplainerTorch(model_torch)
    Xe = torch.from_numpy(X_test[:n_explain]).clone().detach().requires_grad_(True)
    base = Xt.mean(0, keepdim=True).clone().detach().requires_grad_(True)
    inter = explainer.interactions(Xe, base, num_samples=32, use_expectation=False)
    inter = inter.detach().numpy() if hasattr(inter, "detach") else np.asarray(inter)
    main = np.diagonal(inter, axis1=1, axis2=2)          # (n, p) IH main effects
    return np.abs(main).mean(0)


rhos = [0.3, 0.6, 0.9]
betas = [0.0, 2.5, 5.0]

methods = ["KernelSHAP", "TreeSHAP_interaction", "IH"]
tau_grid = {m: np.zeros((len(rhos), len(betas))) for m in methods}

for i, rho in enumerate(rhos):
    for j, beta_int in enumerate(betas):
        print(f"\n=== rho={rho}, beta_int={beta_int} ===")
        X, y = make_dgp(rho=rho, beta_int=beta_int, n=1500, seed=42)
        y_train = y  # for closure inside ih_importance
        feat_names = [f"X{i+1}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feat_names)
        gt = ground_truth_ranking(X, beta_int)
        gt_rank = np.argsort(np.argsort(-gt))

        # Model for SHAP variants
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      random_state=0, verbosity=0)
        xgb_model.fit(X_df, y)

        # KernelSHAP
        k_imp = kernel_shap_importance(xgb_model, X.astype(np.float32),
                                       X.astype(np.float32),
                                       feat_names, n_explain=80)
        k_rank = np.argsort(np.argsort(-k_imp))
        tau_grid["KernelSHAP"][i, j] = kendalltau(k_rank, gt_rank)[0]

        # TreeSHAP interaction (diagonal as main effect)
        t_imp = tree_shap_interaction_importance(xgb_model, X_df.iloc[:300])
        t_rank = np.argsort(np.argsort(-t_imp))
        tau_grid["TreeSHAP_interaction"][i, j] = kendalltau(t_rank, gt_rank)[0]

        # IH
        ih_imp = ih_importance(X.astype(np.float32), X.astype(np.float32), in_dim=9)
        ih_rank = np.argsort(np.argsort(-ih_imp))
        tau_grid["IH"][i, j] = kendalltau(ih_rank, gt_rank)[0]

        print(f"  τ(KernelSHAP)={tau_grid['KernelSHAP'][i, j]:.3f}, "
              f"τ(TreeSHAP_int)={tau_grid['TreeSHAP_interaction'][i, j]:.3f}, "
              f"τ(IH)={tau_grid['IH'][i, j]:.3f}")


# ── Identify which method "wins" each cell ───────────────────────────────
best_method = np.empty((len(rhos), len(betas)), dtype=object)
for i in range(len(rhos)):
    for j in range(len(betas)):
        taus = {m: tau_grid[m][i, j] for m in methods}
        top = max(taus.values())
        # report ties explicitly instead of letting argmax break them silently
        best_method[i, j] = "/".join(m for m in methods if np.isclose(taus[m], top))


print("\n=== Best method per cell ===")
print(f"{'':>5}" + "".join(f"β={b:>6.1f}" for b in betas))
for i, rho in enumerate(rhos):
    row = f"ρ={rho:.1f} "
    for j in range(len(betas)):
        row += f" {best_method[i, j]:<10}"
    print(row)


# ── Save & plot ──────────────────────────────────────────────────────────
out = {
    "rhos": rhos, "betas": betas,
    "tau_grid": {k: v.tolist() for k, v in tau_grid.items()},
    "best_method": best_method.tolist(),
}
np.save(HERE / "e6_results.npy", out)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)
for ax, m in zip(axes, methods):
    # cividis: perceptually uniform and color-blind-safe (replaces RdYlGn, which
    # is the worst case for deuteranopia and unreadable in grayscale). Every cell
    # is also annotated with its numeric tau below, so the figure reads in B&W.
    im = ax.imshow(tau_grid[m], cmap="cividis", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b}" for b in betas])
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f"{r}" for r in rhos])
    ax.set_xlabel("Interaction coefficient β_int")
    if m == "KernelSHAP":
        ax.set_ylabel("Collinearity ρ(X1, X2)")
    ax.set_title(m.replace("_", " "), fontsize=11, pad=8)
    for i in range(len(rhos)):
        for j in range(len(betas)):
            v = tau_grid[m][i, j]
            # cividis is dark at low tau, light at high tau: white text on the
            # dark cells, black text on the light ones, keeps the numbers legible.
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontweight="bold",
                    color="white" if v < 0.45 else "black", fontsize=10)

fig.suptitle(
    "E6 — Kendall τ vs. ground-truth ranking (collinearity × interaction factorial)",
    fontsize=11,
)
fig.colorbar(im, ax=axes, shrink=0.7, label="Kendall τ (1.0 = perfect)")
plt.savefig(HERE / "e6_factorial_heatmap.pdf", dpi=150, bbox_inches="tight")
plt.savefig(HERE / "e6_factorial_heatmap.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {HERE / 'e6_factorial_heatmap.pdf'}")
