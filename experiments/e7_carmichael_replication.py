#!/usr/bin/env python3
"""
E7 — Focused replication of Carmichael & Scheirer (2023): how
feature-additive explainers fail as interaction order increases.

Carmichael's PostHocExplainerEvaluation toolkit
(github.com/craymichael/PostHocExplainerEvaluation) is the published code
for the original paper. That toolkit is heavy — it generates synthetic
expressions with `sympy`, fits multiple models, runs a large explainer
suite. We reproduce the *core* finding with a focused, self-contained
script: as the proportion of variance carried by pairwise interactions
grows, we track rank and magnitude accuracy of per-feature attributions.
Ground truth is the Shapley-allocation convention (each pure interaction split
50/50 between its participants), i.e. the estimand that per-feature Shapley
methods target; the interaction-separated convention is analysed in
t1_estimand_analysis.py. Run over 5 seeds (data, model, explainer).

DGP family:
    y(x) = (1 - α) · [additive part] + α · [pairwise interaction part] + ε
    additive_part      = 3·X1 + 2·X2 + 1·X3 + 0.5·X4
    interaction_part   = 4·X1·X2 + 2·X3·X4
    α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
    Coefficients are deliberately asymmetric so the ground-truth ranking is
    unambiguous and Kendall τ does not collapse from ties.

For each α we generate data, fit an XGBoost regressor, compute attributions,
and report Kendall τ between attributions and the ground-truth ranking.
Ground truth is derived from the analytical decomposition of the DGP.

Run:
    cd revised_v2 && python3 experiments/e7_carmichael_replication.py
    python3 e7_carmichael_replication.py --plot-only   # redraw figure from e7_results.npy
"""

import argparse
import io
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import apply_style, fig_size

HERE = Path(__file__).resolve().parent

ADDITIVE_COEFS = np.array([3.0, 2.0, 1.0, 0.5, 0.0, 0.0])   # X1..X4 each
INTERACTION_PAIRS = [(0, 1, 4.0), (2, 3, 2.0)]              # (j, k, coef)
METHODS = ["KernelSHAP", "LIME", "TreeSHAP_interaction", "IH"]


def make_dgp(alpha, n=2000, seed=42):
    """y = (1-α)·additive + α·interaction + ε"""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 6)).astype(np.float32)
    additive = X @ ADDITIVE_COEFS
    interaction = sum(c * X[:, j] * X[:, k] for j, k, c in INTERACTION_PAIRS)
    y = ((1 - alpha) * additive + alpha * interaction
         + rng.normal(0, 0.3, n).astype(np.float32))
    return X, y.astype(np.float32)


def ground_truth_ranking(X, alpha):
    """Mean |contribution| per feature. Each pairwise interaction is split
    50/50 between its participating features (the SHAP-fair convention)."""
    p = X.shape[1]
    gt = np.zeros(p)
    for j in range(p):
        gt[j] += (1 - alpha) * abs(ADDITIVE_COEFS[j]) * np.abs(X[:, j]).mean()
    for j, k, c in INTERACTION_PAIRS:
        share = alpha * abs(c) * np.abs(X[:, j] * X[:, k]).mean() / 2.0
        gt[j] += share
        gt[k] += share
    return gt


def normalized(v):
    s = np.abs(v).sum()
    return np.abs(v) / s if s > 0 else np.zeros_like(v)

def l1_error(method_imp, gt):
    return float(np.abs(normalized(method_imp) - normalized(gt)).sum())


def run_experiment():
    """Run the 5-seed interaction sweep and save e7_results.npy."""
    import pandas as pd
    import shap
    import torch
    import torch.nn as nn
    import xgboost as xgb
    from scipy.stats import kendalltau
    from path_explain import PathExplainerTorch
    np.random.seed(0)
    torch.manual_seed(0)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    methods = METHODS
    SEEDS = [0, 1, 2, 3, 4]      # seed 0 reproduces the original single-run configuration
    tau_runs = {m: np.zeros((len(SEEDS), len(alphas))) for m in methods}
    l1_runs = {m: np.zeros((len(SEEDS), len(alphas))) for m in methods}

    import lime, lime.lime_tabular

    for si_, seed in enumerate(SEEDS):
      torch.manual_seed(seed)
      for ai, alpha in enumerate(alphas):
        print(f"\n=== seed {seed}, α = {alpha:.2f} (interaction fraction) ===")
        X, y = make_dgp(alpha=alpha, n=2000, seed=42 + seed)
        feat = [f"X{i+1}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feat)
        gt = ground_truth_ranking(X, alpha)
        gt_rank = np.argsort(np.argsort(-gt))

        def record(m, imp):
            tau_runs[m][si_, ai] = kendalltau(np.argsort(np.argsort(-imp)), gt_rank)[0]
            l1_runs[m][si_, ai] = l1_error(imp, gt)

        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      random_state=seed, verbosity=0)
        xgb_model.fit(X_df, y)

        # KernelSHAP (marginal/interventional background sample)
        bg = shap.sample(X, 30, random_state=seed)
        def predict_fn(Z):
            return xgb_model.predict(pd.DataFrame(Z, columns=feat))
        np.random.seed(seed)
        expl_k = shap.KernelExplainer(predict_fn, bg)
        sv_k = expl_k.shap_values(X[:120], nsamples=60, silent=True)
        record("KernelSHAP", np.abs(sv_k).mean(0))

        # LIME (mean |coefficient| across 80 instances)
        lime_expl = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=feat, discretize_continuous=False, mode="regression",
            random_state=seed,
        )
        lime_imp = np.zeros(len(feat))
        for i in range(80):
            e = lime_expl.explain_instance(X[i], predict_fn, num_features=len(feat),
                                           num_samples=200)
            for fi, val in e.as_map()[0]:
                lime_imp[fi] += abs(val)
        record("LIME", lime_imp / 80)

        # TreeSHAP interaction values, read out as per-feature Shapley values:
        # phi_j = sum_k Phi_jk (diagonal main effect + both halves' share of each pair),
        # i.e. the same Shapley-allocation estimand as the ground truth.
        # shap computes interaction values only with the path-dependent algorithm
        expl_t = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
        si = expl_t.shap_interaction_values(X_df.iloc[:300])
        record("TreeSHAP_interaction", np.abs(si.sum(2)).mean(0))

        # IG attributions from a small softplus MLP (per-feature readout of IH: sum_k IH_jk = IG_j)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y).unsqueeze(1)
        mlp = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.Softplus(beta=5),
            nn.Linear(64, 64), nn.Softplus(beta=5),
            nn.Linear(64, 1),
        )
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
        for _ in range(400):
            opt.zero_grad()
            ((mlp(Xt) - yt) ** 2).mean().backward()
            opt.step()
        mlp.eval()
        explainer = PathExplainerTorch(mlp)
        Xe = Xt[:120].clone().detach().requires_grad_(True)
        base = Xt.mean(0, keepdim=True).clone().detach().requires_grad_(True)
        ih_main = explainer.attributions(Xe, base, num_samples=32, use_expectation=False)
        record("IH", np.abs(ih_main.detach().numpy()).mean(0))

        print("  " + " | ".join(f"{m}: τ={tau_runs[m][si_, ai]:.2f}, L1={l1_runs[m][si_, ai]:.3f}"
                                for m in methods))


    # ── Save & plot ──────────────────────────────────────────────────────────
    tau_per_alpha = {m: tau_runs[m].mean(0).tolist() for m in methods}
    l1_per_alpha = {m: l1_runs[m].mean(0).tolist() for m in methods}
    out = {"alphas": alphas, "seeds": SEEDS, "tau_runs": tau_runs, "l1_runs": l1_runs,
           "tau_per_alpha": tau_per_alpha, "l1_per_alpha": l1_per_alpha}
    np.save(HERE / "e7_results.npy", out)
    print("\nMean ± sd over seeds (L1):")
    for m in methods:
        print(f"  {m:<22}" + "  ".join(f"{mu:.3f}±{sd:.3f}" for mu, sd in
                                       zip(l1_runs[m].mean(0), l1_runs[m].std(0))))

    return out


def make_figure(res):
    """Draw Fig. 16 (print size, 0.9 x textwidth) from the saved results dict."""
    apply_style()
    alphas = res["alphas"]
    tau_runs, l1_runs = res["tau_runs"], res["l1_runs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(0.9, 0.4))
    colors = {"KernelSHAP": "#D55E00", "LIME": "#E69F00",
              "TreeSHAP_interaction": "#009E73", "IH": "#0072B2"}
    markers = {"KernelSHAP": "o", "LIME": "s", "TreeSHAP_interaction": "^", "IH": "D"}
    labels = {"KernelSHAP": "KernelSHAP", "LIME": "LIME",
              "TreeSHAP_interaction": "TreeSHAP (int. values, per-feature)",
              "IH": "IG (IH per-feature)"}
    for ax, runs, ylab, title in [
        (ax1, tau_runs, "Kendall τ vs. ground-truth ranking",
         "(a) Rank correlation (higher = better)"),
        (ax2, l1_runs, "$L_1$ error vs. Shapley-allocation\nground truth",
         "(b) Magnitude error (lower = better)"),
    ]:
        for m in METHODS:
            mu, sd = runs[m].mean(0), runs[m].std(0)
            ax.errorbar(alphas, mu, yerr=sd, marker=markers[m], capsize=2,
                        elinewidth=0.8, label=labels[m], color=colors[m])
        ax.set_xlabel("Interaction fraction α\n(0 = fully additive, 1 = pure interactions)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    # The tight box (pad 0.02 in) sets the print size, but in the PDF it under-measures
    # the right end of the (b) x-label by ~2 pt, clipping its closing ")". Record each
    # format's tight box, nudge that label 3 pt left, and save into the recorded box.
    boxes, get_tight = {}, fig.get_tightbbox
    for ext in ("pdf", "png"):
        fig.get_tightbbox = lambda *a, **k: boxes.setdefault(ext, get_tight(*a, **k))
        fig.savefig(io.BytesIO(), format=ext, dpi=300, bbox_inches="tight")
    del fig.get_tightbbox
    ax2.xaxis.label.set_x(0.5 - 3 / (ax2.get_position().width * fig.get_figwidth() * 72))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"e7_interaction_failure.{ext}", dpi=300,
                    bbox_inches=boxes[ext].padded(0.02))
    plt.close(fig)
    print(f"\nFigure saved to {HERE / 'e7_interaction_failure.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from saved e7_results.npy (no experiment)")
    args = ap.parse_args()
    if args.plot_only:
        res = np.load(HERE / "e7_results.npy", allow_pickle=True).item()
    else:
        res = run_experiment()
    make_figure(res)
