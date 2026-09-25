#!/usr/bin/env python3
"""
T1 — Estimand-matched scoring of the worked-example attributions.

States the estimand each method's output represents and scores every method
against a ground truth expressed in that same estimand. Two estimands are in play on the
worked-example DGP  y = 2 X1 + 2 X2 + 5 X3 X4 + 0.5 X5 + eps:

  (ISD) Interaction-separated decomposition: one term per feature (main
        effect) plus one term per feature pair. Ground truth: mains
        |2x1|, |2x2|, 0, 0, |0.5x5|, 0...; pair (3,4) = |5 x3 x4|.
  (SA)  Shapley allocation: one score per feature, each pure interaction
        split equally between its participants. Ground truth: |2x1|, |2x2|,
        |2.5 x3 x4|, |2.5 x3 x4|, |0.5 x5|, 0...  (for zero-mean, near-
        independent features this is the exact interventional Shapley value
        of the true function).

Readouts per method (mean |.| over explained instances, normalized to sum 1):
  KernelSHAP, DeepLIFT, LRP    SA: phi_j.            ISD: phi_j, all pairs 0.
  TreeSHAP interaction values  SA: sum_k Phi_jk (= SHAP value).
                               ISD: Phi_jj; pair = Phi_jk + Phi_kj (both halves).
  IG / Integrated Hessians     SA: IG_j (= sum_k IH_jk).
                               ISD: IH_jj; pair = IH_jk + IH_kj.
  Shapley-Taylor (k=2)         SA: I_j + 0.5 sum_k I_jk.
                               ISD: I_j; pair = I_jk.
ISD vectors contain all 9 mains + all 36 pairs, so mass placed on spurious
pairs counts as error.

Part 2 (additive twin): replace 5 X3 X4 by c X3 + c X4 with c chosen so that
E|c X3| = E|2.5 X3 X4|. A per-feature importance summary should then look
the same for both DGPs (it cannot tell a main effect from interaction
participation), while an ISD readout separates them.

Inputs: a1_raw.npz, e1_raw.npz, e8_raw.npz (written by a1_synthetic_experiment.py,
e1_integrated_hessians.py, e8_deeplift_lrp_taylor.py). Run from revised_v2/:
    python3 experiments/t1_estimand_analysis.py
Outputs: experiments/t1_results.npy, experiments/t1_estimand.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
P = 9
PAIRS = list(combinations(range(P), 2))
I34 = PAIRS.index((2, 3))


# ── Ground truths ─────────────────────────────────────────────────────────
def gt_isd(X):
    m = np.zeros(P)
    m[0] = np.abs(2 * X[:, 0]).mean(); m[1] = np.abs(2 * X[:, 1]).mean()
    m[4] = np.abs(0.5 * X[:, 4]).mean()
    pr = np.zeros(len(PAIRS)); pr[I34] = np.abs(5 * X[:, 2] * X[:, 3]).mean()
    v = np.concatenate([m, pr]); return v / v.sum()


def gt_sa(X):
    m = np.zeros(P)
    m[0] = np.abs(2 * X[:, 0]).mean(); m[1] = np.abs(2 * X[:, 1]).mean()
    m[2] = m[3] = np.abs(2.5 * X[:, 2] * X[:, 3]).mean()
    m[4] = np.abs(0.5 * X[:, 4]).mean()
    return m / m.sum()


# ── Readouts ──────────────────────────────────────────────────────────────
def norm(v):
    return v / v.sum()


def per_feature(phi):                       # (n, p) -> SA and ISD readouts
    sa = norm(np.abs(phi).mean(0))
    return sa, np.concatenate([sa, np.zeros(len(PAIRS))])


def from_matrix(M):                         # (n, p, p) with rows summing to per-feature credit
    sa = norm(np.abs(M.sum(2)).mean(0))
    mains = np.abs(np.einsum("njj->nj", M)).mean(0)
    pr = np.array([np.abs(M[:, j, k] + M[:, k, j]).mean() for j, k in PAIRS])
    return sa, norm(np.concatenate([mains, pr]))


def l1(a, b):
    return float(np.abs(a - b).sum())


a1 = np.load(HERE / "a1_raw.npz")
e1 = np.load(HERE / "e1_raw.npz")
e8 = np.load(HERE / "e8_raw.npz")

# Every script explains a prefix of the same test split, so each method is scored
# against the ground truth evaluated on exactly the rows it explained.
Xk, Xt, Xe = a1["X_kernel"], a1["X_tree"], e1["X"]
methods = {}
methods["KernelSHAP"] = (*per_feature(a1["kernel"]), Xk)
methods["TreeSHAP int."] = (*from_matrix(a1["tree_int"]), Xt)
# path_explain: IG_j = sum_k IH_jk; use the IH matrix for both readouts
ih = e1["ih"]
methods["IG / IH"] = (*from_matrix(ih), Xe)
methods["DeepLIFT"] = (*per_feature(e8["deeplift"]), Xe)
methods["LRP"] = (*per_feature(e8["lrp"]), Xe)
stm, stp = e8["st_mains"].T, np.transpose(e8["st_pairs"], (2, 0, 1))   # (n,p), (n,p,p)
Xst = Xe[: stm.shape[0]]
st_sa = norm(np.abs(stm + 0.5 * stp.sum(2)).mean(0))
st_isd = norm(np.concatenate([np.abs(stm).mean(0),
                              [np.abs(stp[:, j, k]).mean() for j, k in PAIRS]]))
methods["Shapley-Taylor"] = (st_sa, st_isd, Xst)

# Sanity check: IG completeness via IH rows (path_explain: sum_k IH_jk = IG_j)
gap = np.abs(ih.sum(2) - e1["ig"]).mean() / np.abs(e1["ig"]).mean()
print(f"IH row-sum vs IG relative gap: {gap:.3f}")

rows = []
print(f"\n{'Method':<16}{'L1 vs ISD':>11}{'L1 vs SA':>10}   X3,X4 (SA)   X3,X4 main (ISD)  pair34 (ISD)")
for name, (sa, isd, X) in methods.items():
    r = dict(method=name, l1_isd=l1(isd, gt_isd(X)), l1_sa=l1(sa, gt_sa(X)),
             sa=sa, isd_main=isd[:P], isd_pair34=isd[P + I34],
             isd_other_pairs=float(isd[P:].sum() - isd[P + I34]))
    rows.append(r)
    print(f"{name:<16}{r['l1_isd']:>11.3f}{r['l1_sa']:>10.3f}   "
          f"{sa[2]:.3f},{sa[3]:.3f}   {isd[2]:.3f},{isd[3]:.3f}        {isd[P+I34]:.3f}")
print(f"{'Ground truth':<16}{'':>21}   {gt_sa(Xk)[2]:.3f},{gt_sa(Xk)[3]:.3f}   0.000,0.000        "
      f"{gt_isd(Xk)[P+I34]:.3f}")


# ── Part 2: additive twin ─────────────────────────────────────────────────
C = 2.5 * np.sqrt(2 / np.pi)                # E|c X3| = E|2.5 X3 X4| for unit-variance normals
Sigma = np.eye(P)
for i in range(5):
    for j in range(5):
        if i != j:
            Sigma[i, j] = 0.05
Sigma[0, 1] = Sigma[1, 0] = 0.9
names = [f"X{i+1}" for i in range(P)]


def run(seed, twin):
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=2000)
    inter = C * X[:, 2] + C * X[:, 3] if twin else 5.0 * X[:, 2] * X[:, 3]
    y = 2 * X[:, 0] + 2 * X[:, 1] + inter + 0.5 * X[:, 4] + rng.normal(0, 0.5, 2000)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    dtr, dte = pd.DataFrame(Xtr, columns=names), pd.DataFrame(Xte, columns=names)
    m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42,
                         verbosity=0, subsample=0.8, colsample_bytree=0.8).fit(dtr, ytr)
    np.random.seed(seed)
    ke = shap.KernelExplainer(lambda Z: m.predict(pd.DataFrame(Z, columns=names)),
                              shap.kmeans(dtr, 50))
    phi = np.asarray(ke.shap_values(dte.iloc[:200], nsamples=200, silent=True))
    ti = np.asarray(shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
                    .shap_interaction_values(dte))
    return per_feature(phi)[0], from_matrix(ti)[1]


twin_rows = []
for seed in range(5):
    k_int, t_int = run(seed, twin=False)
    k_add, t_add = run(seed, twin=True)
    twin_rows.append(dict(seed=seed, kernel_int=k_int, kernel_add=k_add,
                          kernel_profile_l1=l1(k_int, k_add),
                          tree_pair34_int=t_int[P + I34], tree_pair34_add=t_add[P + I34],
                          tree_main34_int=t_int[2] + t_int[3], tree_main34_add=t_add[2] + t_add[3]))
    print(f"seed {seed}: KernelSHAP profile L1(interaction DGP, additive twin) = "
          f"{twin_rows[-1]['kernel_profile_l1']:.3f} | TreeSHAP-int pair34: "
          f"{t_int[P+I34]:.3f} vs {t_add[P+I34]:.3f} | mains X3+X4: "
          f"{t_int[2]+t_int[3]:.3f} vs {t_add[2]+t_add[3]:.3f}")


# Oracle variant: explain the exact (noise-free) true functions instead of a fitted
# model, isolating the representational question from model-fitting error.
def oracle_profile(twin, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=2000)
    def f(Z):
        inter = C * Z[:, 2] + C * Z[:, 3] if twin else 5.0 * Z[:, 2] * Z[:, 3]
        return 2 * Z[:, 0] + 2 * Z[:, 1] + inter + 0.5 * Z[:, 4]
    np.random.seed(seed)
    ke = shap.KernelExplainer(f, shap.kmeans(X[:1400], 50))
    return per_feature(np.asarray(ke.shap_values(X[1400:1600], nsamples=200, silent=True)))[0]


o_int, o_add = oracle_profile(False), oracle_profile(True)
oracle = dict(int=o_int, add=o_add, l1=l1(o_int, o_add))
print(f"\nOracle KernelSHAP profile, interaction DGP: {np.round(o_int[:5], 3)}")
print(f"Oracle KernelSHAP profile, additive twin:   {np.round(o_add[:5], 3)}")
print(f"Oracle profile L1 between the two DGPs:     {oracle['l1']:.3f}")


def ms(key):
    v = np.array([r[key] for r in twin_rows]); return v.mean(), v.std()


summary = {k: ms(k) for k in ["kernel_profile_l1", "tree_pair34_int", "tree_pair34_add",
                              "tree_main34_int", "tree_main34_add"]}
print("\nAdditive twin, mean ± sd over 5 seeds:")
for k, (mu, sd) in summary.items():
    print(f"  {k:<20} {mu:.3f} ± {sd:.3f}")

np.save(HERE / "t1_results.npy", dict(rows=rows, twin=twin_rows, twin_summary=summary, oracle=oracle, c=C),
        allow_pickle=True)

# ── Figure ────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
labels = [r["method"] for r in rows]
x = np.arange(len(labels)); w = 0.38
ax1.bar(x - w / 2, [r["l1_isd"] for r in rows], w, color="#E69F00", label="vs. interaction-separated GT")
ax1.bar(x + w / 2, [r["l1_sa"] for r in rows], w, color="#56B4E9", label="vs. Shapley-allocation GT",
        hatch="//", edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("$L_1$ error (normalized attributions)")
ax1.set_title("(a) Error depends on the estimand it is scored against\n(worked-example DGP)", fontsize=10)
ax1.legend(fontsize=8.5, frameon=False); ax1.grid(axis="y", alpha=0.3, ls="--")

k_int, k_add = oracle["int"], oracle["add"]
xf = np.arange(5)
ax2.bar(xf - w / 2, k_int[:5], w, color="#0072B2", label=r"$f = \dots + 5X_3X_4$")
ax2.bar(xf + w / 2, k_add[:5], w, color="#CC79A7", label=r"additive twin $f = \dots + cX_3 + cX_4$",
        hatch="..", edgecolor="white")
ax2.set_xticks(xf); ax2.set_xticklabels(names[:5])
ax2.set_ylabel("KernelSHAP mean $|\\phi_j|$ (normalized)")
ax2.set_title(f"(b) Same per-feature profile for two different functions\n(KernelSHAP on the exact functions; $L_1$ = {oracle['l1']:.3f})", fontsize=10)
ax2.set_ylim(0, 0.40)
s = summary
ax2.text(0.98, 0.785, f"TreeSHAP-int. pair $(X_3,X_4)$ on fitted\nXGBoost (5 seeds): {s['tree_pair34_int'][0]:.2f} vs "
         f"{s['tree_pair34_add'][0]:.2f}", transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax2.legend(fontsize=8.5, frameon=False, loc="upper left"); ax2.grid(axis="y", alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(HERE / "t1_estimand.pdf"); fig.savefig(HERE / "t1_estimand.png", dpi=200)
print("\nSaved t1_results.npy and t1_estimand.{pdf,png}")

# ── E1 figure: GT vs KernelSHAP vs TreeSHAP-int vs IH, interaction-separated readout ──
R = {r["method"]: r for r in rows}
gt_ref = gt_isd(Xt)
disp = ["X1", "X2", "X3", "X4", "X5", "X3·X4"]
def isd6(r): return list(r["isd_main"][:5]) + [r["isd_pair34"]]
series = [("Ground truth (ISD)", list(gt_ref[:5]) + [gt_ref[P + I34]], "#333333", None),
          ("KernelSHAP (per-feature)", list(R["KernelSHAP"]["sa"][:5]) + [np.nan], "#D55E00", "//"),
          ("TreeSHAP interaction values", isd6(R["TreeSHAP int."]), "#E69F00", None),
          ("Integrated Hessians", isd6(R["IG / IH"]), "#009E73", ".."),]
fig, ax = plt.subplots(figsize=(9, 4.6)); bw = 0.2; xp = np.arange(6)
for i, (lab, vals, col, h) in enumerate(series):
    ax.bar(xp + (i - 1.5) * bw, vals, bw, label=lab, color=col, hatch=h, edgecolor="white" if h else None)
ax.set_xticks(xp); ax.set_xticklabels(disp)
ax.set_ylabel("Normalized importance (9 mains + 36 pairs = 1)")
ax.set_title("E1 — Worked-example DGP, interaction-separated readout\n"
             "KernelSHAP has no pair term: its X3, X4 bars hold the interaction's Shapley share", fontsize=10)
ax.legend(fontsize=8.5, frameon=False); ax.grid(axis="y", alpha=0.3, ls="--")
fig.tight_layout(); fig.savefig(HERE / "e1_ih_comparison.pdf"); fig.savefig(HERE / "e1_ih_comparison.png", dpi=200)
print("Saved e1_ih_comparison.{pdf,png}")
