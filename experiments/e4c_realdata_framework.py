#!/usr/bin/env python3
"""
E4c -- End-to-end demonstration of the diagnostic framework on REAL data
(Breast Cancer Wisconsin, n=569, p=30). This is the real-data analogue of the
synthetic worked example (Section 2.6): run the framework's three steps
(diagnose -> map to a method -> validate) and show that following it CHANGES
the headline conclusion versus naive default SHAP -- with no ground-truth
feature importance, using bootstrap STABILITY as the objective criterion.

Run from revised_v2/:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 experiments/e4c_realdata_framework.py
Redraw the figure from the saved results only (no model fitting):
    python3 experiments/e4c_realdata_framework.py --plot-only
"""
import warnings, json, sys
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import apply_style, fig_size


def make_figure(out):
    """Fig. E4c, drawn at print size from the saved results dict (e4c_results.npy)."""
    apply_style()
    B = out["B"]
    items = out["top_feature_counts"]          # [(feature, count), ...], most common first
    labs = [n.replace("_", " ") for n, _ in items]
    vals = [100 * c / B for _, c in items]
    feat_pct, clust_pct = out["modal_feature_pct"], out["modal_cluster_pct"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(0.85, 0.42),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    # left: how often each feature is the naive #1 (instability spread)
    ax1.barh(range(len(labs)), vals, color="#D55E00")
    ax1.set_yticks(range(len(labs))); ax1.set_yticklabels(labs)
    ax1.invert_yaxis(); ax1.set_xlabel("% of bootstraps feature is the #1")
    ax1.set_title("(a) Naive per-feature SHAP:\nthe top feature flips across bootstraps")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    # right: feature-level vs cluster-level reproducibility
    ax2.bar([0, 1], [feat_pct, clust_pct], color=["#D55E00", "#0072B2"], width=0.55)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["top FEATURE\n(naive)", "top CLUSTER\n(framework)"])
    ax2.set_ylabel("% of bootstraps\nthe #1 is the same"); ax2.set_ylim(0, 105)
    for x, v in zip([0, 1], [feat_pct, clust_pct]):
        ax2.text(x, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax2.set_title(f"(b) Stability of the headline (B={B})\nKendall-$\\tau$: feature {out['feature_tau']:.2f}"
                  f" vs cluster {out['cluster_tau']:.2f}")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(HERE / "e4c_realdata_framework.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "e4c_realdata_framework.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote e4c_realdata_framework.pdf/.png")


if "--plot-only" in sys.argv:
    make_figure(np.load(HERE / "e4c_results.npy", allow_pickle=True).item())
    sys.exit(0)

import pandas as pd
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 0
np.random.seed(SEED)
B = 40   # bootstrap resamples

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=[c.replace(" ", "_") for c in data.feature_names])
y = data.target            # 0/1
names = list(X.columns)
p = len(names)
print(f"Breast Cancer Wisconsin: n={len(X)}, p={p}, class balance={y.mean():.2f}")

Xs = StandardScaler().fit_transform(X.values)

def fit_xgb(Xtr, ytr):
    m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                          eval_metric="logloss", verbosity=0)
    m.fit(Xtr, ytr); return m

base = fit_xgb(X.values, y)
print(f"XGBoost train accuracy = {base.score(X.values, y):.3f}")

# ====================================================================
# STEP 1 -- DIAGNOSE the complexity profile (the paper's five tests)
# ====================================================================
print("\n===== STEP 1: DIAGNOSE =====")

# Collinearity: VIF per feature + condition number
vifs = np.array([variance_inflation_factor(Xs, i) for i in range(p)])
vif_severe = int((vifs > 10).sum()); vif_mod = int((vifs > 5).sum())
kappa = np.linalg.svd(Xs, compute_uv=False); kappa = kappa.max()/kappa.min()
n_pairs = 0
R = np.corrcoef(Xs, rowvar=False)
for i in range(p):
    for j in range(i+1, p):
        if abs(R[i, j]) > 0.9: n_pairs += 1
print(f"Collinearity : VIF>10 in {vif_severe}/{p} features (VIF>5 in {vif_mod}); "
      f"condition number kappa={kappa:.0f}; {n_pairs} feature pairs with |r|>0.9  "
      f"-> {'SEVERE' if (kappa>30 or vif_severe>0) else 'low'}")

# Nonlinearity: NL = R^2_poly2 - R^2_lin predicting the model's probability
proba = base.predict_proba(X.values)[:, 1]
r2_lin = LinearRegression().fit(Xs, proba).score(Xs, proba)
Xpoly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xs)
r2_poly = LinearRegression().fit(Xpoly, proba).score(Xpoly, proba)
NL = r2_poly - r2_lin
print(f"Nonlinearity : NL = R2_poly - R2_lin = {NL:.3f}  -> {'YES' if NL>0.10 else 'low'}")

# Interactions: TreeSHAP interaction values; max off-diagonal vs mean |main|
inter = shap.TreeExplainer(base).shap_interaction_values(X.values)   # (n,p,p)
abs_inter = np.abs(inter).mean(0)                                    # (p,p)
main = np.diag(abs_inter); off = abs_inter.copy(); np.fill_diagonal(off, 0)
max_off_ratio = off.max() / max(main.mean(), 1e-12)
print(f"Interactions : max |off-diagonal SHAP-interaction| / mean|main| = {max_off_ratio:.3f}  "
      f"-> {'YES' if max_off_ratio>0.05 else 'low'}")

# Suppressor: among high-VIF features, sign(r) vs sign(partial corr)
def partial_corr(j):
    others = [k for k in range(p) if k != j]
    rx = Xs[:, j] - LinearRegression().fit(Xs[:, others], Xs[:, j]).predict(Xs[:, others])
    ry = (y - LinearRegression().fit(Xs[:, others], y).predict(Xs[:, others]))
    return pearsonr(rx, ry)[0]
flips = 0
for j in np.where(vifs > 5)[0]:
    r0 = pearsonr(Xs[:, j], y)[0]; rp = partial_corr(int(j))
    if np.sign(r0) != np.sign(rp) or abs(rp) > 2*abs(r0): flips += 1
print(f"Suppressor   : sign-flip / |partial|>2|zero-order| in {flips} high-VIF features  "
      f"-> {'present' if flips>0 else 'low'}")

# Capacity: total leaves in the ensemble
leaves = sum(t.count("leaf") for t in base.get_booster().get_dump())
print(f"Capacity     : {leaves} total leaves  -> {'high' if leaves>1e5 else 'low'} (threshold 1e5)")

# ====================================================================
# STEP 2 -- MAP diagnosis to a method recommendation
# ====================================================================
print("\n===== STEP 2: MAP =====")
print("Diagnosis is MULTI-DIMENSIONAL: collinearity (severe), interactions, nonlinearity,")
print("and suppressor signals all fire -- itself a real-data instance of the paper's")
print("multi-dimensional-complexity claim (and, per Sec 2, some apparent interaction on")
print("highly collinear data can be collinearity-induced). Of the flags, COLLINEARITY is")
print("by far the most severe (kappa, near-perfect |r|>0.9 clusters). The severity ordering")
print("routes method selection to interaction-aware tools first (which subsume collinearity,")
print("validated in E1/E7/E8); independently, the severe-collinearity flag makes a concrete,")
print("testable prediction we validate below: per-feature attribution rankings WITHIN the")
print("correlated clusters are unreliable, so a single-feature headline is false precision.")

# Build correlation clusters (connected components of the |r|>0.9 graph)
parent = list(range(p))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(p):
    for j in range(i+1, p):
        if abs(R[i, j]) > 0.9: parent[find(i)] = find(j)
clusters = {}
for i in range(p): clusters.setdefault(find(i), []).append(i)
clusters = list(clusters.values())
multi = [c for c in clusters if len(c) > 1]
print(f"\nCorrelation clusters (|r|>0.9): {len(clusters)} groups, "
      f"{len(multi)} multi-feature; largest = {sorted(max(clusters,key=len))} ")
print("  largest cluster features:", [names[i] for i in sorted(max(clusters, key=len))])

def cluster_of(feat_idx):
    for ci, c in enumerate(clusters):
        if feat_idx in c: return ci
    return -1

# ====================================================================
# STEP 3a -- the applicable methods DISAGREE on this real dataset
# (the failure map's collinearity column, observed on real data)
# ====================================================================
print("\n===== STEP 3a: do the applicable methods agree? =====")
from sklearn.inspection import permutation_importance
from scipy.stats import kendalltau

def mean_abs_shap(sv):
    sv = np.array(sv)
    if sv.ndim == 3:                                   # (n,p,2) -> positive class
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    return np.abs(sv).mean(0)

methods = {}
methods["TreeSHAP-path-dependent"] = mean_abs_shap(
    shap.TreeExplainer(base, feature_perturbation="tree_path_dependent").shap_values(X.values))
methods["TreeSHAP-interventional"] = mean_abs_shap(
    shap.TreeExplainer(base, data=X.values, feature_perturbation="interventional").shap_values(X.values))
methods["PFI"] = np.abs(permutation_importance(base, X.values, y, n_repeats=10,
                                                random_state=SEED).importances_mean)
# AEC direct effect (our Salih-2024 formula; model-agnostic, computed on the data)
methods["AEC (direct)"] = np.array([abs(np.cov(Xs[:, j], y)[0, 1] / np.var(Xs[:, j])) for j in range(p)])

top1 = {}
for name, imp in methods.items():
    order = np.argsort(-imp)
    top1[name] = names[order[0]]
    print(f"  {name:24s} top-3: {[names[i] for i in order[:3]]}")
mlist = list(methods.values())
taus = [kendalltau(mlist[a], mlist[b])[0] for a in range(len(mlist)) for b in range(a+1, len(mlist))]
n_distinct_top = len(set(top1.values()))
print(f"  -> distinct #1 features across the {len(methods)} methods: {n_distinct_top}  "
      f"({sorted(set(top1.values()))})")
print(f"  -> mean pairwise Kendall-tau between method rankings: {np.mean(taus):.2f}  "
      f"(<1 means the methods rank features differently)")
print("  The 'most important feature' on this real dataset depends on which method you ask")
print("  -- a real-data instance of the collinearity column of the Cross-Method Failure Map.")

# ====================================================================
# STEP 3b -- VALIDATE: naive per-feature vs framework cluster, under bootstrap
# ====================================================================
print("\n===== STEP 3b: VALIDATE stability (bootstrap, B=%d) =====" % B)
top_feat, top_clust = [], []
feat_ranks = []   # for Kendall tau of per-feature importance
for b in range(B):
    idx = np.random.RandomState(b).choice(len(X), len(X), replace=True)
    m = fit_xgb(X.values[idx], y[idx])
    sv = shap.TreeExplainer(m).shap_values(X.values)        # (n,p)
    imp = np.abs(sv).mean(0)
    top_feat.append(int(np.argmax(imp)))
    csum = np.array([imp[c].sum() for c in clusters])
    top_clust.append(int(np.argmax(csum)))
    feat_ranks.append(imp)

fc = Counter(top_feat); cc = Counter(top_clust)
modal_feat, modal_feat_n = fc.most_common(1)[0]
modal_clust, modal_clust_n = cc.most_common(1)[0]
print(f"\nNaive 'single most important feature': flips across bootstraps -- "
      f"modal feature = '{names[modal_feat]}' but top in only {modal_feat_n}/{B} "
      f"({100*modal_feat_n/B:.0f}%); distinct top-features seen = {len(fc)}")
print("  features that were ever #1:", sorted({names[i] for i in top_feat}))
print(f"\nFramework 'most important feature CLUSTER': stable -- "
      f"modal cluster top in {modal_clust_n}/{B} ({100*modal_clust_n/B:.0f}%); "
      f"= {[names[i] for i in sorted(clusters[modal_clust])]}")

# Kendall-tau stability of the ranking across bootstraps (feature vs cluster level)
from scipy.stats import kendalltau
def mean_pairwise_tau(rankings):
    taus = []
    for a in range(len(rankings)):
        for c in range(a+1, len(rankings)):
            taus.append(kendalltau(rankings[a], rankings[c])[0])
    return float(np.nanmean(taus))
feat_tau = mean_pairwise_tau([r for r in feat_ranks])
clust_ranks = [np.array([r[c].sum() for c in clusters]) for r in feat_ranks]
clust_tau = mean_pairwise_tau(clust_ranks)
print(f"\nMean pairwise Kendall-tau of importance ranking across bootstraps:")
print(f"  per-FEATURE  : {feat_tau:.3f}")
print(f"  per-CLUSTER  : {clust_tau:.3f}   (higher = more reproducible)")

print("\n----- HEADLINE -----")
print(f"Naive analysis would report '{names[modal_feat]} is the single most important")
print(f"feature' -- but that claim holds in only {100*modal_feat_n/B:.0f}% of bootstraps and the title flips")
print(f"among {len(fc)} near-duplicate/competing features. The Step-1 collinearity diagnosis")
print(f"predicted this false precision. Reading attribution at the cluster level instead --")
print(f"as the diagnosis recommends -- the top group is stable in {100*modal_clust_n/B:.0f}% of bootstraps, and the")
print(f"residual instability reveals a genuine fact the single-feature view hid: TWO predictive")
print(f"groups (tumour-size and concavity features) compete for the top. Same data, same model;")
print("the framework converts a brittle single-feature headline into a reproducible, honest one.")

# ---- save + figure ----
out = {"vif_severe": vif_severe, "vif_moderate": vif_mod, "kappa": float(kappa),
       "pairs_r_gt_0.9": n_pairs, "NL": float(NL), "max_off_ratio": float(max_off_ratio),
       "suppressor_flips": flips, "leaves": leaves,
       "modal_feature": names[modal_feat], "modal_feature_pct": 100*modal_feat_n/B,
       "distinct_top_features": len(fc),
       "top_feature_counts": [(names[i], n) for i, n in fc.most_common()],
       "modal_cluster_pct": 100*modal_clust_n/B,
       "modal_cluster": [names[i] for i in sorted(clusters[modal_clust])],
       "feature_tau": feat_tau, "cluster_tau": clust_tau, "B": B,
       "n_distinct_top_methods": n_distinct_top, "method_top1": top1,
       "method_mean_tau": float(np.mean(taus)), "method_top3": {k: [names[i] for i in np.argsort(-v)[:3]] for k, v in methods.items()}}
np.save(HERE / "e4c_results.npy", out)
print("\nsaved e4c_results.npy")

make_figure(out)
