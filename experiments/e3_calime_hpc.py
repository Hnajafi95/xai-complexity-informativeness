#!/usr/bin/env python3
"""
E3 extension (HPC) -- run the OFFICIAL CALIME generative/explainer code on a
binarized version of the E3 collinearity DGP, head-to-head with plain LIME.

CALIME is the one method in the Cross-Method Failure Map we had not executed.
Its reference code is built for *classification* (RandomForestClassifier + LIME
label=1). The paper's E3 is regression, so we binarize the SAME latent DGP
( y = 1[3*s + eps > median] ), run CALIME unmodified, and run plain LIME on the
identical data as the apples-to-apples baseline.

IMPORTANT -- we deliberately AVOID CALIME's causal-DISCOVERY step
(get_causal_model -> NCDApriori), which imports the Borgelt `fim` C library that
isn't pip-installable here. Instead we hand CALIME a supplied graph for our
synthetic DGP and call its CausalDataGeneration directly (sklearn/scipy only).

NOTE: the true structure is a latent common cause
(s -> X1, s -> X2), so over the observed features there is NO directed
X1-X2 edge. X1 -> X2 is supplied as a dependency-preserving factorization
p(X1) p(X2 | X1) of the observed joint -- observationally equivalent, but not
the causal graph. CALIME therefore acts here as an on-manifold neighbourhood
sampler, not as a causal explainer. X3 is independent.

PLACE THIS FILE INSIDE THE CLONED REPO so `from src...` resolves:
    cp e3_calime_hpc.py ~/CALIME/ && cd ~/CALIME
    source ~/calime_venv/bin/activate
    python e3_calime_hpc.py

Outputs (in ~/CALIME/results_e3_calime/): e3_calime_results.npy + e3_calime.png/.pdf
"""
import os, time, warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Discovery-FREE imports only (these do NOT pull in fim or sdv):
from src.gencda.generative_model import CausalDataGeneration
from src.calime_explainer import CALimeExplainer
from src.lime_explainer import LimeExplainer

SEEDS = [0, 1, 2, 3, 4]   # repeat the whole pipeline to check robustness
N_EXPLAIN   = 40          # test instances to explain per seed
NUM_SAMPLES = 500         # neighborhood size per instance
feature_names = ["X1", "X2", "X3"]


def make_data(seed):
    rng = np.random.default_rng(seed)
    n = 2000
    s  = rng.normal(0, 1, n)
    yc = 3.0 * s + rng.normal(0, 0.3, n)
    y  = (yc > np.median(yc)).astype(int)        # balanced binary target
    X1 = s + rng.normal(0, 0.2, n)               # clean read of s
    X2 = s + rng.normal(0, 0.5, n)               # noisier read of s
    X3 = rng.normal(0, 1, n)                      # independent noise
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    return X, y


def aggregate(explainer, test, rf, tag):
    acc = np.zeros(len(feature_names)); ok = 0
    for i in range(len(test)):
        try:
            res = explainer.explain_instance(test.values[i], rf.predict_proba,
                                             labels=(1,), num_features=len(feature_names),
                                             num_samples=NUM_SAMPLES)
            exp = res[0] if isinstance(res, tuple) else res   # (explanation, data, time)
            wmap = dict(exp.as_map()[1])
            acc += np.array([abs(wmap.get(j, 0.0)) for j in range(len(feature_names))])
            ok += 1
        except Exception as e:
            print(f"    [{tag} inst {i} skipped: {type(e).__name__}: {e}]")
    return acc / max(ok, 1), ok


def run_once(seed):
    np.random.seed(seed)
    X, y = make_data(seed)
    train, test = X.iloc[:-N_EXPLAIN], X.iloc[-N_EXPLAIN:]
    ytr, yte = y[:-N_EXPLAIN], y[-N_EXPLAIN:]
    train_bb, train_gen, ybb, ygen = train_test_split(train, ytr, test_size=0.3, random_state=seed)

    # supplied factorization X1->X2 (observationally equivalent to the latent-source structure; not causal), X3 independent
    graph = nx.DiGraph(); graph.add_nodes_from(feature_names); graph.add_edge("X1", "X2")
    gen = CausalDataGeneration(graph, train_gen); gen.generate_data()

    rf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(train_bb, ybb)
    acc = rf.score(test, yte)

    calime = CALimeExplainer(graph, gen, train.values, mode="classification",
                             feature_names=feature_names, class_names=[0, 1],
                             discretize_continuous=False)
    lime_e = LimeExplainer(train.values, mode="classification",
                           feature_names=feature_names, class_names=[0, 1],
                           discretize_continuous=False)
    norm = lambda v: v / max(v.sum(), 1e-12)
    cN, _ = aggregate(calime, test, rf, "CALIME")
    lN, _ = aggregate(lime_e, test, rf, "LIME")
    return norm(cN), norm(lN), acc


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_e3_calime")
os.makedirs(path, exist_ok=True)
calime_runs, lime_runs, accs = [], [], []
for seed in SEEDS:
    t0 = time.time()
    cN, lN, acc = run_once(seed)
    calime_runs.append(cN); lime_runs.append(lN); accs.append(acc)
    print(f"seed {seed}: RF acc={acc:.3f} | "
          f"CALIME X1/X2={cN[0]/max(cN[1],1e-12):.2f} (X1={cN[0]:.3f},X2={cN[1]:.3f},X3={cN[2]:.3f}) | "
          f"LIME X1/X2={lN[0]/max(lN[1],1e-12):.2f} (X1={lN[0]:.3f},X2={lN[1]:.3f},X3={lN[2]:.3f}) "
          f"[{time.time()-t0:.0f}s]")

C = np.array(calime_runs); L = np.array(lime_runs)            # (seeds, 3)
cR = C[:, 0] / np.clip(C[:, 1], 1e-12, None)                  # per-seed X1/X2 ratios
lR = L[:, 0] / np.clip(L[:, 1], 1e-12, None)

print("\n=== robustness over", len(SEEDS), "seeds (mean +/- std) ===")
print(f"RF accuracy:        {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
print(f"CALIME X1/X2 ratio: {cR.mean():.2f} +/- {cR.std():.2f}   (per-seed: {np.round(cR,2).tolist()})")
print(f"LIME   X1/X2 ratio: {lR.mean():.2f} +/- {lR.std():.2f}   (per-seed: {np.round(lR,2).tolist()})")
wins = int((cR < lR).sum())
print(f"CALIME ratio < LIME ratio in {wins}/{len(SEEDS)} seeds "
      f"(i.e. CALIME less asymmetric than LIME)")
print("\nReading: X1,X2 are collinear reads of the same latent s. Plain LIME perturbs")
print("independently and concentrates credit on the cleaner read (large X1/X2). CALIME")
print("keeps X1,X2 coupled, sharing credit (smaller X1/X2) -- toward the symmetric answer")
print("PatternAttribution/AEC give. Same ordering in both; the effect is in the *degree*.")

np.save(os.path.join(path, "e3_calime_results.npy"),
        {"seeds": SEEDS, "calime_runs": C.tolist(), "lime_runs": L.tolist(),
         "calime_ratio_mean": float(cR.mean()), "calime_ratio_std": float(cR.std()),
         "lime_ratio_mean": float(lR.mean()), "lime_ratio_std": float(lR.std()),
         "calime_lt_lime_seeds": wins, "n_seeds": len(SEEDS),
         "rf_acc_mean": float(np.mean(accs)), "n_explain": N_EXPLAIN, "num_samples": NUM_SAMPLES})

# figure: mean importance bars with std error bars
cM, cS = C.mean(0), C.std(0); lM, lS = L.mean(0), L.std(0)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
xs = np.arange(3); w = 0.38
ax.bar(xs - w/2, cM, w, yerr=cS, capsize=4, color="#0072B2",
       label=f"CALIME (X1/X2={cR.mean():.1f}$\\pm${cR.std():.1f})")
ax.bar(xs + w/2, lM, w, yerr=lS, capsize=4, color="#D55E00",
       label=f"plain LIME (X1/X2={lR.mean():.1f}$\\pm${lR.std():.1f})")
ax.set_xticks(xs); ax.set_xticklabels(["X1 (clean)", "X2 (noisy)", "X3 (noise)"])
ax.set_ylabel("normalized importance (mean over seeds)")
ax.set_title(f"E3 (binarized): CALIME causal-aware vs plain LIME\n"
             f"on collinear reads of a latent source ({len(SEEDS)} seeds)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "e3_calime.png"), dpi=160, bbox_inches="tight")
plt.savefig(os.path.join(path, "e3_calime.pdf"), bbox_inches="tight")
print("\nwrote e3_calime_results.npy and e3_calime.png/.pdf in", path)
