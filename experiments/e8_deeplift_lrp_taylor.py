#!/usr/bin/env python3
"""
E8 -- Add three methods we previously described but did not run:
  - DeepLIFT (Shrikumar et al. 2017) via captum reference implementation
  - LRP (Bach et al. 2015) via captum reference implementation
  - Shapley-Taylor Interaction Index (Sundararajan et al. 2020),
    implemented from the paper's formula because no public reference
    implementation was found at the time of writing.

Apply all three to the A.1 worked-example DGP (the same 9-feature dataset
used in E1) so the result is directly comparable with the existing
KernelSHAP / TreeSHAP-interaction / IH numbers.

Run:
    cd revised_v2 && python3 experiments/e8_deeplift_lrp_taylor.py
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from math import comb, factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from captum.attr import DeepLift, LRP

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)
np.random.seed(42)


# ── 1. Recreate the A.1 DGP exactly ──────────────────────────────────────
n = 2000
Sigma = np.array([
    [1.0, 0.9, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
    [0.9, 1.0, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.05, 1.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.05, 0.05, 1.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.05, 0.05, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
X = np.random.multivariate_normal(np.zeros(9), Sigma, size=n).astype(np.float32)
feature_names = [f"X{i+1}" for i in range(9)]
y = (
    2.0 * X[:, 0] + 2.0 * X[:, 1]
    + 0.0 * X[:, 2] + 0.0 * X[:, 3]
    + 5.0 * X[:, 2] * X[:, 3]
    + 0.5 * X[:, 4]
    + np.random.normal(0, 0.5, size=n).astype(np.float32)
).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).unsqueeze(1)
X_test_t = torch.from_numpy(X_test)
print(f"Data: n={n}, p={X.shape[1]}, train={len(X_train)}, test={len(X_test)}")


# ── 2. Train an MLP. DeepLIFT requires ReLU; LRP also works on ReLU. ──────
# We train a separate ReLU-MLP here (the E1 experiment used softplus for IH).
class ReLUMLP(nn.Module):
    def __init__(self, in_dim=9, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


model = ReLUMLP(in_dim=9, hidden=64)
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(500):
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt.step()
model.eval()
with torch.no_grad():
    test_r2 = 1 - ((model(X_test_t) - torch.from_numpy(y_test).unsqueeze(1)) ** 2).sum() \
              / ((torch.from_numpy(y_test).unsqueeze(1) - y_test.mean()) ** 2).sum()
print(f"ReLU MLP test R² = {float(test_r2):.4f}")


# ── 3. DeepLIFT via captum ────────────────────────────────────────────────
print("\nRunning DeepLIFT (captum)...")
deeplift = DeepLift(model)
baseline = X_train_t.mean(0, keepdim=True)
X_explain = X_test_t[:200]
attr_dl = deeplift.attribute(X_explain, baseline)
mean_abs_dl = attr_dl.detach().numpy().__abs__().mean(0)
print(f"DeepLIFT |mean attribution|:")
for name, v in zip(feature_names, mean_abs_dl):
    print(f"  {name}: {v:.4f}")


# ── 4. LRP via captum ─────────────────────────────────────────────────────
print("\nRunning LRP (captum)...")
lrp = LRP(model)
attr_lrp = lrp.attribute(X_explain)
mean_abs_lrp = attr_lrp.detach().numpy().__abs__().mean(0)
print(f"LRP |mean attribution|:")
for name, v in zip(feature_names, mean_abs_lrp):
    print(f"  {name}: {v:.4f}")


# ── 5. Shapley-Taylor Interaction Index (Sundararajan et al. 2020) ────────
# Order-2 Shapley-Taylor index, exactly as defined in Sundararajan,
# Dhamdhere & Agarwal (ICML 2020). For k = 2 on a set function f over
# N = {1..p}, with discrete derivative
#   δ²_{jk} f(T) = f(T∪{j,k}) − f(T∪{j}) − f(T∪{k}) + f(T),
# the index is
#   I_ST({j})   = f({j}) − f(∅)                              (|S| < k)
#   I_ST({j,k}) = (2/p) · ∑_{T ⊆ N\{j,k}}  δ²_{jk} f(T) / C(p−1, |T|)
# The lower-order (main-effect) terms are discrete derivatives at the
# empty coalition; only the top order-k terms are Shapley-style averages
# over coalitions. This differs from the pairwise Shapley interaction
# index of Grabisch & Roubens (1999) — the quantity that SHAP's
# `shap_interaction_values` splits symmetrically — which uses weights
# |T|!(p−|T|−2)!/(p−1)! and Shapley-value mains. An earlier draft of this
# script implemented the Grabisch–Roubens weights by mistake; the code
# below implements the Shapley-Taylor definition and asserts its
# efficiency axiom (all terms sum to f(N) − f(∅)) numerically.
# Combinatorial cost is exponential in p, so we use p=9.
#
# Because evaluating the model on a subset of features requires defining
# f(T) -- the model output when only features in T are "present" -- we
# adopt the Sundararajan convention of replacing absent features with
# their baseline (mean) value.  This is the standard finite-sample
# realization of the conditional expectation under feature independence.

def f_on_subset(model, x_full, baseline, T_set):
    """Evaluate the model on x_full with features outside T replaced by baseline."""
    z = baseline.clone().repeat(x_full.size(0), 1)
    for j in T_set:
        z[:, j] = x_full[:, j]
    with torch.no_grad():
        return model(z).squeeze(1).cpu().numpy()


def shapley_taylor_k2(model, x_full, baseline, p):
    """
    Compute the order-2 Shapley-Taylor decomposition for all single features
    and all feature pairs. Returns arrays:
      mains[j]              -- order-1 (main-effect) ST attribution for j
      pairs[(j, k)]         -- order-2 (interaction) ST attribution for (j, k)
    Each entry is averaged across the n samples in x_full.
    """
    n_samples = x_full.size(0)
    N = list(range(p))
    mains = np.zeros(p)
    pairs = {pair: 0.0 for pair in combinations(range(p), 2)}

    # Precompute model evaluations on all "T" sets we'll need.
    # For order-2 ST, T ranges over subsets of N (efficient if p is small).
    # We restrict to |T| up to p (full enumeration). For p=9 that's 512 calls.
    cache = {}
    from itertools import chain, combinations as _comb
    def all_subsets(N):
        s = list(N)
        return chain.from_iterable(_comb(s, r) for r in range(len(s) + 1))
    for T in all_subsets(N):
        cache[tuple(sorted(T))] = f_on_subset(model, x_full, baseline, T)

    # Order-1 ST terms (|S| < k): discrete derivative at the empty coalition,
    #   I_ST({j}) = f({j}) − f(∅).
    empty = cache[()]
    mains_signed = np.zeros((p, n_samples))
    for j in range(p):
        mains_signed[j] = cache[(j,)] - empty
        mains[j] = np.abs(mains_signed[j]).mean()

    # Order-2 ST terms (|S| = k = 2):
    #   I_ST({j,k}) = (2/p) ∑_{T ⊆ N\{j,k}} δ²_{jk} f(T) / C(p−1, |T|).
    pairs_signed = {pair: np.zeros(n_samples) for pair in pairs}
    for j, k in pairs:
        total = np.zeros(n_samples)
        Nminusjk = [i for i in N if i not in (j, k)]
        for r in range(len(Nminusjk) + 1):
            for T in _comb(Nminusjk, r):
                T_ = tuple(sorted(T))
                Tj = tuple(sorted(T + (j,)))
                Tk = tuple(sorted(T + (k,)))
                Tjk = tuple(sorted(T + (j, k)))
                weight = (2.0 / p) / comb(p - 1, len(T))
                total += weight * (cache[Tjk] - cache[Tj] - cache[Tk] + cache[T_])
        pairs_signed[(j, k)] = total
        pairs[(j, k)] = float(np.abs(total).mean())

    # Efficiency axiom check: ∑_j I({j}) + ∑_{j<k} I({j,k}) = f(N) − f(∅)
    # (holds per sample for the SIGNED terms).
    recon = mains_signed.sum(0) + sum(pairs_signed.values())
    target = cache[tuple(range(p))] - empty
    max_gap = float(np.abs(recon - target).max())
    denom = float(np.abs(target).mean()) or 1.0
    print(f"  ST efficiency check: max |sum(terms) − (f(N)−f(∅))| = {max_gap:.6f} "
          f"({100*max_gap/denom:.4f}% of mean |f(N)−f(∅)|)")
    assert max_gap / denom < 1e-3, "Shapley-Taylor efficiency axiom violated"
    global ST_SIGNED
    ST_SIGNED = (mains_signed, pairs_signed)
    return mains, pairs


print("\nRunning Shapley-Taylor (k=2, our implementation; ~512 model evals)...")
# Use fewer samples to keep cache evaluation tractable
X_for_taylor = X_test_t[:100]
baseline_t = X_train_t.mean(0, keepdim=True)
mains, pairs = shapley_taylor_k2(model, X_for_taylor, baseline_t, p=9)
total_main = mains.sum()
total_pair = sum(pairs.values())
total_attr = total_main + total_pair
mains_norm = mains / total_attr
pairs_norm = {p: v / total_attr for p, v in pairs.items()}

print(f"\nShapley-Taylor main effects (normalized, top 5):")
top_mains = sorted(zip(feature_names, mains_norm), key=lambda x: -x[1])[:5]
for name, v in top_mains:
    print(f"  {name}: {v:.4f}")
print(f"\nShapley-Taylor top pairwise interactions (normalized):")
top_pairs = sorted(pairs_norm.items(), key=lambda x: -x[1])[:5]
for (j, k), v in top_pairs:
    print(f"  {feature_names[j]}-{feature_names[k]}: {v:.4f}")
print(f"\nShapley-Taylor Φ(X3, X4): {pairs_norm[(2, 3)]:.4f}")


# ── 6. Combine with prior E1 results into a consolidated table ────────────
e1_path = HERE / "e1_results.npy"
if e1_path.exists():
    e1 = np.load(e1_path, allow_pickle=True).item()
    kernel = e1["kernel_norm"]
    tree = e1["tree_norm"]
    ih_main = e1["ih_main_norm"]
    ih_int_34 = e1["ih_inter_x3x4_norm"]
    gt_main = e1["gt_main_norm"]
    gt_int_34 = e1["gt_int_x3x4_norm"]
else:
    kernel = tree = ih_main = {n: float("nan") for n in feature_names}
    ih_int_34 = gt_int_34 = float("nan")
    gt_main = {n: float("nan") for n in feature_names}

# Normalize DeepLIFT and LRP to sum-to-1 across features (no interaction term)
dl_norm = mean_abs_dl / mean_abs_dl.sum()
lrp_norm = mean_abs_lrp / mean_abs_lrp.sum()

print("\n" + "=" * 90)
print("CONSOLIDATED FEATURE IMPORTANCE — A.1 DGP (normalized)")
print("=" * 90)
hdr = f"{'Feature':<8}{'GT':>8}{'KSHAP':>8}{'TSHAP':>8}{'IH':>8}{'DeepLIFT':>10}{'LRP':>8}{'ST main':>10}"
print(hdr)
print("-" * len(hdr))
for i, fname in enumerate(feature_names):
    print(f"{fname:<8}"
          f"{gt_main.get(fname, np.nan):>8.3f}"
          f"{kernel.get(fname, np.nan):>8.3f}"
          f"{tree.get(fname, np.nan):>8.3f}"
          f"{ih_main.get(fname, np.nan):>8.3f}"
          f"{dl_norm[i]:>10.3f}"
          f"{lrp_norm[i]:>8.3f}"
          f"{mains_norm[i]:>10.3f}")
print(f"{'Φ(X3X4)':<8}{gt_int_34:>8.3f}{'—':>8}{'0.163':>8}{ih_int_34:>8.3f}{'—':>10}{'—':>8}{pairs_norm[(2, 3)]:>10.3f}")

# ── 7. Save results ──────────────────────────────────────────────────────
results = {
    "test_r2_relu_mlp": float(test_r2),
    "deeplift_norm": dict(zip(feature_names, dl_norm.tolist())),
    "lrp_norm": dict(zip(feature_names, lrp_norm.tolist())),
    "shapley_taylor_main_norm": dict(zip(feature_names, mains_norm.tolist())),
    "shapley_taylor_top_pairs": [
        (feature_names[j], feature_names[k], float(v))
        for (j, k), v in sorted(pairs_norm.items(), key=lambda x: -x[1])[:5]
    ],
    "shapley_taylor_int_x3x4_norm": float(pairs_norm[(2, 3)]),
}
np.save(HERE / "e8_results.npy", results)
# Raw per-instance attributions for t1_estimand_analysis.py.
_st_m, _st_p = ST_SIGNED
_st_pairs = np.zeros((9, 9, _st_m.shape[1]))
for (_j, _k), _v in _st_p.items():
    _st_pairs[_j, _k] = _st_pairs[_k, _j] = _v
np.savez(HERE / "e8_raw.npz", deeplift=attr_dl.detach().numpy(), lrp=attr_lrp.detach().numpy(),
         st_mains=_st_m, st_pairs=_st_pairs)
print(f"\nResults saved to {HERE / 'e8_results.npy'}")


# ── 8. Figure: extend the E1 comparison with the three new methods ───────
fig, ax = plt.subplots(figsize=(11.5, 5.4))
display = ["X1", "X2", "X3", "X4", "X5", "X3·X4"]

def get_vals(d, has_int, int_val):
    out = [d.get(f, np.nan) for f in feature_names[:5]]
    out.append(int_val if has_int else np.nan)
    return out

gt_vals = [gt_main.get(f, np.nan) for f in feature_names[:5]] + [gt_int_34]
ksh_vals = get_vals(kernel, False, np.nan)
tsh_vals = get_vals(tree, True, 0.163)
ih_vals = get_vals(ih_main, True, ih_int_34)
dl_vals = list(dl_norm[:5]) + [np.nan]
lrp_vals = list(lrp_norm[:5]) + [np.nan]
st_vals = list(mains_norm[:5]) + [pairs_norm[(2, 3)]]

methods_to_plot = [
    ("Ground truth", gt_vals, "#34495e"),
    ("KernelSHAP", ksh_vals, "#e74c3c"),
    ("TreeSHAP int.", tsh_vals, "#f39c12"),
    ("Integrated Hess.", ih_vals, "#27ae60"),
    ("DeepLIFT", dl_vals, "#8e44ad"),
    ("LRP", lrp_vals, "#16a085"),
    ("Shapley-Taylor", st_vals, "#2980b9"),
]
bar_w = 0.11
x_pos = np.arange(len(display))
for i, (name, vals, color) in enumerate(methods_to_plot):
    offset = (i - (len(methods_to_plot) - 1) / 2) * bar_w
    ax.bar(x_pos + offset, vals, bar_w, label=name, color=color)

ax.set_xticks(x_pos)
ax.set_xticklabels(display)
ax.set_ylabel("Normalized importance")
ax.set_title(
    "E8 — Seven methods on the A.1 DGP, including captum DeepLIFT/LRP and our Shapley-Taylor.\n"
    "Missing bars (DeepLIFT, LRP, KernelSHAP at X3·X4) indicate the method has no interaction term.",
    fontsize=10,
)
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(HERE / "e8_seven_methods.pdf", dpi=150, bbox_inches="tight")
plt.savefig(HERE / "e8_seven_methods.png", dpi=150, bbox_inches="tight")
print(f"Figure saved to {HERE / 'e8_seven_methods.pdf'}")
