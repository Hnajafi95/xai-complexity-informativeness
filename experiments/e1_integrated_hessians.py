#!/usr/bin/env python3
"""
E1 — Integrated Hessians on the A.1 DGP.

We extend the worked example in Section 2.4 with a third method:
Integrated Hessians (IH; Janizek, Sturmfels, Lee 2021). IH requires a
twice-differentiable model, so we train a small MLP on the exact same DGP
used by a1_synthetic_experiment.py, and apply the official path_explain
reference implementation.

Goal: show that IH correctly recovers the X3 * X4 multiplicative interaction
with low main effects, completing the picture established in Table 6:
KernelSHAP (additive, observational) misattributes; TreeSHAP-interaction
(tree-specific, interventional) partially captures; IH (continuous, second-
order, path-integral) captures it cleanly.

Run:
    cd revised_v2 && python3 experiments/e1_integrated_hessians.py

Outputs:
    experiments/e1_results.npy            (numerical results)
    experiments/e1_ih_comparison.pdf/.png (figure)
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from path_explain import PathExplainerTorch

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)
np.random.seed(42)

# ── 1. Reproduce the A.1 DGP exactly ─────────────────────────────────────
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
y_test_t = torch.from_numpy(y_test).unsqueeze(1)

print(f"Data: n={n}, p={X.shape[1]}, train={len(X_train)}, test={len(X_test)}")
print(f"True corr(X1,X2)={np.corrcoef(X[:,0], X[:,1])[0,1]:.3f}, corr(X3,X4)={np.corrcoef(X[:,2], X[:,3])[0,1]:.3f}")


# ── 2. Train a small twice-differentiable MLP (softplus, not ReLU) ────────
# IH requires C^2 activations; softplus is a smooth analogue of ReLU.
class MLP(nn.Module):
    def __init__(self, in_dim=9, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(in_dim=9, hidden=64)
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

model.train()
n_epochs = 800
for epoch in range(n_epochs):
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt.step()
    if (epoch + 1) % 200 == 0:
        with torch.no_grad():
            pred_te = model(X_test_t)
            tr_r2 = 1 - ((pred - y_train_t) ** 2).sum() / ((y_train_t - y_train_t.mean()) ** 2).sum()
            te_r2 = 1 - ((pred_te - y_test_t) ** 2).sum() / ((y_test_t - y_test_t.mean()) ** 2).sum()
        print(f"  epoch {epoch+1:4d}: train MSE={loss.item():.4f}, R²(train)={tr_r2:.3f}, R²(test)={te_r2:.3f}")

model.eval()
with torch.no_grad():
    pred_te = model(X_test_t).squeeze()
test_r2 = float(1 - ((pred_te - y_test_t.squeeze()) ** 2).sum() / ((y_test_t.squeeze() - y_test_t.squeeze().mean()) ** 2).sum())
print(f"\nFinal MLP test R²: {test_r2:.4f}")


# ── 3. Apply Integrated Hessians via path_explain ─────────────────────────
# Note: path_explain 1.0 has a typo in PathExplainerTorch.attributions:
#   input_tensor.requires_grad_ = True
# This assigns an attribute instead of calling the .requires_grad_() method,
# so the input never actually gets its autograd flag set, and downstream
# grad() calls fail with "does not require grad". Workaround: enable grad
# on the inputs ourselves before passing them in. (Detaching samples_input
# inside _get_samples_input would break the second-order chain that
# interactions() relies on, so we have to fix it at the source.)
print("\nRunning Integrated Hessians (this can take a minute)...")
explainer = PathExplainerTorch(model)

n_explain = 200
X_explain = X_test_t[:n_explain].clone().detach().requires_grad_(True)
baseline = X_train_t.mean(0, keepdim=True).clone().detach().requires_grad_(True)

attr_main = explainer.attributions(
    input_tensor=X_explain,
    baseline=baseline,
    num_samples=64,
    use_expectation=False,
)
attr_interactions = explainer.interactions(
    input_tensor=X_explain,
    baseline=baseline,
    num_samples=32,
    use_expectation=False,
    verbose=False,
)

attr_main_np = attr_main.detach().cpu().numpy()
attr_int_np = attr_interactions.detach().cpu().numpy()


# ── 4. Aggregate to per-feature main effects and interaction strengths ────
mean_abs_main = np.abs(attr_main_np).mean(0)                     # (p,)

p = X.shape[1]
pair_strength = np.zeros((p, p))
for j in range(p):
    for k in range(p):
        if j == k:
            continue
        pair_strength[j, k] = np.abs(attr_int_np[:, j, k]).mean()
# Symmetrize for reporting:
pair_strength_sym = 0.5 * (pair_strength + pair_strength.T)

# Total attribution = sum of main effects + sum of pairwise interactions (upper triangle).
upper_tri_idx = np.triu_indices(p, k=1)
total_main = mean_abs_main.sum()
total_inter = pair_strength_sym[upper_tri_idx].sum()
total_attr = total_main + total_inter

main_norm = mean_abs_main / total_attr
inter_norm = pair_strength_sym / total_attr  # symmetric, full matrix

print("\nIntegrated Hessians main effects (normalized):")
for name, v in sorted(zip(feature_names, main_norm), key=lambda x: -x[1]):
    print(f"  {name}: {v:.4f}")

print("\nTop IH pairwise interactions (normalized, upper triangle):")
pairs = []
for j in range(p):
    for k in range(j + 1, p):
        pairs.append((feature_names[j], feature_names[k], inter_norm[j, k]))
pairs.sort(key=lambda x: -x[2])
for name_j, name_k, v in pairs[:8]:
    print(f"  {name_j}-{name_k}: {v:.4f}")


# ── 5. Compute ground truth (same as A.1) ────────────────────────────────
gt_main = np.array([
    np.abs(2.0 * X_test[:, 0]).mean(),
    np.abs(2.0 * X_test[:, 1]).mean(),
    0.0,
    0.0,
    np.abs(0.5 * X_test[:, 4]).mean(),
    0.0, 0.0, 0.0, 0.0,
])
gt_int_34 = np.abs(5.0 * X_test[:, 2] * X_test[:, 3]).mean()
gt_total = gt_main.sum() + gt_int_34
gt_main_norm = gt_main / gt_total
gt_int_norm = gt_int_34 / gt_total
print(f"\nGround truth interaction X3-X4 (normalized): {gt_int_norm:.4f}")
print(f"IH interaction X3-X4 (normalized):           {inter_norm[2, 3]:.4f}")


# ── 6. Build the consolidated comparison vs. earlier methods ─────────────
# Load A.1 results to put IH in the same table.
a1_results_path = HERE / "a1_results.npy"
if a1_results_path.exists():
    a1 = np.load(a1_results_path, allow_pickle=True).item()
    kernel_norm = a1["kernel_importance_norm"]
    tree_norm = a1["tree_main_norm"]
else:
    kernel_norm = {n: float("nan") for n in feature_names}
    tree_norm = {n: float("nan") for n in feature_names}

print("\n" + "=" * 72)
print("CONSOLIDATED FEATURE IMPORTANCE: GT vs KernelSHAP vs TreeSHAP vs IH")
print("=" * 72)
header = f"{'Feature':<8}{'GT':>8}{'KernelSHAP':>12}{'TreeSHAP':>11}{'IH':>9}"
print(header)
print("-" * len(header))
for i, fname in enumerate(feature_names):
    print(f"{fname:<8}{gt_main_norm[i]:>8.3f}{kernel_norm.get(fname, np.nan):>12.3f}"
          f"{tree_norm.get(fname, np.nan):>11.3f}{main_norm[i]:>9.3f}")
print(f"{'Φ(X3,X4)':<8}{gt_int_norm:>8.3f}{'—':>12}{'0.163':>11}{inter_norm[2, 3]:>9.3f}")


# ── 7. Save numerical results ────────────────────────────────────────────
results = {
    "test_r2_mlp": test_r2,
    "ih_main_norm": dict(zip(feature_names, main_norm.tolist())),
    "ih_inter_x3x4_norm": float(inter_norm[2, 3]),
    "ih_top_pairs": [(a, b, float(v)) for a, b, v in pairs[:8]],
    "gt_main_norm": dict(zip(feature_names, gt_main_norm.tolist())),
    "gt_int_x3x4_norm": float(gt_int_norm),
    "kernel_norm": kernel_norm,
    "tree_norm": tree_norm,
}
np.save(HERE / "e1_results.npy", results)
# Raw per-instance IG (main) and IH (pairwise) attributions for t1_estimand_analysis.py.
np.savez(HERE / "e1_raw.npz", ig=attr_main_np, ih=attr_int_np, X=X_explain.detach().numpy())
print(f"\nResults saved to {HERE / 'e1_results.npy'}")


# ── 8. Figure ─────────────────────────────────────────────────────────────
# The E1 figure (e1_ih_comparison.pdf) is drawn by t1_estimand_analysis.py, which
# reads the raw attributions saved above and reports every method in the same
# interaction-separated readout (both halves of each pair term counted). An
# earlier version drew it here from normalized summaries that mixed IG main
# effects with half of each IH pair term.
