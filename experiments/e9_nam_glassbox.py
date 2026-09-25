#!/usr/bin/env python3
"""
E9 -- Neural Additive Models (NAMs) as a glass-box baseline on the worked-example
DGP (the same DGP as E1/A.1: X1,X2 collinear; X3*X4 pure interaction with zero
main effects; X5 weak; Gaussian noise).

Point: a NAM is additive by construction -- one subnetwork per feature, summed,
with NO cross terms -- so it *structurally cannot* represent the 5*X3*X4
interaction. Because that interaction is the dominant variance component here,
a NAM hits a hard accuracy ceiling that a model able to capture interactions
(XGBoost) sails past. This is the empirical face of "NAMs exclude interactions
by design" (Agarwal et al. 2021).

Official NAM code: github.com/agarwl/neural_additive_models (TF) and
github.com/lemeln/nam (PyTorch). We implement the minimal NAM definition
directly (a per-feature MLP summed) to keep the run dependency-light and exact.

Run from revised_v2/:  python3 experiments/e9_nam_glassbox.py
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---- same DGP as A.1 / E1 -------------------------------------------------
n = 4000
Sigma = np.array([
    [1.0, 0.9, 0.05, 0.05, 0.05, 0, 0, 0, 0],
    [0.9, 1.0, 0.05, 0.05, 0.05, 0, 0, 0, 0],
    [0.05, 0.05, 1.0, 0.0, 0.0, 0, 0, 0, 0],
    [0.05, 0.05, 0.0, 1.0, 0.0, 0, 0, 0, 0],
    [0.05, 0.05, 0.0, 0.0, 1.0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1.0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1.0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1.0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1.0],
])
X = np.random.multivariate_normal(np.zeros(9), Sigma, size=n)
y = (2.0*X[:, 0] + 2.0*X[:, 1] + 0.0*X[:, 2] + 0.0*X[:, 3]
     + 5.0*X[:, 2]*X[:, 3] + 0.5*X[:, 4]
     + np.random.normal(0, 0.5, size=n))

# Variance decomposition (population intuition)
add_part = 2.0*X[:, 0] + 2.0*X[:, 1] + 0.5*X[:, 4]
int_part = 5.0*X[:, 2]*X[:, 3]
print(f"variance share -- additive: {np.var(add_part)/np.var(y):.2%}, "
      f"interaction X3*X4: {np.var(int_part)/np.var(y):.2%}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)

# standardize for stable NN training
xm, xs = Xtr.mean(0), Xtr.std(0)
ym, ysd = ytr.mean(), ytr.std()
Xtr_s = (Xtr - xm)/xs; Xte_s = (Xte - xm)/xs
ytr_s = (ytr - ym)/ysd


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.ELU(),
                                 nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, x):  # x: (B,1)
        return self.net(x)


class NAM(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.feats = nn.ModuleList([FeatureNet() for _ in range(p)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):                      # X: (B,p)
        contribs = [self.feats[j](X[:, j:j+1]) for j in range(X.shape[1])]
        self.last_contribs = torch.cat(contribs, dim=1)   # (B,p)
        return self.last_contribs.sum(1, keepdim=True) + self.bias


Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
ytr_t = torch.tensor(ytr_s, dtype=torch.float32).view(-1, 1)
Xte_t = torch.tensor(Xte_s, dtype=torch.float32)

model = NAM(9)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lossf = nn.MSELoss()
model.train()
for epoch in range(600):
    opt.zero_grad()
    out = model(Xtr_t)
    loss = lossf(out, ytr_t)
    loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    nam_pred = model(Xte_t).numpy().ravel()*ysd + ym
    nam_contribs = model.last_contribs.numpy()           # standardized scale
nam_r2 = r2_score(yte, nam_pred)

# XGBoost reference (captures interactions)
xgbm = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                        random_state=SEED, subsample=0.8, colsample_bytree=0.8)
xgbm.fit(Xtr, ytr)
xgb_r2 = r2_score(yte, xgbm.predict(Xte))

# mean |per-feature contribution| from the NAM (on standardized scale)
mean_abs = np.abs(nam_contribs).mean(0)
mean_abs = mean_abs/mean_abs.sum()

print("\n=== E9: NAM (additive glass-box) vs XGBoost on the X3*X4-interaction DGP ===")
print(f"NAM  test R^2 = {nam_r2:.3f}   (additive model: cannot represent X3*X4)")
print(f"XGB  test R^2 = {xgb_r2:.3f}   (captures the interaction)")
print("\nNAM normalized mean|contribution| per feature:")
for j, nm in enumerate(['X1','X2','X3','X4','X5','X6','X7','X8','X9']):
    print(f"  {nm}: {mean_abs[j]:.3f}")
print("\nReading: NAM puts mass on X1,X2 (and a little X5) but ~0 on X3,X4 -- "
      "it correctly sees their MAIN effects are zero, yet has no way to encode\n"
      "their joint 5*X3*X4 effect, so it forfeits the dominant variance component "
      "and its R^2 is capped far below XGBoost's.")
add_share = float(np.var(add_part)/np.var(y))
np.save('experiments/e9_results.npy', {'nam_r2': nam_r2, 'xgb_r2': xgb_r2,
        'mean_abs_contrib': mean_abs, 'add_share': add_share,
        'int_share': float(np.var(int_part)/np.var(y))})

# ---- figure: R^2 vs the additive-variance ceiling ------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4.2))
labels = ["NAM\n(additive\nglass-box)", "XGBoost\n(captures\ninteractions)"]
vals = [nam_r2, xgb_r2]
bars = ax.bar(labels, vals, width=0.55, color=["#E69F00", "#56B4E9"],
              edgecolor="black", linewidth=1.0, hatch=["//", ".."])
ax.axhline(add_share, ls="--", color="#444444", lw=1.5)
ax.text(1.46, add_share + 0.012,
        f"additive-variance ceiling ({add_share*100:.0f}%)",
        ha="right", va="bottom", fontsize=9, fontstyle="italic", color="#444444")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center",
            va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Test $R^2$", fontsize=12)
ax.set_ylim(0, 1.02)
ax.set_title("E9 — A Neural Additive Model cannot represent the $X_3X_4$ "
             "interaction\n(the dominant variance component), so its $R^2$ is "
             "capped at the additive ceiling", fontsize=10.5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("experiments/e9_nam_glassbox.pdf", bbox_inches="tight")
plt.savefig("experiments/e9_nam_glassbox.png", dpi=170, bbox_inches="tight")
print("\nFigure written to experiments/e9_nam_glassbox.pdf and .png")
