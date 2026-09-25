#!/usr/bin/env python3
"""
Color-blind-safe rebuild of the suppressor schematic (Figure 2).

Replaces the static raster figures/sup.png. The causal story is encoded on
THREE redundant channels so it survives grayscale printing and color-vision
deficiency:
  * colour      : Okabe-Ito blue (causal) vs. orange (non-causal)
  * line style  : solid thick arrow (causal) vs. dotted arrow (non-causal)
  * glyph       : check-mark (causal) vs. cross (non-causal)
plus the explicit text labels.

Run from revised_v2/:  python3 figures/make_suppressor.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Okabe-Ito palette
BLUE   = "#0072B2"   # causal
ORANGE = "#D55E00"   # non-causal
GREY   = "#999999"
INK    = "#222222"

fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

# --- nodes -----------------------------------------------------------------
nodes = {
    "X1": (2.1, 4.6, BLUE,  "$X_1$"),
    "X2": (2.1, 1.4, GREY,  "$X_2$"),
    "y":  (8.0, 3.0, ORANGE, "$y$"),
}
R = 0.62
for x, y, c, lab in nodes.values():
    ax.add_patch(Circle((x, y), R, facecolor=c, edgecolor="black",
                         linewidth=1.6, zorder=3))
    ax.text(x, y, lab, ha="center", va="center", color="white",
            fontsize=17, fontweight="bold", zorder=4)

x1 = nodes["X1"]; x2 = nodes["X2"]; yy = nodes["y"]

# --- correlation (undirected, black, double-headed) ------------------------
ax.add_patch(FancyArrowPatch((x1[0], x1[1] - R), (x2[0], x2[1] + R),
             arrowstyle="<->", mutation_scale=18, color="black",
             linewidth=1.8, zorder=2))
ax.add_patch(FancyBboxPatch((0.15, 2.62), 1.5, 0.78,
             boxstyle="round,pad=0.06", facecolor="white",
             edgecolor=GREY, linewidth=1.0, zorder=2))
ax.text(0.9, 3.0, "Correlated\n(e.g., $\\rho \\approx 0.9$)",
        ha="center", va="center", fontsize=10, color=INK)

# --- causal edge  X1 -> y  (solid, blue, check) ----------------------------
def edge(p, q, color, style, scale=22, lw=3.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale,
                 color=color, linewidth=lw, linestyle=style, zorder=2,
                 shrinkA=0, shrinkB=0))

edge((x1[0] + R*0.85, x1[1] - R*0.35), (yy[0] - R*0.9, yy[1] + R*0.55),
     BLUE, "solid", lw=3.4)
ax.text(5.0, 4.55, r"Direct causal effect", ha="center", va="center",
        fontsize=10.5, color=BLUE, fontweight="bold")
ax.text(5.0, 4.08, r"$\checkmark$", ha="center", va="center",
        fontsize=15, color=BLUE, fontweight="bold")

# --- non-causal edge  X2 -> y  (dotted, orange, cross) ---------------------
edge((x2[0] + R*0.85, x2[1] + R*0.35), (yy[0] - R*0.9, yy[1] - R*0.6),
     ORANGE, (0, (1, 1.6)), lw=2.6)
ax.text(5.0, 1.35, r"No direct causal effect", ha="center", va="center",
        fontsize=10.5, color=ORANGE, fontweight="bold")
# cross glyph on a white badge sitting on the dotted edge
bx, by = 5.05, 2.02
ax.add_patch(Circle((bx, by), 0.30, facecolor="white", edgecolor=ORANGE,
                    linewidth=2.0, zorder=4))
ax.text(bx, by, r"$\times$", ha="center", va="center", fontsize=17,
        color=ORANGE, fontweight="bold", zorder=5)

plt.tight_layout()
plt.savefig("figures/suppressor_cb.pdf", bbox_inches="tight")
plt.savefig("figures/suppressor_cb.png", dpi=200, bbox_inches="tight")
print("wrote figures/suppressor_cb.pdf and .png")
