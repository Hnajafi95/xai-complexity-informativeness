#!/usr/bin/env python3
"""
Figure 1 -- complexity taxonomy (v2), regenerable replacement for the static
raster figures/complexity_dims.png.

Hierarchy shown (consistent with the manuscript text):
  * Root: Informativeness vs. Complexity Conflation (+ one-line thesis).
  * Cross-cutting challenges: lack of ground truth, definitional ambiguity.
  * Two complexity LAYERS holding the five DIMENSIONS:
      - Data-level  (exists before any model is fit; primary drivers):
            collinearity, feature interactions, suppressor effects
      - Model-level (introduced by the fitted model):
            nonlinearity, model capacity & architecture (attention)
  * Amplifiers (NOT dimensions): training-induced regularization and
    optimization / seed variability / underspecification -- a subordinate,
    dashed band attached to the model level.
  * Matching layer: method-level assumptions (NOT part of the taxonomy); an
    attribution failure is a method assumption violated by data- or
    model-level complexity ("mismatch" arrows).

Every dimension carries an orange "x suffers" block (documented failures) and
a sky-blue "check addresses" block (designed to be robust).  Okabe-Ito colors
are always paired with a drawn glyph AND a word, so the figure reads in
grayscale and under color-vision deficiency.

Output (vector PDF + 300-dpi PNG + grayscale preview), full JAIR text width:
    figures/complexity_taxonomy_v2.pdf
    figures/complexity_taxonomy_v2.png
    figures/complexity_taxonomy_v2_gray.png

Run:  python3 figures/make_complexity_taxonomy.py   (from revised_v2/ or anywhere)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "complexity_taxonomy_v2")

# ---------------------------------------------------------------- fonts ----
for f in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"):
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.usetex": False,
})

# -------------------------------------------------------------- palette ----
INK = "#1F2A36"          # main text
CITE = "#4D5966"         # citations (still >= 6 pt, darker than 50% gray)
FRAME = "#3B4754"
SUFF = "#E69F00"         # Okabe-Ito orange
SUFF_BG = "#FDF3E1"
ADDR = "#56B4E9"         # Okabe-Ito sky blue
ADDR_BG = "#E6F4FC"
PANEL_BG = "#F3F5F8"
LAYER_HD = "#2E3E50"
DIM_BG = "#FFFFFF"
CROSS_BG = "#E4E7EB"
AMP_EC = "#6B7682"
MATCH_BG = "#FAF8F4"
MATCH_EC = "#5B5347"
CHIP_BG = "#FFFFFF"

# ------------------------------------------------------------ geometry -----
W = 6.5                       # full JAIR text width (in)
LM, RM = 0.06, 0.06
FS = 6.0                      # body font size (pt) -- the minimum used anywhere
LH = FS * 1.16 / 72.0         # body line height (in)
PAD = 0.04                    # inner text padding (in)
G_PANEL = 0.09                # gutter between the two layer panels
P_IN = 0.045                  # panel inner padding
CG = 0.045                    # column gap
X0, X1 = LM, W - RM
CW = ((X1 - X0) - G_PANEL - 4 * P_IN - 3 * CG) / 5.0   # mean column width
# the capacity column carries the longest citation list: borrow width from
# the (short) nonlinearity column so both model-level columns stay compact
SHIFT = 0.095
COL_W = [CW, CW, CW, CW - SHIFT, CW + SHIFT]

# ------------------------------------------------------------- content -----
# A line is a list of runs (text, style); style keys: c(olour), w(eight), s(tyle)
M = {"c": INK}                        # method name
C = {"c": CITE}                       # citation
I = {"c": INK, "s": "italic"}         # italic note
B = {"c": INK, "w": "bold"}


LINKS = []   # (bibkey, x0, y0, x1, y1) in figure fractions, for the LaTeX link overlay


def L(*runs):
    return list(runs)


DIMS = [
    dict(name="Collinearity", layer="data",
         suffers=[
             L(("KernelSHAP, marginal SHAP", M)),
             L(("  (Lundberg & Lee 2017)", {"c": CITE, "k": "Lundberg2017unified"})),
             L(("LIME ", M), ("(Ribeiro et al. 2016)", {"c": CITE, "k": "Ribeiro2016why"})),
             L(("PFI ", M), ("(Fisher et al. 2019)", {"c": CITE, "k": "Fisher2019all"})),
             L(("Evidence: ", I), ("Haufe et al. (2014);", {"c": CITE, "k": "Haufe2014interpretation"})),
             L(("  Wilming et al. (2022)", {"c": CITE, "k": "Wilming2021scrutinizing"})),
         ],
         addresses=[
             L(("AEC ", M), ("(Salih 2024b)", {"c": CITE, "k": "Salih2024aec"})),
             L(("Grouped / cluster-level", M)),
             L(("  attribution", M)),
             L(("PatternAttribution", M)),
             L(("  (Kindermans et al. 2018)", {"c": CITE, "k": "Kindermans2017learning"})),
         ]),
    dict(name="Feature interactions", layer="data",
         suffers=[
             L(("Cannot isolate the", I)),
             L(("interaction term:", I)),
             L(("SHAP ", M), ("(per-feature)", C), (", LIME", M)),
             L(("Evidence: ", I), ("Carmichael &", {"c": CITE, "k": "Carmichael2023how"})),
             L(("  Scheirer (2023)", {"c": CITE, "k": "Carmichael2023how"})),
         ],
         addresses=[
             L(("Integrated Hessians", M)),
             L(("  (Janizek et al. 2021)", {"c": CITE, "k": "Janizek2021explaining"})),
             L(("Shapley-Taylor", M)),
             L(("  (Sundararajan et al. 2020)", {"c": CITE, "k": "Sundararajan2020shapley"})),
             L(("TreeSHAP interaction values", M)),
             L(("  (Lundberg et al. 2020)", {"c": CITE, "k": "Lundberg2020local"})),
         ]),
    dict(name="Suppressor effects", layer="data",
         suffers=[
             L(("SHAP, LIME, PFI", M)),
             L(("IG ", M), ("(Sundararajan et al. 2017)", {"c": CITE, "k": "Sundararajan2017axiomatic"})),
             L(("LRP ", M), ("(Bach et al. 2015)", {"c": CITE, "k": "Bach2015pixel"})),
             L(("Evidence: ", I), ("Conger (1974);", {"c": CITE, "k": "Conger1974revised"})),
             L(("  Wilming et al. (2022)", {"c": CITE, "k": "Wilming2021scrutinizing"})),
         ],
         addresses=[
             L(("PatternAttribution", M)),
             L(("  (Kindermans et al. 2018)", {"c": CITE, "k": "Kindermans2017learning"})),
             L(("Haufe activation patterns", M)),
             L(("  (Haufe et al. 2014)", {"c": CITE, "k": "Haufe2014interpretation"})),
         ]),
    dict(name="Nonlinearity", layer="model",
         suffers=[
             L(("LIME, Gradient \u00d7 Input", M)),
             L(("Evidence: ", I), ("XAI-TRIS", M)),
             L(("  (Clark et al. 2024)", {"c": CITE, "k": "Clark2024xaitris"})),
         ],
         addresses=[
             L(("IG", M)),
             L(("  (Sundararajan et al. 2017)", {"c": CITE, "k": "Sundararajan2017axiomatic"})),
             L(("DeepLIFT", M)),
             L(("  (Shrikumar et al. 2017)", {"c": CITE, "k": "Shrikumar2017learning"})),
             L(("NAMs", M)),
             L(("  (Agarwal et al. 2021)", {"c": CITE, "k": "Agarwal2021neural"})),
         ]),
    dict(name="Capacity & architecture", layer="model",
         suffers=[
             L(("Grad-CAM, Guided Backprop", M)),
             L(("  (Selvaraju et al. 2017;", {"c": CITE, "k": "Selvaraju2017gradcam"})),
             L(("  Springenberg et al. 2015)", {"c": CITE, "k": "Springenberg2015striving"})),
             L(("  fail sanity checks ", I), ("(Adebayo", {"c": CITE, "k": "Adebayo2018sanity"})),
             L(("  et al. 2018; ", {"c": CITE, "k": "Adebayo2018sanity"}), ("Sixt et al. 2020)", {"c": CITE, "k": "Sixt2019when"})),
             L(("Raw attention ", M), ("(Jain & Wallace", {"c": CITE, "k": "Jain2019attention"})),
             L(("  2019; ", {"c": CITE, "k": "Jain2019attention"}), ("Wiegreffe & Pinter 2019)", {"c": CITE, "k": "Wiegreffe2019attention"})),
             L(("Additive surrogates under", M)),
             L(("  attention ", M), ("(Leemann et al. 2025)", {"c": CITE, "k": "Leemann2024attention"})),
         ],
         addresses=[
             L(("SLALOM ", M), ("(Leemann et al. 2025)", {"c": CITE, "k": "Leemann2024attention"})),
         ]),
]

ASSUMPTIONS = [
    ("Feature independence / marginal sampling",
     "KernelSHAP, PFI, LIME",
     "collinearity, suppressors"),
    ("Local linearity",
     "LIME, Gradient \u00d7 Input",
     "nonlinearity"),
    ("Additivity (one score per feature)",
     "SHAP, LIME, IG",
     "interactions, attention"),
    ("Differentiability",
     "IG, Grad-CAM, gradients",
     "non-smooth architectures"),
]

# ------------------------------------------------------------ helpers ------
WARN = []


def rbox(ax, x, y, w, h, fc, ec, lw=0.6, r=0.03, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, size=FS, ha="left", va="baseline", c=INK, w="normal",
         st="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=c,
                   fontweight=w, fontstyle=st, zorder=z, **kw)


def lines_block(ax, x, ytop, lines, width, size=FS, lh=LH, tag=""):
    """Render run-lines from ytop downwards; returns y of last baseline."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    y = ytop - size / 72.0 * 0.95          # first baseline (cap height ~0.72em)
    for ln in lines:
        cx = x
        for s, st in ln:
            t = text(ax, cx, y, s, size=size, c=st.get("c", INK),
                     w=st.get("w", "normal"), st=st.get("s", "normal"))
            bb = t.get_window_extent(renderer=r)
            if st.get("k"):
                fw, fh = fig.bbox.width, fig.bbox.height
                LINKS.append((st["k"], bb.x0 / fw, bb.y0 / fh, bb.x1 / fw, bb.y1 / fh))
            cx += bb.width / fig.dpi
        if cx - x > width + 1e-3:
            WARN.append(f"overflow [{tag}] '{''.join(s for s, _ in ln)}' "
                        f"{cx - x:.3f} > {width:.3f}")
        y -= lh
    return y + lh


def glyph_cross(ax, cx, cy, s, c="black", lw=1.1):
    ax.add_line(Line2D([cx - s, cx + s], [cy - s, cy + s], c=c, lw=lw,
                       solid_capstyle="round", zorder=6))
    ax.add_line(Line2D([cx - s, cx + s], [cy + s, cy - s], c=c, lw=lw,
                       solid_capstyle="round", zorder=6))


def glyph_check(ax, cx, cy, s, c="black", lw=1.1):
    ax.add_line(Line2D([cx - s, cx - 0.25 * s, cx + s],
                       [cy + 0.05 * s, cy - 0.8 * s, cy + s], c=c, lw=lw,
                       solid_capstyle="round", solid_joinstyle="round",
                       zorder=6))


def arrow(ax, p0, p1, c=FRAME, lw=0.8, style="-|>", ms=6, ls="-", z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=c, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def check_inside(ax, t, box, tag):
    fig = ax.figure
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    x, y, w, h = box
    if bb.x0 < x - 1e-3 or bb.x1 > x + w + 1e-3 or bb.y0 < y - 1e-3 or bb.y1 > y + h + 1e-3:
        WARN.append(f"text outside box [{tag}]: {bb} vs {box}")


# ------------------------------------------------------ vertical layout ----
TITLE_H = 0.36
CROSS_H = 0.16
LAYER_H = 0.27
DIM_H = 0.16
HDR_H = 0.13
DIM_GAP = 0.025               # dimension name -> suffers block
BLK_GAP = 0.035               # suffers block -> addresses block


def body_h(n):
    return n * LH + 2 * PAD - 0.015


COL_H = [DIM_H + DIM_GAP + HDR_H + body_h(len(d["suffers"])) + BLK_GAP
         + HDR_H + body_h(len(d["addresses"])) for d in DIMS]
PANEL_H = LAYER_H + P_IN + max(COL_H) + P_IN
AMP_LINES = 4
SLOT_H = AMP_LINES * LH + 0.085
MATCH_H = 0.155 + 3 * LH + 0.06 + 0.045
LEG_H = 0.12
# gaps: top, title-cross, cross-panels, panels-slot, slot-match, match-legend, bottom
GAPS = [0.035, 0.06, 0.095, 0.13, 0.065, 0.05, 0.03]
H = TITLE_H + CROSS_H + PANEL_H + SLOT_H + MATCH_H + LEG_H + sum(GAPS)


def build():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def tw(t):
        return t.get_window_extent(renderer=r).width / fig.dpi

    y = H - GAPS[0]
    # ---------------------------------------------------------- title ----
    tbw = X1 - X0
    tx = (W - tbw) / 2
    rbox(ax, tx, y - TITLE_H, tbw, TITLE_H, "white", FRAME, lw=1.1, r=0.04)
    text(ax, W / 2, y - 0.155, "Informativeness vs. Complexity Conflation",
         size=10, ha="center", w="bold")
    t2 = text(ax, W / 2, y - 0.285,
              "Attribution reliability depends on the match between a method’s "
              "assumptions and the complexity profile of the model–data system.",
              size=6.8, ha="center", st="italic", c="#33404D")
    check_inside(ax, t2, (tx, y - TITLE_H, tbw, TITLE_H), "title")
    title_bot = y - TITLE_H
    y = title_bot - GAPS[1]

    # ------------------------------------------------- cross-cutting bar ----
    cbw = 3.7
    cbx = (W - cbw) / 2
    rbox(ax, cbx, y - CROSS_H, cbw, CROSS_H, CROSS_BG, "#9AA3AD", lw=0.6, r=0.03)
    t = text(ax, W / 2, y - CROSS_H / 2,
             "CROSS-CUTTING CHALLENGES:   lack of ground truth   ·   definitional ambiguity",
             size=6.5, ha="center", va="center_baseline", c=INK)
    check_inside(ax, t, (cbx, y - CROSS_H, cbw, CROSS_H), "cross")
    ax.add_line(Line2D([W / 2, W / 2], [title_bot, y], c=FRAME, lw=0.8, zorder=2))
    cross_bot = y - CROSS_H
    y = cross_bot - GAPS[2]

    # ----------------------------------------------------- layer panels ----
    ptop = y
    pbot = ptop - PANEL_H
    dw = 3 * CW + 2 * CG + 2 * P_IN
    mw = 2 * CW + CG + 2 * P_IN
    panels = {"data": (X0, dw), "model": (X1 - mw, mw)}
    layer_txt = {
        "data": ("DATA-LEVEL COMPLEXITY",
                 "exists before any model is fit  ·  primary drivers"),
        "model": ("MODEL-LEVEL COMPLEXITY",
                  "introduced by the fitted model"),
    }
    xm = W / 2
    ybus = cross_bot - GAPS[2] * 0.45
    ax.add_line(Line2D([xm, xm], [cross_bot, ybus], c=FRAME, lw=0.8, zorder=2))
    centers = [panels[k][0] + panels[k][1] / 2 for k in ("data", "model")]
    ax.add_line(Line2D([centers[0], centers[1]], [ybus, ybus], c=FRAME, lw=0.8, zorder=2))
    for cxp in centers:
        arrow(ax, (cxp, ybus), (cxp, ptop), ms=5.5)

    col_x = []
    for key, (px, pw) in panels.items():
        rbox(ax, px, pbot, pw, PANEL_H, PANEL_BG, FRAME, lw=0.8, r=0.04)
        rbox(ax, px, ptop - LAYER_H, pw, LAYER_H, LAYER_HD, LAYER_HD, lw=0.8, r=0.04, z=2)
        a, b = layer_txt[key]
        text(ax, px + pw / 2, ptop - 0.115, a, size=7.6, ha="center", w="bold", c="white")
        text(ax, px + pw / 2, ptop - 0.225, b, size=6.3, ha="center", st="italic", c="#E3E8EE")
        idx = [0, 1, 2] if key == "data" else [3, 4]
        xx = px + P_IN
        for i in idx:
            col_x.append(xx)
            xx += COL_W[i] + CG

    # ----------------------------------------------------- dimensions ------
    yd = ptop - LAYER_H - P_IN
    ybot = pbot + P_IN
    for d, cx, cw in zip(DIMS, col_x, COL_W):
        rbox(ax, cx, yd - DIM_H, cw, DIM_H, DIM_BG, FRAME, lw=0.7, r=0.025)
        t = text(ax, cx + cw / 2, yd - DIM_H / 2, d["name"], size=6.9,
                 ha="center", va="center_baseline", w="bold")
        check_inside(ax, t, (cx, yd - DIM_H, cw, DIM_H), d["name"])
        # suffers
        ys = yd - DIM_H - DIM_GAP
        sb = body_h(len(d["suffers"]))
        rbox(ax, cx, ys - HDR_H - sb, cw, HDR_H + sb, SUFF_BG, SUFF, lw=0.7, r=0.02)
        rbox(ax, cx, ys - HDR_H, cw, HDR_H, SUFF, SUFF, lw=0.7, r=0.02, z=2)
        glyph_cross(ax, cx + 0.075, ys - HDR_H / 2, 0.026)
        text(ax, cx + 0.14, ys - HDR_H / 2, "suffers", size=6.5, w="bold",
             va="center_baseline", c="black")
        yl = lines_block(ax, cx + PAD, ys - HDR_H - PAD + 0.005, d["suffers"],
                         cw - 2 * PAD, tag=d["name"] + "/suffers")
        if yl - 0.02 < ys - HDR_H - sb:
            WARN.append(f"vertical overflow {d['name']}/suffers")
        # addresses (stretched to the common panel bottom)
        ya = ys - HDR_H - sb - BLK_GAP
        rbox(ax, cx, ybot, cw, ya - ybot, ADDR_BG, ADDR, lw=0.7, r=0.02)
        rbox(ax, cx, ya - HDR_H, cw, HDR_H, ADDR, ADDR, lw=0.7, r=0.02, z=2)
        glyph_check(ax, cx + 0.075, ya - HDR_H / 2, 0.028)
        text(ax, cx + 0.14, ya - HDR_H / 2, "addresses", size=6.5, w="bold",
             va="center_baseline", c="black")
        yl = lines_block(ax, cx + PAD, ya - HDR_H - PAD + 0.005, d["addresses"],
                         cw - 2 * PAD, tag=d["name"] + "/addresses")
        if yl - 0.02 < ybot:
            WARN.append(f"vertical overflow {d['name']}/addresses")

    # ------------------------------------------ slot row: mismatch + amps ---
    stop = pbot - GAPS[3]
    sbot = stop - SLOT_H
    mtop = sbot - GAPS[4]
    mbot = mtop - MATCH_H
    mpx, mpw = panels["model"]
    mx_data = X0 + 0.24
    mx_model = mpx + 0.19
    for xa in (mx_data, mx_model):
        arrow(ax, (xa, mtop), (xa, pbot), c=INK, lw=1.3, style="<|-|>", ms=7, z=4)
        text(ax, xa - 0.055, (pbot + mtop) / 2, "mismatch", size=6.3, ha="center",
             va="center", rotation=90, w="bold", c=INK, st="italic")
    # principle callout between the two mismatch arrows
    cbx0 = mx_data + 0.15
    cbx1 = mpx - 0.10
    cbh = SLOT_H - 0.07
    cby = sbot + (SLOT_H - cbh) / 2
    rbox(ax, cbx0, cby, cbx1 - cbx0, cbh, "white", INK, lw=0.9, r=0.03)
    ccx = (cbx0 + cbx1) / 2
    t = text(ax, ccx, cby + cbh / 2 + 0.035,
             "Attribution failure  =  mismatch", size=7.4, ha="center", w="bold")
    check_inside(ax, t, (cbx0, cby, cbx1 - cbx0, cbh), "callout1")
    t = text(ax, ccx, cby + cbh / 2 - 0.105,
             "a method-level assumption (below) violated by data- or model-level complexity (above)",
             size=6.0, ha="center", c=INK)
    check_inside(ax, t, (cbx0, cby, cbx1 - cbx0, cbh), "callout2")
    ymid = cby + cbh / 2
    dash = (0, (2, 1.5))
    ax.add_line(Line2D([mx_data + 0.03, cbx0], [ymid, ymid], c=INK, lw=0.6, ls=dash, zorder=3))
    ax.add_line(Line2D([cbx1, mx_model - 0.1], [ymid, ymid], c=INK, lw=0.6, ls=dash, zorder=3))

    # amplifiers band: subordinate (dashed outline, gray text, no color)
    ax0 = mx_model + 0.15
    ax1 = mpx + mpw
    AMP = "#46505B"
    rbox(ax, ax0, sbot, ax1 - ax0, SLOT_H, "white", AMP_EC, lw=0.8, r=0.03, ls=(0, (3, 2)))
    amp_lines = [
        L(("AMPLIFIERS", {"c": AMP, "w": "bold"}),
          ("  (not a sixth dimension)", {"c": AMP, "s": "italic"})),
        L(("Training-induced: regularization (L1/L2, dropout,", {"c": AMP})),
        L(("early stopping); optimization & seed variability /", {"c": AMP})),
        L(("underspecification ", {"c": AMP}), ("(D’Amour et al. 2022)", {"c": CITE, "k": "DAmour2022underspecification"})),
    ]
    lines_block(ax, ax0 + 0.05, sbot + SLOT_H - 0.035, amp_lines, ax1 - ax0 - 0.09,
                size=6.0, tag="amplifiers")
    # dotted "amplifies" arrows up into the model-level panel
    xs = [ax0 + 0.12 * (ax1 - ax0), ax0 + 0.92 * (ax1 - ax0)]
    for xa in xs:
        arrow(ax, (xa, sbot + SLOT_H), (xa, pbot), c=AMP_EC, lw=0.8, ms=5, ls=(0, (1.5, 1.2)))
    text(ax, (xs[0] + xs[1]) / 2, (sbot + SLOT_H + pbot) / 2, "amplifies existing dimensions",
         size=6.0, ha="center", va="center", st="italic", c=AMP)

    # --------------------------------------------------- matching layer ----
    rbox(ax, X0, mbot, X1 - X0, MATCH_H, MATCH_BG, MATCH_EC, lw=0.9, r=0.04, ls=(0, (5, 1.5)))
    text(ax, X0 + 0.07, mtop - 0.115, "MATCHING LAYER:  METHOD-LEVEL ASSUMPTIONS", size=7.2,
         w="bold", c="#3A332A")
    text(ax, X1 - 0.07, mtop - 0.115,
         "not part of the complexity taxonomy — what complexity is matched against",
         size=6.2, ha="right", st="italic", c="#5B5347")
    lns_all = []
    for nm, by, vio in ASSUMPTIONS:
        lns_all.append([
            L((nm, {"c": INK, "w": "bold"})),
            L(("assumed by: ", {"c": CITE, "s": "italic"}), (by, {"c": INK})),
            L(("violated by: ", {"c": CITE, "s": "italic"}), (vio, {"c": INK})),
        ])
    # natural widths -> distribute spare width evenly
    def lw_(ln):
        tot = 0
        for s_, st in ln:
            t = text(ax, 0, -5, s_, size=6.0, w=st.get("w", "normal"), st=st.get("s", "normal"))
            tot += tw(t); t.remove()
        return tot
    nat = [max(lw_(ln) for ln in lns) + 0.08 for lns in lns_all]
    cg = 0.05
    avail = (X1 - X0) - 0.10 - cg * (len(nat) - 1)
    spare = (avail - sum(nat)) / len(nat)
    if spare < 0:
        WARN.append(f"assumption chips too wide by {-spare * len(nat):.3f}")
    x = X0 + 0.05
    chh = MATCH_H - 0.155 - 0.045
    yb = mbot + 0.045
    for lns, nw in zip(lns_all, nat):
        cwid = nw + spare
        rbox(ax, x, yb, cwid, chh, CHIP_BG, "#8C8273", lw=0.6, r=0.025)
        yl = lines_block(ax, x + 0.04, yb + chh - 0.032, lns, cwid - 0.08, size=6.0,
                         tag="assumption")
        if yl - 0.02 < yb:
            WARN.append("vertical overflow assumption chip")
        x += cwid + cg

    # ---------------------------------------------------------- legend -----
    ly = mbot - GAPS[5] - LEG_H / 2

    def chip(x, fc, kind, word):
        t = text(ax, x + 0.12, ly, word, size=6.3, w="bold", va="center_baseline", c="black")
        w_ = tw(t) + 0.17
        rbox(ax, x, ly - 0.055, w_, 0.11, fc, fc, lw=0.5, r=0.015)
        (glyph_cross if kind == "x" else glyph_check)(ax, x + 0.062, ly, 0.023, lw=1.0)
        return x + w_ + 0.05

    def run(x, s, **kw):
        t = text(ax, x, ly, s, size=6.0, va="center_baseline", **kw)
        return x + tw(t)

    x = X0 + 0.02
    x = chip(x, SUFF, "x", "suffers")
    x = run(x, "= documented failures") + 0.25
    x = chip(x, ADDR, "check", "addresses")
    x = run(x, "= designed to be robust") + 0.25
    t = text(ax, X1 - 0.02, ly,
             "Okabe\u2013Ito colors paired with glyph + word: grayscale- and color-blind-safe",
             size=6.0, ha="right", va="center_baseline", st="italic", c=CITE)
    xe = x + tw(t)
    if xe > X1:
        WARN.append(f"legend overflow {xe:.3f} > {X1:.3f}")
    return fig


def main():
    fig = build()
    fig.savefig(OUT + ".pdf")
    # LaTeX overlay: one invisible biblatex link per drawn citation (Figure 1 in camera-ready.tex)
    aspect = H / W
    with open(OUT + "_links.tex", "w") as fh:
        fh.write("% Generated by make_complexity_taxonomy.py -- clickable citations for Figure 1.\n")
        for k, x0, y0, x1, y1 in LINKS:
            fh.write("\\node[anchor=south west,inner sep=0pt] at (%.4f,%.4f) "
                     "{\\figcitelink{%.4f\\textwidth}{%.4f\\textwidth}{%s}};\n"
                     % (x0, y0, x1 - x0, (y1 - y0) * aspect, k))
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    try:
        from PIL import Image
        Image.open(OUT + ".png").convert("L").save(OUT + "_gray.png")
    except ImportError:
        print("PIL not available: grayscale preview skipped")
    print(f"size: {W:.2f} x {H:.2f} in  (aspect {W / H:.3f}; 16:9=1.778, 16:10=1.6)")
    print(f"column width: {CW:.3f} in")
    for w_ in WARN:
        print("WARNING:", w_)
    if not WARN:
        print("layout check: no overflow detected")


if __name__ == "__main__":
    main()
