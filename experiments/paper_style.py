"""Print-size matplotlib style for the paper's experiment figures (E1-E10).

Each figure is drawn at exactly the width it is printed at in the manuscript
(JAIR text width = 452.3pt = 6.27in), so fonts set here are the sizes that
appear on paper: 8pt labels, 7pt ticks/legends/annotations.

Usage in a plotting function:
    from paper_style import apply_style, fig_size
    apply_style()
    fig, ax = plt.subplots(figsize=fig_size(0.65, 0.62))
"""
import matplotlib as mpl

TEXTWIDTH_IN = 452.295 / 72.27  # \textwidth of jair.cls, in inches


def fig_size(frac, aspect=0.62):
    """(width, height) in inches for a figure printed at `frac` x \\textwidth."""
    w = TEXTWIDTH_IN * frac
    return (w, w * aspect)


def apply_style():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "figure.titlesize": 8,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
