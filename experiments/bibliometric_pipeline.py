#!/usr/bin/env python3
"""
Bibliometric pipeline for the maturation-evidence figure (Section 3).

Reads:
  - ../ref-gem.bib          (BibTeX database; supplies per-key publication years)
  - ../camera-ready.tex     (manuscript; we extract the cited keys)
  - categorization.json     (manual per-key category assignment)
  - milestones.json         (curated list of 14 milestone papers with citation snapshot)

Writes:
  - a2_maturation_evidence.pdf / .png   (Figure 4 in the manuscript)
  - composition_report.txt              (audit log: which key went into which bucket,
                                         counts, percentages, missing keys, surplus keys)

Figure layout (revised for the JAIR resubmission):
  The two SYSTEMATIC panels, both derived from the full categorized bibliography,
  now lead the figure:
    (a) Composition of the bibliography by paper role (pie).
    (b) Papers per year by category (stacked bars) -- the publication-trend view.
    (c) Cumulative papers by category over time (lines) -- makes the
        method-proposal plateau vs. critique/complexity-aware growth explicit.
  The EXEMPLAR milestone timeline is demoted to a single panel:
    (d) Milestone timeline (curated exemplar set, not a systematic enumeration).

Reproducibility:
  Every number in panels (a)-(c) is derivable from ref-gem.bib + camera-ready.tex +
  categorization.json. Panel (d) uses the curated milestone snapshot in
  milestones.json (citation counts are a fixed snapshot; date recorded inside).

Run from the repository root or the experiments/ folder:
    cd revised_v2 && python experiments/bibliometric_pipeline.py
"""

import json
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BIB_PATH = ROOT / "ref-gem.bib"
TEX_PATH = ROOT / "camera-ready.tex"
CITED_PATH = HERE / "cited_keys.txt"   # snapshot used when the manuscript source is absent
CATEG_PATH = HERE / "categorization.json"
MILESTONES_PATH = HERE / "milestones.json"
OUT_PDF = HERE / "a2_maturation_evidence.pdf"
OUT_PNG = HERE / "a2_maturation_evidence.png"
REPORT_PATH = HERE / "composition_report.txt"

# First XAI-era year shown in the temporal panels (b, c). Foundational theory
# citations that predate this window (e.g., causality/information-theory texts)
# are still counted in the composition panel (a) but would stretch the time
# axis; we note how many are excluded from (b)/(c) in the audit log and caption.
YEAR_MIN = 2014

# Map the seven fine-grained categories onto four trend buckets used in the
# temporal panels. The three "argument" buckets (method / scrutiny /
# complexity-aware) carry the maturation thesis; surveys+practitioner are shown
# for completeness so the per-year bars sum to the true paper counts.
BUCKET_OF = {
    "method": "Method proposals",
    "critique": "Critique / theory / benchmarks",
    "theory": "Critique / theory / benchmarks",
    "benchmark": "Critique / theory / benchmarks",
    "complexity_aware": "Complexity-aware methods",
    "survey": "Surveys / practitioner",
    "practitioner": "Surveys / practitioner",
}
BUCKET_ORDER = [
    "Method proposals",
    "Critique / theory / benchmarks",
    "Complexity-aware methods",
    "Surveys / practitioner",
]
# Color-blind-safe (Okabe-Ito) palette; each bucket also gets a redundant hatch
# (bars) and marker+linestyle (lines) so the panels read in grayscale and for
# color-blind readers without relying on hue.
BUCKET_COLORS = {
    "Method proposals": "#0072B2",            # blue
    "Critique / theory / benchmarks": "#D55E00",  # vermillion
    "Complexity-aware methods": "#009E73",    # bluish green
    "Surveys / practitioner": "#999999",      # gray
}
BUCKET_HATCH = {
    "Method proposals": "//",
    "Critique / theory / benchmarks": "..",
    "Complexity-aware methods": "xx",
    "Surveys / practitioner": "",
}
BUCKET_MARKER = {
    "Method proposals": "o",
    "Critique / theory / benchmarks": "s",
    "Complexity-aware methods": "^",
    "Surveys / practitioner": "D",
}
BUCKET_LS = {
    "Method proposals": "-",
    "Critique / theory / benchmarks": "--",
    "Complexity-aware methods": ":",
    "Surveys / practitioner": "-.",
}
# The three buckets that carry the maturation argument (emphasised in panel c).
CORE_BUCKETS = [
    "Method proposals",
    "Critique / theory / benchmarks",
    "Complexity-aware methods",
]


def parse_bib_keys(path: Path) -> set[str]:
    """Return all entry keys defined in a BibTeX file."""
    text = path.read_text(encoding="utf-8")
    keys = re.findall(r"^@[a-zA-Z]+\s*\{\s*([^,\s]+)", text, flags=re.MULTILINE)
    return set(keys)


def parse_bib_years(path: Path) -> dict:
    """Return {entry_key: int(year)} for every entry with a parseable year."""
    text = path.read_text(encoding="utf-8")
    years = {}
    # Split into entries on the @type{ key, boundary.
    entries = re.split(r"(?=^@[a-zA-Z]+\s*\{)", text, flags=re.MULTILINE)
    for entry in entries:
        m_key = re.match(r"^@[a-zA-Z]+\s*\{\s*([^,\s]+)", entry)
        if not m_key:
            continue
        key = m_key.group(1)
        m_year = re.search(r"year\s*=\s*\{?\s*(\d{4})", entry)
        if m_year:
            years[key] = int(m_year.group(1))
    return years


def parse_cited_keys(path: Path) -> set[str]:
    """Return all citation keys referenced in a LaTeX file (any \\cite* variant)."""
    text = path.read_text(encoding="utf-8")
    cited = set()
    for match in re.finditer(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                cited.add(key)
    return cited


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def categorize(cited: set[str], categ: dict) -> tuple[OrderedDict, list, list]:
    """Apply categorization to the cited set; return (counts, uncategorized, surplus)."""
    counts = OrderedDict([
        ("method", 0),
        ("critique", 0),
        ("complexity_aware", 0),
        ("benchmark", 0),
        ("theory", 0),
        ("survey", 0),
        ("practitioner", 0),
    ])
    assignments = categ["assignments"]
    uncategorized = []
    for key in sorted(cited):
        cat = assignments.get(key)
        if cat is None:
            uncategorized.append(key)
            continue
        if cat not in counts:
            raise ValueError(f"Unknown category '{cat}' for {key}")
        counts[cat] += 1
    surplus = sorted(set(assignments) - cited)
    return counts, uncategorized, surplus


def build_temporal(cited: set[str], categ: dict, years: dict):
    """Return (year_grid, per_year, cumulative, n_pre, n_no_year) for the trend panels.

    per_year[bucket]  -> np.array aligned with year_grid (count published that year)
    cumulative[bucket]-> np.array aligned with year_grid (running total through that year)
    """
    assignments = categ["assignments"]
    points = []  # (year, bucket)
    n_pre = 0
    n_no_year = 0
    for key in cited:
        cat = assignments.get(key)
        if cat is None:
            continue
        bucket = BUCKET_OF[cat]
        yr = years.get(key)
        if yr is None:
            n_no_year += 1
            continue
        if yr < YEAR_MIN:
            n_pre += 1
            continue
        points.append((yr, bucket))

    if points:
        y_max = max(y for y, _ in points)
    else:
        y_max = YEAR_MIN
    year_grid = list(range(YEAR_MIN, y_max + 1))

    per_year = {b: np.zeros(len(year_grid), dtype=int) for b in BUCKET_ORDER}
    for yr, bucket in points:
        per_year[bucket][year_grid.index(yr)] += 1
    cumulative = {b: np.cumsum(per_year[b]) for b in BUCKET_ORDER}
    return year_grid, per_year, cumulative, n_pre, n_no_year


def write_report(counts, uncategorized, surplus, total_cited, total_defined,
                 categ, n_pre, n_no_year):
    """Write a plain-text audit log."""
    lines = []
    lines.append("=" * 72)
    lines.append("BIBLIOMETRIC COMPOSITION AUDIT (Section 3, Figure 4)")
    lines.append("=" * 72)
    lines.append(f"Snapshot date: {categ.get('_snapshot_date')}")
    lines.append(f"Total unique cited keys in camera-ready.tex: {total_cited}")
    lines.append(f"Total unique keys defined in ref-gem.bib:  {total_defined}")
    lines.append("")
    lines.append("Composition (counts and percentages of cited keys):")
    n = sum(counts.values())
    for cat, c in counts.items():
        pct = 100.0 * c / n if n else 0.0
        lines.append(f"  {cat:>18s}: {c:3d}  ({pct:5.1f}%)")
    lines.append(f"  {'TOTAL':>18s}: {n:3d}")
    crit_bench_complex = counts["critique"] + counts["benchmark"] + counts["complexity_aware"]
    pct_combined = 100.0 * crit_bench_complex / n if n else 0.0
    pct_method = 100.0 * counts["method"] / n if n else 0.0
    lines.append("")
    lines.append("Key contrast for the maturation argument:")
    lines.append(f"  critique + benchmark + complexity-aware = {crit_bench_complex}/{n}  ({pct_combined:.1f}%)")
    lines.append(f"  method proposals                         = {counts['method']}/{n}  ({pct_method:.1f}%)")
    lines.append("")
    lines.append("Temporal panels (b, c) coverage:")
    lines.append(f"  XAI-era window shown: {YEAR_MIN}-present")
    lines.append(f"  cited+categorized keys before {YEAR_MIN} (in pie, excluded from b/c): {n_pre}")
    lines.append(f"  cited+categorized keys with no parseable year:                 {n_no_year}")
    lines.append("")
    if uncategorized:
        lines.append(f"WARNING: {len(uncategorized)} cited keys lack a category (extend categorization.json):")
        for k in uncategorized:
            lines.append(f"  - {k}")
    else:
        lines.append("All cited keys are categorized.")
    lines.append("")
    if surplus:
        lines.append(f"Note: {len(surplus)} categorized keys are no longer cited (safe to prune):")
        for k in surplus[:20]:
            lines.append(f"  - {k}")
        if len(surplus) > 20:
            lines.append(f"  ... and {len(surplus) - 20} more")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_figure(counts, milestones, snapshot_date, year_grid, per_year,
                cumulative, n_pre):
    # Three systematic panels only. The former panel (d) -- the curated
    # milestone timeline -- was removed for the JAIR resubmission because it
    # duplicated the standalone phase-timeline figure; the milestone narrative
    # now lives there. The drawing code for (d) is retained (commented out
    # below) so it can be restored if ever needed.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    plt.rcParams.update({"font.size": 10.5})

    total_cited = sum(counts.values())

    # ── (a) [LEAD] Bibliography composition (systematic, full bibliography) ──
    ax = axes[0]
    label_map = OrderedDict([
        ("method", "Method\nproposals"),
        ("critique", "Critique /\nfailure analysis"),
        ("complexity_aware", "Complexity-\naware methods"),
        ("benchmark", "Benchmarks /\nevaluation"),
        ("theory", "Theory\nfoundations"),
        ("survey", "Surveys /\nposition"),
        ("practitioner", "Practitioner /\nadoption"),
    ])
    cat_keys = [k for k in label_map]
    cat_names = [label_map[k] for k in cat_keys]
    cat_counts = [counts[k] for k in cat_keys]
    # Color-blind-safe wedge colors + a redundant hatch per wedge so the
    # composition is readable in grayscale (wedges are also directly labelled).
    cat_colors_pie = [
        "#0072B2", "#D55E00", "#009E73", "#E69F00",
        "#CC79A7", "#999999", "#56B4E9",
    ]
    cat_hatches_pie = ["//", "..", "xx", "\\\\", "||", "", "--"]
    wedges, texts, autotexts = ax.pie(
        cat_counts,
        labels=cat_names,
        autopct="%1.1f%%",
        colors=cat_colors_pie[: len(cat_names)],
        startangle=90,
        textprops={"fontsize": 8.5},
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    )
    for w, h in zip(wedges, cat_hatches_pie[: len(wedges)]):
        w.set_hatch(h)
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(8.5)
    ax.set_title(
        f"(a) Bibliography composition\n(N = {total_cited} cited keys; systematic)",
        fontweight="bold",
        fontsize=10.5,
    )

    # ── (b) [NEW] Papers per year by category (stacked bars, systematic) ────
    ax = axes[1]
    bottom = np.zeros(len(year_grid))
    for b in BUCKET_ORDER:
        vals = per_year[b]
        ax.bar(year_grid, vals, bottom=bottom, color=BUCKET_COLORS[b],
               label=b, edgecolor="white", linewidth=0.5, alpha=0.95,
               hatch=BUCKET_HATCH[b])
        bottom += vals
    ax.set_xlabel("Publication year", fontsize=10)
    ax.set_ylabel("Cited papers published that year", fontsize=10)
    ax.set_title(
        "(b) Cited papers per year by role\n"
        "(from 2019, critique & complexity-aware work outnumber method proposals)",
        fontweight="bold", fontsize=10.5,
    )
    ax.set_xticks(year_grid)
    ax.set_xticklabels([str(y) for y in year_grid], rotation=45, fontsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── (c) [NEW] Cumulative papers by category over time (lines) ──────────
    ax = axes[2]
    for b in CORE_BUCKETS:
        ax.plot(year_grid, cumulative[b], marker=BUCKET_MARKER[b], markersize=6,
                linewidth=2.4, linestyle=BUCKET_LS[b],
                color=BUCKET_COLORS[b], label=b)
    # surveys/practitioner as a faint reference line
    ax.plot(year_grid, cumulative["Surveys / practitioner"],
            marker=BUCKET_MARKER["Surveys / practitioner"], markersize=5,
            linewidth=1.2, linestyle=BUCKET_LS["Surveys / practitioner"],
            color=BUCKET_COLORS["Surveys / practitioner"],
            alpha=0.8, label="Surveys / practitioner")
    ax.set_xlabel("Publication year", fontsize=10)
    ax.set_ylabel("Cumulative cited papers", fontsize=10)
    ax.set_title(
        "(c) Cumulative cited papers by role\n"
        "(method proposals plateau; scrutiny & complexity-aware climb)",
        fontweight="bold", fontsize=10.5,
    )
    ax.set_xticks(year_grid)
    ax.set_xticklabels([str(y) for y in year_grid], rotation=45, fontsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    # The pre-YEAR_MIN / undated note is stated in the figure caption; it is
    # no longer drawn in-panel because it overlapped the complexity-aware curve.

    # ── (d) REMOVED for the JAIR resubmission ──────────────────────────────
    # The curated milestone timeline previously shown here duplicated the
    # standalone phase-timeline figure (the "evolution of complexity-aware XAI"
    # figure in the manuscript), so it was dropped to remove redundancy and
    # lead with the three systematic panels. The drawing code is preserved in
    # version control; `milestones`/`snapshot_date` are retained as arguments
    # only so the audit-report path and call site are unchanged.

    plt.tight_layout(pad=2.5)
    plt.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to:\n  {OUT_PDF}\n  {OUT_PNG}")


def main():
    defined_keys = parse_bib_keys(BIB_PATH)
    bib_years = parse_bib_years(BIB_PATH)
    if TEX_PATH.exists():
        cited_keys = parse_cited_keys(TEX_PATH)
        CITED_PATH.write_text("\n".join(sorted(cited_keys)) + "\n", encoding="utf-8")
    else:
        cited_keys = set(CITED_PATH.read_text(encoding="utf-8").split())
    categ = load_json(CATEG_PATH)
    milestones_data = load_json(MILESTONES_PATH)
    milestones = milestones_data["milestones"]
    snapshot = milestones_data.get("snapshot_date", categ.get("_snapshot_date", "n.d."))

    counts, uncategorized, surplus = categorize(cited_keys, categ)
    year_grid, per_year, cumulative, n_pre, n_no_year = build_temporal(
        cited_keys, categ, bib_years
    )
    write_report(
        counts, uncategorized, surplus,
        total_cited=len(cited_keys),
        total_defined=len(defined_keys),
        categ=categ, n_pre=n_pre, n_no_year=n_no_year,
    )
    plot_figure(counts, milestones, snapshot, year_grid, per_year, cumulative, n_pre)

    if uncategorized:
        print(
            f"\nERROR: {len(uncategorized)} cited keys are not categorized. "
            "Extend categorization.json before publishing.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
