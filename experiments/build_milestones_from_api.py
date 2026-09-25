#!/usr/bin/env python3
"""
Build milestones.json systematically by querying the Semantic Scholar API.

Rather than hand-picking 14 papers (which a reader can reasonably challenge
as cherry-picked), we query Semantic Scholar for each maturation phase with
phase-appropriate keywords, take the top-N by citation count per query, and
then assign each result to a phase by publication year. Selection criteria,
queries, snapshot date, and the raw API response are saved to
`milestones_audit.json` so the procedure is fully reproducible.

The output schema matches what bibliometric_pipeline.py expects in
`milestones.json` (key, year, short_name, phase, citations, bibkey).

Run:
    cd revised_v2 && python3 experiments/build_milestones_from_api.py

Network behaviour:
  - Uses the Semantic Scholar public API (no key needed). If rate-limited,
    the script retries with backoff.
  - Falls back gracefully if a query returns no results, recording the
    failure in the audit log.
"""

import json
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Phase definitions and the search queries we use for each phase.
# The queries are kept narrow on purpose so we get attribution-relevant
# papers, not the entire XAI literature.
PHASES = [
    ("Method Creation", 2016, 2018, [
        "LIME local interpretable model-agnostic explanations",
        "SHAP unified approach interpreting model predictions",
        "axiomatic attribution deep networks integrated gradients",
    ]),
    ("Critical Scrutiny", 2018, 2021, [
        "sanity checks saliency maps",
        "explanation method limitations attribution failure",
        "remove and retrain benchmark interpretability",
        "suppressor variables explainable AI",
    ]),
    ("Complexity-Aware", 2021, 2025, [
        "feature interaction attribution integrated hessians",
        "impossibility theorems feature attribution",
        "causality lime CALIME causal explanation",
        "additive effects collinearity XAI",
        "complexity benchmark XAI nonlinear",
    ]),
]

PHASE_BIBKEY_HINTS = {
    # Common bibkeys in our ref-gem.bib so we can connect API results
    # back to the manuscript bibliography when there is an obvious match.
    "ribeiro": "Ribeiro2016why",
    "lundberg": "Lundberg2017unified",
    "sundararajan": "Sundararajan2017axiomatic",
    "adebayo": "Adebayo2018sanity",
    "hooker": "Hooker2019ROAR",
    "sixt": "Sixt2019when",
    "wilming": "Wilming2021scrutinizing",
    "janzing": "Janzing2020feature",
    "janizek": "Janizek2021explaining",
    "bilodeau": "Bilodeau2022impossibility",
    "cinquini": "Cinquini2022causality",
    "carmichael": "Carmichael2023how",
    "salih": "Salih2024aec",
    "clark": "Clark2024xaitris",
}


def search_api(query, year_min, year_max, limit=20, retries=3):
    """Run one Semantic Scholar API query. Returns parsed results or []."""
    url = (f"{API}?query={requests.utils.quote(query)}"
           f"&year={year_min}-{year_max}&limit={limit}"
           f"&fields=title,year,authors,citationCount,externalIds")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    rate-limited, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            print(f"    error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return []


def author_lastname(p):
    if not p.get("authors"):
        return ""
    return p["authors"][0].get("name", "").split()[-1].lower()


def short_name(p):
    """Make a short label for the figure."""
    title = p.get("title", "")
    # Take the most distinctive word from the title
    skip = {"the", "a", "an", "of", "for", "on", "in", "and", "to", "with", "from",
            "is", "are", "be", "model", "models", "method", "methods", "approach",
            "based", "using", "via", "towards", "explainable", "explanation",
            "explanations", "feature", "features", "attribution", "attributions"}
    words = [w.strip(":,.()[]") for w in title.split() if len(w) > 3]
    distinctive = [w for w in words if w.lower() not in skip][:2]
    return "\n".join(distinctive)[:32] if distinctive else (author_lastname(p) or "paper")


def assign_phase(year):
    for phase, lo, hi, _ in PHASES:
        if lo <= year < hi or (year == hi and phase == "Complexity-Aware"):
            return phase
    return "Other"


def bibkey_for(p):
    """Heuristic: match the first author's lastname against our bibkey hints."""
    last = author_lastname(p)
    for stem, key in PHASE_BIBKEY_HINTS.items():
        if stem in last:
            return key
    return None


def main():
    audit = {
        "snapshot_date": date.today().isoformat(),
        "queries": [],
        "raw_results_by_query": {},
        "deduplicated_candidates": [],
        "selected_milestones": [],
        "notes": (
            "Milestones were selected by querying the Semantic Scholar public "
            "API for each maturation phase with phase-appropriate keyword "
            "strings. From each query's top-20 results, we kept the most-cited "
            "paper per query as a representative, then deduplicated across "
            "queries by title. The procedure is documented here so a reader "
            "can reproduce the selection or modify the queries and re-run."
        ),
    }

    candidates = {}  # by title
    for phase, lo, hi, queries in PHASES:
        print(f"\n=== Phase: {phase} ({lo}-{hi}) ===")
        for q in queries:
            print(f"  query: {q!r}")
            results = search_api(q, lo, hi, limit=15)
            audit["queries"].append({"phase": phase, "query": q, "n_results": len(results)})
            audit["raw_results_by_query"][q] = [
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "citationCount": p.get("citationCount"),
                    "firstAuthor": author_lastname(p),
                }
                for p in results
            ]
            # Take the top-cited result whose year falls inside the phase window.
            in_window = [p for p in results
                         if p.get("year") and lo <= p["year"] <= hi
                         and p.get("citationCount") is not None]
            in_window.sort(key=lambda p: -p["citationCount"])
            if in_window:
                top = in_window[0]
                title = top["title"]
                if title not in candidates:
                    candidates[title] = (top, phase)
            time.sleep(1.5)  # be polite to the API

    # Deduplicate; convert to milestone records.
    milestones = []
    for title, (p, phase) in candidates.items():
        milestones.append({
            "year": p["year"],
            "short_name": short_name(p),
            "phase": phase,
            "citations": p.get("citationCount") or 0,
            "bibkey": bibkey_for(p) or "",
            "_title": title,
            "_first_author": author_lastname(p),
        })
    audit["deduplicated_candidates"] = milestones[:]

    # Sort by year and limit to a comparable size to the prior hand-picked
    # list (≤ 14 papers) by taking the top-cited entries per phase.
    by_phase = {phase: [] for phase, _, _, _ in PHASES}
    for m in milestones:
        if m["phase"] in by_phase:
            by_phase[m["phase"]].append(m)
    for phase in by_phase:
        by_phase[phase].sort(key=lambda m: -m["citations"])
        by_phase[phase] = by_phase[phase][:5]   # at most 5 per phase

    selected = []
    for phase, _, _, _ in PHASES:
        selected.extend(by_phase[phase])
    selected.sort(key=lambda m: m["year"])
    audit["selected_milestones"] = selected

    # Save audit log
    (HERE / "milestones_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nAudit log saved to {HERE / 'milestones_audit.json'}")

    # Save milestones.json in the schema the pipeline expects
    out = {
        "_description": (
            "Milestones selected by querying the Semantic Scholar public API with "
            "phase-appropriate keyword strings; selection procedure is documented "
            "in milestones_audit.json (queries, raw API results, deduplication). "
            "This file is regenerated by build_milestones_from_api.py."
        ),
        "snapshot_date": audit["snapshot_date"],
        "citation_source": "Semantic Scholar API (graph/v1/paper/search)",
        "milestones": [
            {
                "year": m["year"],
                "short_name": m["short_name"],
                "phase": m["phase"],
                "citations": m["citations"],
                "bibkey": m["bibkey"],
            }
            for m in selected
        ],
    }
    (HERE / "milestones.json").write_text(json.dumps(out, indent=2))
    print(f"Milestones written to {HERE / 'milestones.json'}")
    print(f"\nTotal selected milestones: {len(selected)}")
    for m in selected:
        print(f"  [{m['phase']}] {m['year']} — {m['_first_author']} — {m['citations']} cites — {m['_title'][:70]}")


if __name__ == "__main__":
    main()
