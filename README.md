# Complexity vs. Informativeness in Explainable AI — code

Code, result files, and figure sources for the paper

> H. Najafi, D. Luo, K. Bahreini, J. Liu. **Complexity vs. Informativeness in Explainable AI: Why Feature Attributions Fail and How to Diagnose the Failure.** *Journal of Artificial Intelligence Research*, 2026.

Every number, table, and figure in the paper's worked example (Section 2.5), bibliometric analysis (Section 3), and experiments E1–E10 (Section 5) is produced by a script in this repository. Each script also saves its results (`*_results.npy`, and `*_raw.npz` for per-instance attributions), so you can check the reported values without re-running anything.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper. Two experiments need extra setup:

- **E10** (SLALOM on a DistilBERT-IMDB transformer): `pip install slalom-explanations transformers datasets`. A GPU is recommended; the model and data download from the Hugging Face hub on first run.
- **E3's CALIME extension** uses the official CALIME code, which pins old package versions. Run `experiments/setup_calime.sh` to create a separate environment, then follow the instructions at the top of `experiments/e3_calime_hpc.py`.

## Running

Run the scripts from the repository root, for example:

```bash
python experiments/e2_wilming_suppressor.py
```

Each script writes its figure (`.pdf`/`.png`) and results file next to itself in `experiments/`. All randomness is seeded, so re-runs reproduce the saved results (small differences can come from different library versions or hardware).

| Paper item | Script | Output |
|---|---|---|
| Worked example (§2.5): diagnosis, method comparison | `a1_synthetic_experiment.py` | `a1_results.npy`, `a1_raw.npz`, `a1_comparison_chart.*` |
| Worked example: two ground-truth conventions (ISD vs. Shapley allocation), additive twin, E1 figure | `t1_estimand_analysis.py` (reads `a1_raw.npz`, `e1_raw.npz`, `e8_raw.npz`) | `t1_results.npy`, `t1_estimand.*`, `e1_ih_comparison.*` |
| Worked example: X1/X2 split stability | `t1_split_stability.py` | `t1_split_stability.json` |
| E1 — Integrated Hessians on the worked-example data | `e1_integrated_hessians.py` | `e1_results.npy`, `e1_raw.npz` |
| E2 — suppressor benchmark | `e2_wilming_suppressor.py` | `e2_results.npy`, `e2_suppressor_comparison.*` |
| E3 — interventional vs. observational SHAP, Haufe pattern, AEC under collinearity | `e3_collinearity_methods.py` | `e3_results.npy`, `e3_collinearity_comparison.*` |
| E3 — CALIME extension (separate environment) | `e3_calime_hpc.py`, `setup_calime.sh` | `e3_calime_results.npy`, `e3_calime.*` |
| E4 — ROAR on UCI Adult | `e4_roar_real_data.py` | `e4_results.npy`, `e4_roar_curves.*` |
| E4b — ROAR on Breast Cancer Wisconsin | `e4b_roar_breast_cancer.py` | `e4b_results.npy`, `e4b_roar_curves.*` |
| E4c — the diagnostic framework end-to-end on real data | `e4c_realdata_framework.py` | `e4c_results.npy`, `e4c_realdata_framework.*` |
| E5 — scalability | `e5_scalability.py` | `e5_results.npy`, `e5_scalability.*` |
| E6 — collinearity × interaction factorial | `e6_multidim_factorial.py` | `e6_results.npy`, `e6_factorial_heatmap.*` |
| E7 — interaction degradation | `e7_carmichael_replication.py` | `e7_results.npy`, `e7_interaction_failure.*` |
| E8 — DeepLIFT, LRP, Shapley-Taylor | `e8_deeplift_lrp_taylor.py` | `e8_results.npy`, `e8_raw.npz`, `e8_seven_methods.*` |
| E9 — neural additive model baseline | `e9_nam_glassbox.py` | `e9_results.npy`, `e9_nam_glassbox.*` |
| E10 — SLALOM vs. raw attention on a transformer | `e10_slalom_transformer.py` | `e10_results.npy`, `e10_slalom_*` |
| Failure map, NAM row: NAM credit split under collinearity | `check_nam_collinearity.py` | printed output |
| Bibliometric analysis (§3, maturation figure) | `bibliometric_pipeline.py` (reads `../ref-gem.bib`, `categorization.json`, `cited_keys.txt`) | `a2_maturation_evidence.*`, `composition_report.txt` |
| Milestone selection for the timeline | `build_milestones_from_api.py` (queries the Semantic Scholar API) | `milestones.json`, `milestones_audit.json` |

Figures in `figures/`:

| Figure | Source |
|---|---|
| Figure 1 — complexity taxonomy | `figures/make_complexity_taxonomy.py` → `complexity_taxonomy_v2.{pdf,png}` and `complexity_taxonomy_v2_links.tex` (clickable citation overlay for LaTeX) |
| Suppressor illustration | `figures/make_suppressor.py` → `suppressor_cb.{pdf,png}` |
| Figure 2 — diagnostic workflow | `figures/diagnostic_workflow.svg` (editable), converted to PDF with `cairosvg` |

## Notes on the methods

- **SHAP variants.** KernelSHAP with a background sample and `TreeExplainer(data=..., feature_perturbation="interventional")` compute interventional SHAP values. `shap` computes TreeSHAP *interaction* values only with the path-dependent algorithm, so all interaction values here are path-dependent. Each script's header says which variant it uses.
- **Re-implemented methods.** AEC, the closed-form Haufe activation pattern (the signal estimate behind PatternAttribution), exact Shapley-Taylor enumeration, and the minimal NAM baseline are implemented from the published equations. All other methods use their reference libraries (`shap`, `lime`, `captum`, `path-explain`, `scikit-learn`, the official CALIME and SLALOM code).
- **Datasets.** Synthetic data is generated by the scripts. Breast Cancer Wisconsin ships with scikit-learn; UCI Adult (OpenML) and IMDB (Hugging Face) are downloaded automatically.
- **Bibliometrics.** `categorization.json` holds the role assigned to each cited paper and the category definitions; `cited_keys.txt` lists the keys cited in the paper; `ref-gem.bib` is the paper's bibliography.

## License

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the paper (see `CITATION.cff`).
