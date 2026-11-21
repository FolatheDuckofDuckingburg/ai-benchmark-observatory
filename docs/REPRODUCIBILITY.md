# Reproducibility and provenance notes

## Scope

This repository compares chat-tuned and pretrained model entries across MMLU-Pro, GPQA, MATH Level 5, BBH, MuSR, and IFEval using an archived Open LLM Leaderboard snapshot dated 2025-03-20.

## Important status distinction

- The presence of a script, table, plot, or run log is evidence that the artifact exists in the repository.
- It is not proof that the current script was run successfully against the original snapshot.
- The header of `audit_analysis_for_raw_TSV.py` explicitly says it was smoke-tested on synthetic data only and had not been run on the real snapshot in the session that produced it.
- The analysis scripts now resolve input and output paths relative to the repository root. `analysis/scripts/run_pipeline.py` runs source-row validation, the primary analyses, heuristic pairing, and figure generation in sequence. The workflow in `.github/workflows/analysis.yml` runs this pipeline on pushes and pull requests.
- The pipeline consumes `data/transcriptions/chat_rows.txt` and `data/transcriptions/pretrained_rows.txt`. It does not independently verify those transcriptions against the original 2025-03-20 leaderboard export. A successful pipeline run therefore establishes computational reproducibility from the committed transcriptions, not source-data provenance or transcription accuracy.
- The pipeline writes regenerated tables to `results/tables/`, figures to `results/figures/`, and a fresh execution log to `results/logs/latest_pipeline_run.log`. The existing `results/logs/run_log.txt` is a historical artifact and is not overwritten.

## Current automated run

The GitHub Actions run on 2026-11-21 completed successfully on Python 3.11.18 with NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, and Matplotlib 3.11.2. The validator found 717 chat rows and 275 pretrained rows. It also found one chat row with a parameter-count placeholder of -1.00 and one pretrained row with a placeholder of 0.00; these rows remain in score-only comparisons but are excluded from analyses requiring positive parameter counts. The pipeline log records source-transcription SHA-256 hashes and dependency versions.

## Regenerated-output comparison

A CI run on 2026-11-21 regenerated all tables and figures. Byte-for-byte Git blob comparisons show that `table_rows.csv`, `table_consolidated.csv`, `size_strata.csv`, `ols_cluster.csv`, and `matched_pairs_results.csv` match their committed versions. The regenerated `sensitivity.csv` differs at the file level and still needs a cell-level review before it is promoted. The regenerated `cluster_bootstrap.csv` intentionally differs because the prior code obtained each confidence-interval endpoint from separate bootstrap runs; the committed CSV has now been updated to the corrected single-run-per-benchmark intervals. Figures were regenerated in the CI artifact but have not yet been compared and promoted to the repository. The CI artifact is therefore the complete record of the regenerated run; do not assume every committed figure is already current.

## Data handling

The repository includes transcribed text files, a combined CSV, derived statistical CSVs, and figures. Preserve the distinction between the original leaderboard snapshot, transcribed or manually assembled source rows, cleaned/combined analysis data, and derived tables/visualizations.

Record the source URL, snapshot date, retrieval date, file hashes, transformation steps, and redistribution terms for any source data added later. Do not silently replace a transcription with a newer leaderboard export.

## Statistical interpretation

The comparison is observational, not a randomized experiment. Chat and pretrained groups may differ in architecture, parameter count, training data, model family, and evaluation coverage. Welch tests, effect sizes, matching heuristics, size adjustment, and cluster bootstrap analyses answer different questions and rely on different assumptions. Label exploratory analyses as exploratory and report sample sizes, exclusions, matching rules, confidence intervals, and multiple-comparison corrections.

A model-name normalization heuristic is not proof that two checkpoints share the same base model. Validate pairings against authoritative model cards or source metadata before treating them as matched pairs. The current matched-pair analysis contains 81 heuristic pairs across 18 repository organizations. Its paired t-test and Wilcoxon p-values operate at the pair level and do not account for multiple pairs from the same organization; the organization-cluster bootstrap intervals do account for that grouping. Treat those pair-level p-values as exploratory until cluster-aware inference is added.

## Reproduction checklist

- [ ] Identify and archive the exact 2025-03-20 source snapshot and its terms.
- [ ] Verify each transcription against the source export.
- [ ] Document all exclusions, duplicate handling, and group classification rules.
- [x] Reconcile script input/output paths with the repository layout.
- [x] Pin the direct Python dependencies and record their versions in the run log.
- [x] Run the complete pipeline from a clean GitHub Actions environment.
- [x] Regenerate every reported table and figure in the CI artifact.
- [ ] Compare regenerated outputs with committed artifacts and explain any differences.
- [x] Record the command, environment, source hashes, date, and exit status in the CI run log artifact.

Until this checklist is completed, describe existing results as repository-reported outputs rather than independently reproduced findings.
