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

## Data handling

The repository includes transcribed text files, a combined CSV, derived statistical CSVs, and figures. Preserve the distinction between the original leaderboard snapshot, transcribed or manually assembled source rows, cleaned/combined analysis data, and derived tables/visualizations.

Record the source URL, snapshot date, retrieval date, file hashes, transformation steps, and redistribution terms for any source data added later. Do not silently replace a transcription with a newer leaderboard export.

## Statistical interpretation

The comparison is observational, not a randomized experiment. Chat and pretrained groups may differ in architecture, parameter count, training data, model family, and evaluation coverage. Welch tests, effect sizes, matching heuristics, size adjustment, and cluster bootstrap analyses answer different questions and rely on different assumptions. Label exploratory analyses as exploratory and report sample sizes, exclusions, matching rules, confidence intervals, and multiple-comparison corrections.

A model-name normalization heuristic is not proof that two checkpoints share the same base model. Validate pairings against authoritative model cards or source metadata before treating them as matched pairs.

## Reproduction checklist

- [ ] Identify and archive the exact 2025-03-20 source snapshot and its terms.
- [ ] Verify each transcription against the source export.
- [ ] Document all exclusions, duplicate handling, and group classification rules.
- [ ] Reconcile script input/output paths with the repository layout.
- [ ] Pin Python and package versions.
- [ ] Run the complete pipeline from a clean environment.
- [ ] Regenerate every reported table and figure.
- [ ] Compare regenerated outputs with committed artifacts and explain any differences.
- [ ] Record the command, environment, source hashes, date, and exit status in a new run log.

Until this checklist is completed, describe existing results as repository-reported outputs rather than independently reproduced findings.
