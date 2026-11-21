# Reproducibility and provenance notes

## Scope

This repository compares chat-tuned and pretrained model entries across MMLU-Pro, GPQA, MATH Level 5, BBH, MuSR, and IFEval using an archived Open LLM Leaderboard snapshot dated 2025-03-20.

## Important status distinction

- The presence of a script, table, plot, or run log is evidence that the artifact exists in the repository.
- It is not proof that the current script was run successfully against the original snapshot.
- The header of `audit_analysis_for_raw_TSV.py` explicitly says it was smoke-tested on synthetic data only and had not been run on the real snapshot in the session that produced it.
- The analysis scripts have differing input schemas and working-directory assumptions. Some refer to paths such as `data/chat.txt`, `data/pretrained.txt`, or `out/`; these are not all present in the repository tree. Treat them as historical analysis scripts until their inputs and paths are reconciled and a clean end-to-end run is recorded.

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
