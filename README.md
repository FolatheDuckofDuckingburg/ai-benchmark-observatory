# AI Benchmark Observatory

AI Benchmark Observatory investigates whether chat-tuned language models perform differently from pretrained base models on reasoning and instruction-following benchmarks.

## Research question

**Do chat AI models perform better at complex reasoning than raw pretrained base models?**

The study compares model scores across six benchmarks using an archived Open LLM Leaderboard snapshot dated **20 March 2025**: MMLU-Pro, GPQA, MATH Level 5, BBH, MuSR, and IFEval.

This is observational research. Group differences do not establish that conversational fine-tuning caused any performance difference. Architecture, parameter count, training data, model-family composition, and evaluation conditions may also matter.

## Repository map

```text
.
├── analysis/scripts/       # Analysis and pairing scripts
├── data/processed/         # Combined/transformed data and pair lists
├── data/transcriptions/    # Text transcriptions of leaderboard rows
├── results/figures/        # Existing figures
├── results/logs/           # Existing run log
├── results/tables/         # Existing statistical outputs
├── docs/                   # Reproducibility and provenance notes
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Where to start

- Read [the reproducibility notes](docs/REPRODUCIBILITY.md) before attempting to rerun or interpret the analyses.
- See [CITATION.cff](CITATION.cff) for citation metadata.
- Install the listed Python dependencies with `python -m pip install -r requirements.txt`.
- Run the transcription-based analysis pipeline from the repository root with `python analysis/scripts/run_pipeline.py`. The command validates the source rows, regenerates the statistical tables and figures, and writes `results/logs/latest_pipeline_run.log`.

The pipeline starts from the committed transcriptions, not the original leaderboard TSV. It therefore tests reproducibility of the transcribed-data analysis, not independent verification of the transcription against the archived source snapshot.

## Analysis status and interpretation

The repository contains analysis scripts, transcribed/processed data, tables, figures, and a run log. Their presence does **not** by itself establish that every script can be rerun from this reorganized layout or that all outputs were regenerated from the archived source snapshot. Consult the reproducibility notes before interpreting any result as independently reproduced.

Do not treat exploratory matched-family comparisons or heuristic model-family matching as definitive pairing. Report the exact sample, filtering, matching, and uncertainty procedure used for each result.

## Data provenance

The research references an archived Open LLM Leaderboard snapshot dated 2025-03-20. The original snapshot, its source URL, acquisition details, and redistribution permissions should be documented separately. Files here are identified as transcriptions or processed artifacts; do not assume they are the original leaderboard export.

## Maintainer

FolatheDuckofDuckingburg

## Citation

See [CITATION.cff](CITATION.cff). The citation metadata currently points to the Hugging Face dataset record.

## License

The MIT license applies only to original materials that the researcher has the rights to license under those terms. Third-party materials retain their applicable licenses and restrictions.
