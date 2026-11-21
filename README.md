# AI Benchmark Observatory audit - reproducibility package
Source data: Google Sheet 11Ll1x8ZDU0nObhVw2jTtOTcAZLvov66Brwk5SKGKVbM, tab Analysis_Data (chat block A2:K718, pretrained block M2:W276).
Values are the sheet's 2-decimal rounded numbers, hand-transcribed from connector output (no network in the analysis container). Transcription check: recomputed N, mean, SD, median, quartiles, skewness, kurtosis and size-tier counts match the sheet's Descriptives tab exactly.
Not used: raw tab open_llm_leaderboard (full precision, 4,576 rows) and the HF TSV (4.2 MB) could not be loaded into the analysis container.
Run order: run_audit.py -> pairs.py -> pairs2.py (reads data/*.txt; edit paths). audit_analysis_for_raw_TSV.py is a separate script for the raw TSV, tested only on synthetic data.
Original data are never modified. Matched pairs are a name-based heuristic (pairs.py) reviewed by eye; see matched_pairs_list.csv.
