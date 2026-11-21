#!/usr/bin/env python3
"""Validate the committed pipe-delimited source transcriptions before analysis."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "chat": ROOT / "data" / "transcriptions" / "chat_rows.txt",
    "pre": ROOT / "data" / "transcriptions" / "pretrained_rows.txt",
}
EXPECTED_FIELDS = 8  # model id, parameter count, six benchmark scores


def validate_file(group, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {group} transcription: {path}")
    rows = []
    unknown_parameter_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != EXPECTED_FIELDS:
                raise ValueError(
                    f"{path}:{line_number}: expected {EXPECTED_FIELDS} fields, got {len(row)}"
                )
            model_id = row[0].strip()
            if not model_id or model_id != row[0]:
                raise ValueError(f"{path}:{line_number}: empty or whitespace-padded model id")
            try:
                values = [float(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite numeric value")
            # The source transcription uses -1.00 and 0.00 as non-positive
            # parameter-size placeholders. Preserve these rows for score-only
            # analyses; the size-adjusted analysis excludes params <= 0.
            if values[0] <= 0:
                unknown_parameter_rows += 1
            if values[0] < 0 and values[0] != -1:
                raise ValueError(f"{path}:{line_number}: unsupported negative parameter-count sentinel")
            if any(value < 0 or value > 100 for value in values[1:]):
                raise ValueError(f"{path}:{line_number}: benchmark score outside [0, 100]")
            rows.append(model_id)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    duplicates = len(rows) - len(set(rows))
    print(f"{group}: {len(rows)} rows; {len(set(rows))} distinct model IDs; {duplicates} duplicate-ID rows; {unknown_parameter_rows} unknown-size rows")
    return len(rows)


def main():
    counts = {group: validate_file(group, path) for group, path in INPUTS.items()}
    print(f"Validated {sum(counts.values())} source rows across {len(counts)} groups.")


if __name__ == "__main__":
    main()
