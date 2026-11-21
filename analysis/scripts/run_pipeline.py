#!/usr/bin/env python3
"""Run validation and all analysis stages from any working directory."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import platform
import numpy
import pandas
import scipy
import matplotlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "analysis" / "scripts"
LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "latest_pipeline_run.log"

COMMANDS = [
    [sys.executable, str(SCRIPTS / "validate_inputs.py")],
    [sys.executable, str(SCRIPTS / "run_audit.py")],
    [sys.executable, str(SCRIPTS / "pairs.py")],
    [sys.executable, str(SCRIPTS / "pairs2.py")],
]


def main():
    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write(f"started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"python={platform.python_version()}\n")
        log.write(f"platform={platform.platform()}\n")
        log.write(f"numpy={numpy.__version__}\n")
        log.write(f"pandas={pandas.__version__}\n")
        log.write(f"scipy={scipy.__version__}\n")
        log.write(f"matplotlib={matplotlib.__version__}\n")
        for source in (ROOT / "data" / "transcriptions" / "chat_rows.txt", ROOT / "data" / "transcriptions" / "pretrained_rows.txt"):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            log.write(f"sha256 {source.relative_to(ROOT)} {digest}\n")
        for command in COMMANDS:
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
            )
            if result.returncode:
                log.write(f"FAILED exit_code={result.returncode}\n")
                raise SystemExit(result.returncode)
        log.write("\nstatus=SUCCESS\n")
        log.write(f"finished_utc={datetime.now(timezone.utc).isoformat()}\n")
    print(f"Pipeline completed. Log: {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
