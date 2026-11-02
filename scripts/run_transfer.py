"""Compatibility wrapper for the Table 4 transfer suite.

All logic lives in ``transfer_eval.py`` (protocol-matched two-stage
fine-tuning: source train -> freeze operator -> 1-battery target
adaptation, for both models).  This module simply forwards argv to it.

Run:
    python scripts/run_transfer.py --repeats 10 --out results/transfer
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transfer_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
