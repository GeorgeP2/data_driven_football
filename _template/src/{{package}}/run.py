"""Entry point for {{title}}.

Run from the project folder: ``PYTHONPATH=src python -m {{package}}.run``
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="seed for anything random")
    ap.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    # TODO: load data from DATA, run the analysis, write results to OUTPUTS
    print(f"Nothing to run yet; results will go to {OUTPUTS}")


if __name__ == "__main__":
    main()
