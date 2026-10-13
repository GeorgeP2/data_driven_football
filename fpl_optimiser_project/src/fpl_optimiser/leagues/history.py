"""Helpers for downloading Fantasy Premier League entry history as CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def pull_fpl_history(team_id: int, output_dir: str = ".") -> tuple[Path, Path]:
    """Fetch FPL history for a team and save raw JSON + current history CSV.

    Returns:
        Tuple of (csv_path, json_path).
    """
    api_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"fpl_entry_{team_id}_history.json"
    csv_path = output_path / f"fpl_entry_{team_id}_history.csv"

    try:
        with urlopen(api_url) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"FPL API returned HTTP {exc.code} for team_id={team_id}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach FPL API for team_id={team_id}") from exc

    current_rows = payload.get("current", [])
    if not isinstance(current_rows, list):
        raise RuntimeError("Unexpected API response: 'current' is not a list")

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    if current_rows:
        headers = list(current_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(current_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    return csv_path, json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FPL team history to CSV.")
    parser.add_argument("team_id", type=int, help="FPL team/entry id")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where CSV and JSON files are written (default: current directory)",
    )
    args = parser.parse_args()

    csv_file, json_file = pull_fpl_history(args.team_id, args.output_dir)
    print(f"Wrote CSV: {csv_file}")
    print(f"Wrote JSON: {json_file}")
