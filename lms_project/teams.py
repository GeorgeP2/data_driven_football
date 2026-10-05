"""Shared team-name normalisation: football-data.co.uk name -> canonical (FPL-style) name.
Extend if fetch_data.py prints "unmatched" warnings."""

NAME_MAP = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
}


def canon(name) -> str:
    name = str(name).strip()
    return NAME_MAP.get(name, name)
