"""Shared helpers for football-data.co.uk files."""
import warnings

import pandas as pd

# football-data name -> canonical (FPL-style) name. Extend if fetch_data.py prints "unmatched" warnings.
NAME_MAP = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
}


def canon(name) -> str:
    name = str(name).strip()
    return NAME_MAP.get(name, name)


def read_fd_csv(src) -> pd.DataFrame:
    """Read a football-data CSV. Some seasons (e.g. 1999/00, 2003/04, 2004/05) have rows with
    trailing empty fields; keep those matches (pandas drops the extra fields) instead of skipping them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning)
        return pd.read_csv(src, encoding_errors="replace", engine="python", on_bad_lines=lambda row: row)
