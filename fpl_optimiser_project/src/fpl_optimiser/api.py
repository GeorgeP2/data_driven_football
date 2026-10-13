"""Fetch the FPL `bootstrap-static` payload, with an on-disk cache.

Follows the error-handling shape already used by `leagues/history.py`: a plain stdlib
`urlopen`, with `HTTPError`/`URLError` translated into a `RuntimeError` that names what failed.

`bootstrap-static` is a single ~1 MB document holding every player, team, position rule and
gameweek. It changes at most a few times a day (prices, injuries), so the cache exists to keep
iteration fast and to stay a polite consumer of a free public API.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "fpl_optimiser" / "bootstrap-static.json"

DEFAULT_CACHE_TTL_SECONDS = 6 * 60 * 60
"""Six hours. Prices and injury flags move at most a couple of times a day."""

DEFAULT_TIMEOUT_SECONDS = 30.0

_USER_AGENT = "fpl-optimiser/0.1 (+https://github.com/GeorgeP2/data_driven_football)"


def fetch_bootstrap(
    *,
    cache_path: Path | None = None,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    refresh: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return the `bootstrap-static` payload, from cache when it is still fresh.

    Args:
        cache_path: Where to read/write the cached payload. Defaults to
            `~/.cache/fpl_optimiser/bootstrap-static.json`.
        cache_ttl_seconds: Treat a cached file younger than this as fresh. Zero disables
            reading from cache (but the fetched payload is still written back).
        refresh: Ignore any cached payload and re-fetch.
        timeout_seconds: Network timeout for the API call.

    Raises:
        RuntimeError: if the API is unreachable or returns a non-200, and no usable cache
            exists to fall back on.
    """
    path = DEFAULT_CACHE_PATH if cache_path is None else cache_path

    if not refresh:
        cached = _read_cache(path, cache_ttl_seconds)
        if cached is not None:
            logger.debug("Using cached bootstrap-static from %s", path)
            return cached

    try:
        payload = _fetch(timeout_seconds)
    except RuntimeError:
        # A stale cache beats no squad at all — but say so loudly rather than silently
        # optimising against last week's prices.
        stale = _read_cache(path, cache_ttl_seconds=float("inf"))
        if stale is None:
            raise
        logger.warning(
            "FPL API unreachable; falling back to the stale cached payload at %s. "
            "Prices and injury flags may be out of date.",
            path,
        )
        return stale

    _write_cache(path, payload)
    return payload


def _fetch(timeout_seconds: float) -> dict[str, Any]:
    request = Request(BOOTSTRAP_URL, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"FPL API returned HTTP {exc.code} for {BOOTSTRAP_URL}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the FPL API at {BOOTSTRAP_URL}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FPL API returned malformed JSON from {BOOTSTRAP_URL}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected bootstrap-static response: expected an object, got "
            f"{type(payload).__name__}"
        )
    return payload


def _read_cache(path: Path, cache_ttl_seconds: float) -> dict[str, Any] | None:
    if cache_ttl_seconds <= 0 or not path.is_file():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > cache_ttl_seconds:
        return None
    try:
        with path.open(encoding="utf-8") as cache_file:
            payload: Any = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable bootstrap-static cache at %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file)
    except OSError as exc:
        # A cache we cannot write is an inconvenience, not a failure.
        logger.warning("Could not write bootstrap-static cache to %s: %s", path, exc)
