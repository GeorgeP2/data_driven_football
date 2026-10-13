"""Shared fixtures.

The `bootstrap-static` fixture is built as a Python dict rather than a committed JSON file, so
the test data is readable next to the tests that use it. Plain builders live in fpl_helpers.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fpl_optimiser.rules import GameRules, PositionRule

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpl_helpers import bootstrap_payload


@pytest.fixture
def rules() -> GameRules:
    """The real FPL rules, as `bootstrap-static` reports them."""
    return GameRules(
        squad_size=15,
        starting_size=11,
        club_limit=3,
        budget_tenths=1000,
        positions=(
            PositionRule(1, "GKP", squad_select=2, min_play=1, max_play=1),
            PositionRule(2, "DEF", squad_select=5, min_play=3, max_play=5),
            PositionRule(3, "MID", squad_select=5, min_play=2, max_play=5),
            PositionRule(4, "FWD", squad_select=3, min_play=1, max_play=3),
        ),
    )


@pytest.fixture
def payload() -> dict[str, Any]:
    return bootstrap_payload()
