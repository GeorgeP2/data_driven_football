"""Plain test builders shared across test modules.

Fixtures live in conftest.py; these are ordinary functions, so they need an importable module.
`conftest.py` puts this folder on sys.path.
"""

from __future__ import annotations

from typing import Any

from fpl_optimiser.backtest.history import Fixture, GameweekRecord
from fpl_optimiser.backtest.season import LaggedStats
from fpl_optimiser.expected_points import StaticProvider
from fpl_optimiser.players import Player

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def make_player(
    player_id: int,
    element_type: int,
    *,
    team_id: int = 1,
    cost_tenths: int = 40,
    ep_next: float = 0.0,
    availability: float = 1.0,
    status: str = "a",
) -> Player:
    return Player(
        id=player_id,
        name=f"p{player_id}",
        element_type=element_type,
        position=POSITIONS[element_type],
        team_id=team_id,
        team_name=f"T{team_id}",
        cost_tenths=cost_tenths,
        status=status,
        availability=availability,
        ep_next=ep_next,
        minutes=900,
        total_points=int(ep_next * 10),
    )


def build_pool(
    *,
    per_position: int = 8,
    teams: int = 20,
    cost_tenths: int = 40,
    ep_next: float = 0.0,
) -> list[Player]:
    """A uniform pool of interchangeable players — cheap filler to build scenarios on top of.

    Ids are stable and deterministic: position-major, so id 1..N are keepers, and so on.
    Players are spread across `teams` clubs so the per-club limit never binds by accident.
    """
    players: list[Player] = []
    next_id = 1
    for element_type in sorted(POSITIONS):
        for index in range(per_position):
            players.append(
                make_player(
                    next_id,
                    element_type,
                    team_id=(index % teams) + 1,
                    cost_tenths=cost_tenths,
                    ep_next=ep_next,
                )
            )
            next_id += 1
    return players


def points_provider(points_by_id: dict[int, float]) -> StaticProvider:
    return StaticProvider(points_by_player_id=points_by_id)


def bootstrap_payload() -> dict[str, Any]:
    """A minimal but structurally faithful `bootstrap-static` response."""
    return {
        "game_settings": {
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": 1000,
        },
        "element_types": [
            {
                "id": 1,
                "singular_name_short": "GKP",
                "squad_select": 2,
                "squad_min_play": 1,
                "squad_max_play": 1,
            },
            {
                "id": 2,
                "singular_name_short": "DEF",
                "squad_select": 5,
                "squad_min_play": 3,
                "squad_max_play": 5,
            },
            {
                "id": 3,
                "singular_name_short": "MID",
                "squad_select": 5,
                "squad_min_play": 2,
                "squad_max_play": 5,
            },
            {
                "id": 4,
                "singular_name_short": "FWD",
                "squad_select": 3,
                "squad_min_play": 1,
                "squad_max_play": 3,
            },
        ],
        "teams": [
            {"id": 1, "short_name": "ARS", "name": "Arsenal"},
            {"id": 2, "short_name": "LIV", "name": "Liverpool"},
        ],
        "elements": [
            {
                "id": 10,
                "web_name": "Raya",
                "element_type": 1,
                "team": 1,
                "now_cost": 62,
                "ep_next": "8.8",
                "status": "a",
                "chance_of_playing_next_round": None,
                "minutes": 3330,
                "total_points": 162,
            },
            {
                "id": 11,
                "web_name": "Salah",
                "element_type": 3,
                "team": 2,
                "now_cost": 145,
                "ep_next": "6.4",
                "status": "d",
                "chance_of_playing_next_round": 75,
                "minutes": 3100,
                "total_points": 210,
            },
            {
                "id": 12,
                "web_name": "Crocked",
                "element_type": 4,
                "team": 2,
                "now_cost": 90,
                "ep_next": "5.0",
                "status": "i",
                "chance_of_playing_next_round": None,
                "minutes": 0,
                "total_points": 0,
            },
        ],
    }


def record(
    player_id: int,
    gameweek: int,
    *,
    element_type: int = 3,
    points: int = 0,
    minutes: int = 90,
    price: int = 50,
    team: str = "ARS",
    team_id: int = 1,
    opponents: tuple[Fixture, ...] | None = None,
) -> GameweekRecord:
    if opponents is None:
        opponents = (Fixture(opponent_team_id=20, was_home=True, goals_for=1, goals_against=1),)
    return GameweekRecord(
        player_id=player_id,
        gameweek=gameweek,
        name=f"p{player_id}",
        element_type=element_type,
        team_id=team_id,
        team_name=team,
        price_tenths=price,
        total_points=points,
        minutes=minutes,
        started=1 if minutes >= 60 else 0,
        expected_goals=0.0,
        expected_assists=0.0,
        bonus=0,
        saves=0,
        defensive_contribution=0,
        yellow_cards=0,
        red_cards=0,
        bonus_points_system=0,
        expected_goals_conceded=0.0,
        fpl_expected_points=None,
        opponents=opponents,
    )


def lagged(
    player_id: int,
    *,
    points: tuple[int, ...] = (),
    minutes: tuple[int, ...] = (),
    previous_season_points: int = 0,
    previous_season_minutes: int = 0,
) -> LaggedStats:
    """A LaggedStats with only the fields a test cares about; the rest zeroed to match."""
    length = len(minutes)
    return LaggedStats(
        player_id=player_id,
        points=points,
        minutes=minutes,
        starts=tuple(1 if m >= 60 else 0 for m in minutes),
        expected_goals=(0.0,) * length,
        expected_assists=(0.0,) * length,
        bonus=(0,) * length,
        bps=(0,) * length,
        saves=(0,) * length,
        defensive_contribution=(0,) * length,
        previous_season_points=previous_season_points,
        previous_season_minutes=previous_season_minutes,
    )
