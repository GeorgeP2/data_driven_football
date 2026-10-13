"""Parse `bootstrap-static` players into a typed pool the optimiser can consume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_optimiser.rules import GameRules

_UNAVAILABLE_STATUSES = frozenset({"i", "s", "u"})
"""Injured, suspended, unavailable. `a` is available and `d` is doubtful."""


@dataclass(frozen=True)
class Player:
    """A single selectable player."""

    id: int
    name: str
    element_type: int
    position: str
    team_id: int
    team_name: str
    cost_tenths: int
    """Price in tenths of a million, as the API reports it (55 = £5.5m).

    Kept as an integer throughout: the budget constraint is exact in integer arithmetic and
    would be subject to float drift otherwise.
    """

    status: str
    availability: float
    """Probability of featuring, in 0.0..1.0, derived from `status` and
    `chance_of_playing_next_round`."""

    ep_next: float
    """FPL's own expected points for the next gameweek (the API returns it as a string)."""

    minutes: int
    total_points: int

    @property
    def cost_millions(self) -> float:
        return self.cost_tenths / 10


def availability_from(status: str, chance_of_playing: int | None) -> float:
    """Convert FPL's availability flags into a 0..1 multiplier.

    FPL reports availability two ways and they disagree often enough to matter. `status` is a
    coarse flag; `chance_of_playing_next_round` is a percentage that is frequently `null` even
    for fit players. The rule here:

    - an explicit percentage always wins, whatever the status says;
    - otherwise, a hard-unavailable status (injured/suspended/unavailable) is zero;
    - otherwise (available, or doubtful with no percentage), assume they play.

    Discounting rather than hard-filtering keeps a 75%-fit premium striker selectable at a
    75% price in expected points, which is the decision a manager actually faces.
    """
    if chance_of_playing is not None:
        return _clamp_unit(chance_of_playing / 100)
    if status in _UNAVAILABLE_STATUSES:
        return 0.0
    return 1.0


def parse_players(payload: dict[str, Any], rules: GameRules) -> list[Player]:
    """Build the player pool from a `bootstrap-static` payload.

    Raises:
        ValueError: if the payload is missing `elements`/`teams`, or a player carries a field
            the model cannot interpret.
    """
    elements = payload.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("bootstrap-static payload has no 'elements' (players) list to select from")

    team_names = _team_names(payload)
    valid_types = set(rules.element_types)

    players: list[Player] = []
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError(f"Expected each element to be an object, got {element!r}")

        player_id = _require_int(element, "id")
        element_type = _require_int(element, "element_type")
        if element_type not in valid_types:
            expected = ", ".join(str(t) for t in sorted(valid_types))
            raise ValueError(
                f"Player {player_id} has element_type {element_type}, which is not one of "
                f"the positions declared in element_types: {expected}"
            )

        team_id = _require_int(element, "team")
        if team_id not in team_names:
            available = ", ".join(str(t) for t in sorted(team_names))
            raise ValueError(
                f"Player {player_id} belongs to team {team_id}, which is not in the "
                f"payload's team list: {available}"
            )

        status = str(element.get("status", "a"))
        players.append(
            Player(
                id=player_id,
                name=str(element.get("web_name") or f"player-{player_id}"),
                element_type=element_type,
                position=rules.position(element_type).short_name,
                team_id=team_id,
                team_name=team_names[team_id],
                cost_tenths=_require_int(element, "now_cost"),
                status=status,
                availability=availability_from(
                    status, _optional_int(element, "chance_of_playing_next_round")
                ),
                ep_next=_optional_float(element, "ep_next"),
                minutes=_require_int(element, "minutes"),
                total_points=_require_int(element, "total_points"),
            )
        )
    return players


def _team_names(payload: dict[str, Any]) -> dict[int, str]:
    teams = payload.get("teams")
    if not isinstance(teams, list) or not teams:
        raise ValueError("bootstrap-static payload has no 'teams' list")
    names: dict[int, str] = {}
    for team in teams:
        if not isinstance(team, dict):
            raise ValueError(f"Expected each team to be an object, got {team!r}")
        team_id = _require_int(team, "id")
        names[team_id] = str(team.get("short_name") or team.get("name") or team_id)
    return names


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _require_int(entry: dict[str, Any], key: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected an integer '{key}', got {value!r} ({type(value).__name__})")
    return value


def _optional_int(entry: dict[str, Any], key: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Expected '{key}' to be a number or null, got {value!r} ({type(value).__name__})"
        )
    return int(value)


def _optional_float(entry: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric field that the FPL API serialises as a string (e.g. `ep_next: "4.2"`).

    A `null` means the API has published no forecast for that player — which happens for
    players who have never featured — and is worth `default`, not an error. A *malformed*
    value still raises: that would signal the payload has changed shape under us.
    """
    value = entry.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Expected a number for '{key}', got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected '{key}' to be a number, got the unparseable string {value!r}"
            ) from exc
    raise ValueError(f"Expected a number for '{key}', got {value!r} ({type(value).__name__})")
