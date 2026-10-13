"""Squad-selection rules, read from the FPL API rather than hardcoded.

`bootstrap-static` publishes every rule the optimiser needs: squad size, budget, the per-club
cap, how many of each position make up the 15, and the min/max of each position allowed in the
starting XI. Parsing them out of the payload means the model tracks the game if FPL ever
changes it, and keeps the constraint code free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PositionRule:
    """Rules for one position (goalkeeper, defender, midfielder, forward)."""

    element_type: int
    """FPL's position id: 1=GKP, 2=DEF, 3=MID, 4=FWD."""

    short_name: str
    squad_select: int
    """How many of this position must be in the 15-man squad."""

    min_play: int
    """Fewest of this position allowed in the starting XI."""

    max_play: int
    """Most of this position allowed in the starting XI."""


@dataclass(frozen=True)
class GameRules:
    """The complete set of squad-legality rules for a gameweek."""

    squad_size: int
    starting_size: int
    club_limit: int
    budget_tenths: int
    """Budget in tenths of a million, matching the API's `now_cost` units (1000 = £100.0m)."""

    positions: tuple[PositionRule, ...]

    @property
    def element_types(self) -> tuple[int, ...]:
        return tuple(rule.element_type for rule in self.positions)

    def position(self, element_type: int) -> PositionRule:
        for rule in self.positions:
            if rule.element_type == element_type:
                return rule
        available = ", ".join(str(t) for t in self.element_types)
        raise ValueError(f"Unknown element_type {element_type!r}; expected one of: {available}")

    @classmethod
    def from_bootstrap(cls, payload: dict[str, Any]) -> GameRules:
        """Build the rules from a `bootstrap-static` payload.

        Raises:
            ValueError: if a required key is missing or has an unexpected type.
        """
        settings = _require_mapping(payload, "game_settings")
        element_types = _require_list(payload, "element_types")
        if not element_types:
            raise ValueError("bootstrap-static payload has an empty 'element_types' list")

        positions = tuple(
            PositionRule(
                element_type=_require_int(entry, "id", "element_types"),
                short_name=str(
                    entry.get("singular_name_short")
                    or f"TYPE{_require_int(entry, 'id', 'element_types')}"
                ),
                squad_select=_require_int(entry, "squad_select", "element_types"),
                min_play=_require_int(entry, "squad_min_play", "element_types"),
                max_play=_require_int(entry, "squad_max_play", "element_types"),
            )
            for entry in element_types
        )

        rules = cls(
            squad_size=_require_int(settings, "squad_squadsize", "game_settings"),
            starting_size=_require_int(settings, "squad_squadplay", "game_settings"),
            club_limit=_require_int(settings, "squad_team_limit", "game_settings"),
            budget_tenths=_require_int(settings, "squad_total_spend", "game_settings"),
            positions=positions,
        )
        rules.validate()
        return rules

    def validate(self) -> None:
        """Check the rules are internally consistent.

        A payload that says "pick 15" but whose positions sum to 14 would otherwise surface as
        a baffling infeasible solve rather than a clear error.
        """
        selected = sum(rule.squad_select for rule in self.positions)
        if selected != self.squad_size:
            raise ValueError(
                f"Position quotas sum to {selected} but squad_size is {self.squad_size}"
            )
        if not 0 < self.starting_size <= self.squad_size:
            raise ValueError(f"starting_size {self.starting_size} must be in 1..{self.squad_size}")
        min_players = sum(rule.min_play for rule in self.positions)
        max_players = sum(rule.max_play for rule in self.positions)
        if not min_players <= self.starting_size <= max_players:
            raise ValueError(
                f"No legal formation: starting_size {self.starting_size} is outside the "
                f"{min_players}..{max_players} allowed by the per-position play limits"
            )
        for rule in self.positions:
            if rule.max_play > rule.squad_select:
                raise ValueError(
                    f"{rule.short_name}: max_play {rule.max_play} exceeds squad_select "
                    f"{rule.squad_select}"
                )
        if self.club_limit < 1:
            raise ValueError(f"club_limit must be >= 1, got {self.club_limit}")
        if self.budget_tenths < 1:
            raise ValueError(f"budget_tenths must be >= 1, got {self.budget_tenths}")


HIT_COST_POINTS = 4
"""Points docked for each transfer beyond your free ones. Not published in `game_settings`, so
this is the one FPL rule the code has to know by heart."""


@dataclass(frozen=True)
class SeasonRules:
    """What the rules actually were in a given season.

    The live API tells you the rules of *today's* game, which is exactly wrong for a backtest of
    an older one. Two things have changed inside the window we validate against, and both would
    corrupt a multi-season backtest silently and in our favour:

    - **Defensive contribution** (2 points for 10 defensive actions, 12 for a midfielder) was
      introduced in 2025-26. Award it in a 2023-24 simulation and the model is scoring points
      for a rule that had not been invented.
    - **Transfer banking**: from 2024-25 you may bank up to five free transfers. Before that it
      was two. A backtest of 2022-23 run with today's cap gives the simulated manager a
      flexibility no real manager had.

    Neither would ever throw an error. They would just quietly make us look better.
    """

    season: str
    max_free_transfers: int
    defensive_contribution: bool
    sell_on_fee: float = 0.5
    hit_cost_points: int = HIT_COST_POINTS

    def transfer_rules(self) -> TransferRules:
        return TransferRules(
            sell_on_fee=self.sell_on_fee,
            max_free_transfers=self.max_free_transfers,
            hit_cost_points=self.hit_cost_points,
        )


SEASON_RULES: dict[str, SeasonRules] = {
    "2021-22": SeasonRules("2021-22", max_free_transfers=2, defensive_contribution=False),
    "2022-23": SeasonRules("2022-23", max_free_transfers=2, defensive_contribution=False),
    "2023-24": SeasonRules("2023-24", max_free_transfers=2, defensive_contribution=False),
    "2024-25": SeasonRules("2024-25", max_free_transfers=5, defensive_contribution=False),
    "2025-26": SeasonRules("2025-26", max_free_transfers=5, defensive_contribution=True),
}


def rules_for_season(season: str) -> SeasonRules:
    """The rules in force that season.

    Raises:
        ValueError: on an unknown season. Deliberately an error rather than a default — silently
            backtesting an unknown season under today's rules is precisely the failure this
            exists to prevent.
    """
    try:
        return SEASON_RULES[season]
    except KeyError:
        known = ", ".join(sorted(SEASON_RULES))
        raise ValueError(
            f"No rules recorded for season {season!r}. The game's scoring and transfer rules "
            f"change between seasons, so backtesting one we have not pinned down would be "
            f"quietly wrong. Known seasons: {known}"
        ) from None


@dataclass(frozen=True)
class TransferRules:
    """The rules governing transfers between gameweeks."""

    sell_on_fee: float
    """Fraction of any *profit* the game keeps when you sell (the API says 0.5). Buy at £5.0m,
    sell at £5.4m, and you receive £5.2m — not £5.4m."""

    max_free_transfers: int
    """How many free transfers you may bank. The API publishes `max_extra_free_transfers`, the
    number *on top of* the one you earn each week, so 4 there means 5 here."""

    hit_cost_points: int = HIT_COST_POINTS

    @classmethod
    def from_bootstrap(cls, payload: dict[str, Any]) -> TransferRules:
        settings = _require_mapping(payload, "game_settings")
        fee = settings.get("transfers_sell_on_fee")
        if not isinstance(fee, (int, float)) or isinstance(fee, bool):
            raise ValueError(
                f"game_settings: expected a numeric 'transfers_sell_on_fee', got {fee!r}"
            )
        if not 0.0 <= float(fee) <= 1.0:
            raise ValueError(f"transfers_sell_on_fee must be in 0..1, got {fee!r}")

        extra = _require_int(settings, "max_extra_free_transfers", "game_settings")
        rules = cls(sell_on_fee=float(fee), max_free_transfers=extra + 1)
        rules.validate()
        return rules

    def validate(self) -> None:
        if self.max_free_transfers < 1:
            raise ValueError(f"max_free_transfers must be >= 1, got {self.max_free_transfers}")
        if self.hit_cost_points < 0:
            raise ValueError(f"hit_cost_points must be >= 0, got {self.hit_cost_points}")

    def selling_price(self, purchase_price_tenths: int, market_price_tenths: int) -> int:
        """What you actually receive for a player, in tenths.

        You keep only `1 - sell_on_fee` of any rise, rounded *down* to the nearest £0.1m — and
        a player who has fallen in value is sold at the lower market price with no cushion.
        Tracking this is what stops a simulated manager getting quietly richer than a real one.
        """
        if purchase_price_tenths < 1 or market_price_tenths < 1:
            raise ValueError(
                f"Prices must be >= 1 tenth, got purchase={purchase_price_tenths}, "
                f"market={market_price_tenths}"
            )
        profit = market_price_tenths - purchase_price_tenths
        if profit <= 0:
            return market_price_tenths
        kept = int(profit * (1.0 - self.sell_on_fee))  # int() truncates: rounds down
        return purchase_price_tenths + kept


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"bootstrap-static payload is missing a '{key}' object (got {type(value).__name__})"
        )
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"bootstrap-static payload is missing a '{key}' list (got {type(value).__name__})"
        )
    return value


def _require_int(entry: dict[str, Any], key: str, context: str) -> int:
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{context}: expected an integer '{key}', got {value!r} ({type(value).__name__})"
        )
    return value
