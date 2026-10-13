"""Play a whole season: pick at GW1, then transfer, field and score through GW38.

The manager's state is four things — the fifteen owned, the cash in the bank, what each player
was *bought* for (the sell-on fee makes that load-bearing), and how many free transfers have been
banked. Everything else is derived.

Chips are optional (`use_chips`). The free hit is the fiddly one: it lends you a squad for a
single gameweek and then takes it back, so the state carried forward is the one you had *before*
playing it — get that wrong and the manager keeps a squad they never really owned.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from fpl_optimiser.backtest.chips import Chip, ChipState, choose_chip
from fpl_optimiser.backtest.planner import DEFAULT_DISCOUNT, plan_transfers
from fpl_optimiser.backtest.providers import ProviderFactory
from fpl_optimiser.backtest.scoring import GameweekScore, score_gameweek
from fpl_optimiser.backtest.season import GameweekView, SeasonHistory
from fpl_optimiser.backtest.transfers import TransferDecision, optimise_transfers
from fpl_optimiser.model import DEFAULT_BENCH_WEIGHTS, DEFAULT_SOLVER_TIMEOUT_SECONDS, BenchWeights
from fpl_optimiser.optimise import optimise_squad
from fpl_optimiser.rules import GameRules, TransferRules

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeasonResult:
    """The outcome of one provider playing one season."""

    provider: str
    scores: tuple[GameweekScore, ...]
    transfers_made: int
    hits_taken: int
    chips_played: dict[int, object]
    """Gameweek to the chip played in it."""

    final_squad_value_tenths: int

    @property
    def total_points(self) -> int:
        """Net points, after transfer hits — the number that would appear on your FPL page."""
        return sum(score.points for score in self.scores)

    @property
    def points_before_hits(self) -> int:
        return sum(score.points_before_hits for score in self.scores)

    @property
    def hit_points(self) -> int:
        return sum(score.transfer_hit_points for score in self.scores)

    @property
    def auto_subs(self) -> int:
        return sum(len(score.substitutions) for score in self.scores)


def _expected_points_map(
    provider: ProviderFactory, view: GameweekView, history: SeasonHistory
) -> dict[int, float]:
    """The provider's forecast for one gameweek, as a plain lookup the planner can index."""
    forecast = provider.build(view, history)
    return {player.id: float(forecast(player)) for player in view.pool}


def _consider_chips(
    *,
    gameweek: int,
    weeks: list[int],
    view: GameweekView,
    decision: TransferDecision,
    chip_state: ChipState,
    state: _ManagerState,
    rules: GameRules,
    transfer_rules: TransferRules,
    expected_points,
    views: list[GameweekView],
    points_by_gameweek: dict[int, dict[int, float]],
    bench_weights: BenchWeights,
    timeout_seconds: float,
) -> tuple[Chip | None, TransferDecision]:
    """Decide whether to play a chip, and re-solve the squad if one changes the transfer rules.

    Returns the chip played (if any) and the decision to act on — which is a *different* squad
    for a wildcard or free hit, since both lift the transfer limit entirely.
    """
    squad = decision.squad

    # A free hit lends you a squad for exactly one gameweek and then takes it back, so it should
    # be built to win *this* week and nothing else. The single-gameweek solve is the right one.
    free_hit: TransferDecision | None = None
    if chip_state.available(Chip.FREE_HIT, gameweek):
        free_hit = optimise_transfers(
            list(view.pool),
            rules,
            transfer_rules,
            expected_points,
            current_squad_ids=state.squad_ids,
            purchase_prices=state.purchase_prices,
            bank_tenths=state.bank_tenths,
            free_transfers=rules.squad_size,  # unlimited, and free
            bench_weights=bench_weights,
            timeout_seconds=timeout_seconds,
        )

    # A wildcard is the opposite: the squad it builds is the one you *keep*. Optimising it for a
    # single gameweek — which the first version did — produces a side that wins this week and
    # then falls apart, and it measurably cost points. It has to be planned over the horizon,
    # like any other transfer decision, only without the transfer limit.
    wildcard_gain = 0.0
    wildcard: TransferDecision | None = None
    if chip_state.available(Chip.WILDCARD, gameweek) and views:
        wildcard = plan_transfers(
            views,
            points_by_gameweek,
            rules,
            transfer_rules,
            current_squad_ids=state.squad_ids,
            purchase_prices=state.purchase_prices,
            bank_tenths=state.bank_tenths,
            free_transfers=rules.squad_size,  # unlimited, and free
            bench_weights=bench_weights,
            timeout_seconds=timeout_seconds,
        )
        wildcard_gain = wildcard.squad.starting_expected_points - squad.starting_expected_points

    chip_decision = choose_chip(
        gameweek,
        weeks,
        chip_state,
        bench_expected_points=squad.bench_expected_points,
        captain_expected_points=squad.expected_points[squad.captain.id],
        # The squad you *hold*, not the one the planner has just built. Measured after the
        # transfers, the planner has already dodged the blank — using real transfers to do what
        # a free hit would have done for nothing — and the chip never fires.
        squad_with_fixtures=sum(
            1 for player_id in state.squad_ids if view.fixture_count(player_id) > 0
        ),
        wildcard_gain=wildcard_gain,
    )
    chip = chip_decision.chip
    if chip is None:
        return None, decision

    chip_state.play(chip, gameweek)
    logger.info("GW%d: playing %s — %s", gameweek, chip.value, chip_decision.reason)

    if chip is Chip.FREE_HIT and free_hit is not None:
        return chip, free_hit
    if chip is Chip.WILDCARD and wildcard is not None:
        return chip, wildcard
    # Bench boost and triple captain change how the squad *scores*, not what it is.
    return chip, decision


@dataclass
class _ManagerState:
    """Mutable state carried across gameweeks."""

    squad_ids: set[int] = field(default_factory=set)
    bank_tenths: int = 0
    purchase_prices: dict[int, int] = field(default_factory=dict)
    free_transfers: int = 1


def run_season(
    history: SeasonHistory,
    rules: GameRules,
    transfer_rules: TransferRules,
    provider: ProviderFactory,
    *,
    bench_weights: BenchWeights = DEFAULT_BENCH_WEIGHTS,
    horizon: int = 1,
    discount: float = DEFAULT_DISCOUNT,
    use_chips: bool = False,
    timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
    gameweeks: Sequence[int] | None = None,
) -> SeasonResult:
    """Simulate a full season under the real transfer rules.

    At each gameweek the provider is rebuilt from that gameweek's *pre-deadline* view, the squad
    is re-optimised subject to the transfers available, and the result is scored against what
    actually happened, with auto-substitutions applied.

    Args:
        horizon: How many gameweeks the transfer planner looks ahead. 1 is the greedy policy,
            which churns and panics at blanks; 4-6 lets it bank transfers and plan through them.
        discount: How much a point next gameweek is worth relative to this one, compounded.
        use_chips: Play the wildcard, free hit, bench boost and triple captain when they clear
            their thresholds. Off by default, which makes the score conservative.
    """
    weeks = list(history.gameweeks if gameweeks is None else gameweeks)
    if not weeks:
        raise ValueError("No gameweeks to simulate")

    state = _ManagerState()
    chip_state = ChipState()
    scores: list[GameweekScore] = []
    transfers_made = 0
    hits_taken = 0
    squad_value = 0

    for gameweek in weeks:
        view = history.view_at(gameweek)
        expected_points = provider.build(view, history)

        if not state.squad_ids:
            # Opening gameweek: nothing owned, so this is the plain from-scratch pick.
            squad = optimise_squad(
                list(view.pool),
                rules,
                expected_points,
                bench_weights=bench_weights,
                timeout_seconds=timeout_seconds,
            )
            hit_points = 0
            state.squad_ids = set(squad.player_ids)
            state.purchase_prices = {p.id: p.cost_tenths for p in squad.players}
            state.bank_tenths = rules.budget_tenths - squad.total_cost_tenths
            squad_value = squad.total_cost_tenths
            scores.append(score_gameweek(squad, history.actuals_at(gameweek), rules, gameweek))
            continue

        views: list[GameweekView] = [view]
        points_by_gameweek = {view.gameweek: _expected_points_map(provider, view, history)}

        if horizon > 1:
            # Look ahead: solve the next few gameweeks together, execute only this one, and
            # re-plan next week with fresh information. This is what stops the blank-gameweek
            # panic that the one-week policy pays 16-point hits for.
            upcoming = [gw for gw in weeks if gw >= gameweek][:horizon]
            views = [view] + [history.view_at(gw) for gw in upcoming[1:]]
            points_by_gameweek = {
                v.gameweek: _expected_points_map(provider, v, history) for v in views
            }
            decision = plan_transfers(
                views,
                points_by_gameweek,
                rules,
                transfer_rules,
                current_squad_ids=state.squad_ids,
                purchase_prices=state.purchase_prices,
                bank_tenths=state.bank_tenths,
                free_transfers=state.free_transfers,
                bench_weights=bench_weights,
                discount=discount,
                timeout_seconds=timeout_seconds,
            )
        else:
            decision = optimise_transfers(
                list(view.pool),
                rules,
                transfer_rules,
                expected_points,
                current_squad_ids=state.squad_ids,
                purchase_prices=state.purchase_prices,
                bank_tenths=state.bank_tenths,
                free_transfers=state.free_transfers,
                bench_weights=bench_weights,
                timeout_seconds=timeout_seconds,
            )

        chip = None
        if use_chips:
            chip, decision = _consider_chips(
                gameweek=gameweek,
                weeks=weeks,
                view=view,
                decision=decision,
                chip_state=chip_state,
                state=state,
                rules=rules,
                transfer_rules=transfer_rules,
                expected_points=expected_points,
                views=views,
                points_by_gameweek=points_by_gameweek,
                bench_weights=bench_weights,
                timeout_seconds=timeout_seconds,
            )

        squad = decision.squad
        hit_points = decision.hit_points
        transfers_made += decision.transfer_count
        hits_taken += decision.hits

        # A free hit lends you a squad for one gameweek and then takes it back, so the state to
        # carry forward is the one you had *before* playing it.
        before_free_hit = (
            (set(state.squad_ids), dict(state.purchase_prices), state.bank_tenths)
            if chip is Chip.FREE_HIT
            else None
        )

        for player in decision.transfers_out:
            state.purchase_prices.pop(player.id, None)
        for player in decision.transfers_in:
            # You buy at the market price; the sell-on fee bites only when you sell.
            state.purchase_prices[player.id] = player.cost_tenths
        state.squad_ids = set(squad.player_ids)
        state.bank_tenths = decision.bank_tenths

        if chip in (Chip.WILDCARD, Chip.FREE_HIT):
            # Both give unlimited transfers at no cost, so this week's allowance is untouched.
            state.free_transfers = min(transfer_rules.max_free_transfers, state.free_transfers + 1)
        else:
            remaining = max(0, state.free_transfers - decision.transfer_count)
            state.free_transfers = min(transfer_rules.max_free_transfers, remaining + 1)

        squad_value = squad.total_cost_tenths
        scores.append(
            score_gameweek(
                squad,
                history.actuals_at(gameweek),
                rules,
                gameweek,
                transfer_hit_points=hit_points,
                bench_boost=chip is Chip.BENCH_BOOST,
                captain_multiplier=3 if chip is Chip.TRIPLE_CAPTAIN else 2,
            )
        )

        if before_free_hit is not None:
            state.squad_ids, state.purchase_prices, state.bank_tenths = before_free_hit

    return SeasonResult(
        provider=provider.name,
        scores=tuple(scores),
        transfers_made=transfers_made,
        hits_taken=hits_taken,
        chips_played=dict(chip_state.played),
        final_squad_value_tenths=squad_value + state.bank_tenths,
    )
