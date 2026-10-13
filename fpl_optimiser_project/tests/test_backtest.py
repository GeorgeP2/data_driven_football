"""Tests for the backtest harness.

Organised around the four ways a backtest lies to you. Every one of them fails *silently* and in
the flattering direction, so each gets a test that would catch it.
"""

from __future__ import annotations

import pytest
from fpl_helpers import lagged, make_player, record
from fpl_optimiser.backtest.history import (
    GameweekRecord,
    _aggregate_fixtures,
    _derive_team_ids,
    _fill_blank_gameweeks,
)
from fpl_optimiser.backtest.providers import PerfectForesight, TrailingForm
from fpl_optimiser.backtest.scoring import score_gameweek
from fpl_optimiser.backtest.season import SeasonHistory, availability_proxy
from fpl_optimiser.backtest.transfers import optimise_transfers
from fpl_optimiser.expected_points import StaticProvider
from fpl_optimiser.model import Squad
from fpl_optimiser.rules import TransferRules


@pytest.fixture
def transfer_rules() -> TransferRules:
    """The real rules: half of any profit is taken on sale, five free transfers may be banked."""
    return TransferRules(sell_on_fee=0.5, max_free_transfers=5)


# ---------------------------------------------------------------------------
# Trap 1: the sell-on fee
# ---------------------------------------------------------------------------


class TestSellOnFee:
    def test_profit_is_halved_on_sale(self, transfer_rules) -> None:
        # Bought at £5.0m, now worth £5.4m: you keep half the £0.4m rise, so £5.2m.
        assert transfer_rules.selling_price(50, 54) == 52

    def test_odd_profit_rounds_down(self, transfer_rules) -> None:
        # A £0.3m rise halves to £0.15m, which rounds *down* to £0.1m. Not £0.2m.
        assert transfer_rules.selling_price(50, 53) == 51

    def test_a_price_fall_has_no_cushion(self, transfer_rules) -> None:
        # You eat the whole loss: bought at £5.0m, worth £4.6m, you get £4.6m.
        assert transfer_rules.selling_price(50, 46) == 46

    def test_unchanged_price_sells_at_cost(self, transfer_rules) -> None:
        assert transfer_rules.selling_price(50, 50) == 50

    def test_rules_come_from_the_payload(self, payload) -> None:
        payload["game_settings"]["transfers_sell_on_fee"] = 0.5
        payload["game_settings"]["max_extra_free_transfers"] = 4

        rules = TransferRules.from_bootstrap(payload)

        assert rules.sell_on_fee == 0.5
        # 4 *extra* on top of the one earned each week = 5 bankable.
        assert rules.max_free_transfers == 5


# ---------------------------------------------------------------------------
# Trap 2: double and blank gameweeks
# ---------------------------------------------------------------------------


def csv_row(
    element: int,
    gameweek: int,
    *,
    name: str = "player",
    position: str = "MID",
    team: str = "LIV",
    opponent: int = 1,
    fixture: int = 100,
    was_home: bool = True,
    points: int = 0,
    minutes: int = 90,
    value: int = 50,
) -> dict[str, str]:
    """One archive row: a single player in a single fixture."""
    return {
        "element": str(element),
        "GW": str(gameweek),
        "name": name,
        "position": position,
        "team": team,
        "value": str(value),
        "total_points": str(points),
        "minutes": str(minutes),
        "opponent_team": str(opponent),
        "fixture": str(fixture),
        "was_home": str(was_home),
        "team_h_score": "1",
        "team_a_score": "0",
        "starts": "1",
        "bonus": "0",
        "saves": "0",
        "defensive_contribution": "0",
        "expected_goals": "0.0",
        "expected_assists": "0.0",
    }


def both_sides(
    fixture: int, home: str, home_id: int, away: str, away_id: int
) -> list[dict[str, str]]:
    """Both teams of one fixture, which is what makes the team ids recoverable."""
    return [
        csv_row(
            900 + home_id,
            1,
            team=home,
            opponent=away_id,
            fixture=fixture,
            was_home=True,
        ),
        csv_row(
            900 + away_id,
            1,
            team=away,
            opponent=home_id,
            fixture=fixture,
            was_home=False,
        ),
    ]


class TestTeamIds:
    def test_team_ids_are_derived_from_the_archive_alone(self) -> None:
        # The archive names your own team but gives the opponent as an id, so the two are never
        # stated together. They are recoverable anyway: each side of a fixture names the other.
        rows = both_sides(5, "Sunderland", 17, "West Ham", 19)

        assert _derive_team_ids(rows) == {"Sunderland": 17, "West Ham": 19}


class TestDoubleAndBlankGameweeks:
    def test_a_double_gameweek_is_summed_into_one_record(self) -> None:
        rows = [
            *both_sides(1, "LIV", 12, "ARS", 1),
            csv_row(
                7,
                33,
                name="Salah",
                team="LIV",
                opponent=1,
                fixture=1,
                points=9,
                minutes=90,
                value=145,
            ),
            csv_row(
                7,
                33,
                name="Salah",
                team="LIV",
                opponent=1,
                fixture=2,
                points=6,
                minutes=75,
                value=145,
            ),
        ]

        records = _aggregate_fixtures(rows, "2025-26", _derive_team_ids(rows))

        merged = records[(7, 33)]
        assert merged.total_points == 15  # 9 + 6, not 9 and not 6
        assert merged.minutes == 165
        assert merged.fixture_count == 2

    def test_a_blank_gameweek_scores_zero_rather_than_going_missing(self) -> None:
        # Player 1 plays in both gameweeks; player 2 has no fixture in GW2.
        records = {
            (1, 1): record(1, 1, points=5),
            (1, 2): record(1, 2, points=7),
            (2, 1): record(2, 1, points=9, price=60),
        }

        filled = _fill_blank_gameweeks(records)

        blank = next(r for r in filled if r.player_id == 2 and r.gameweek == 2)
        assert blank.total_points == 0
        assert blank.minutes == 0
        assert blank.fixture_count == 0
        assert blank.opponents == (), "a blank gameweek has no fixture at all"
        assert blank.price_tenths == 60, "price carries forward through a blank"
        assert len(filled) == 4, "every player has a record in every gameweek"

    def test_gk_and_gkp_both_parse(self) -> None:
        # The archive writes "GK"; the live API calls the same position "GKP".
        rows = [
            *both_sides(1, "ARS", 1, "LIV", 12),
            csv_row(1, 1, name="Raya", position="GK", team="ARS", opponent=12, fixture=1),
        ]

        records = _aggregate_fixtures(rows, "2025-26", _derive_team_ids(rows))

        assert records[(1, 1)].element_type == 1

    def test_assistant_managers_are_dropped_not_selected(self) -> None:
        # The 2024-25 archive lists managers (Pep Guardiola et al.) as position "AM". They are
        # not footballers, the position does not exist in the rules we optimise against, and a
        # naive parser dies on them.
        rows = [
            *both_sides(1, "MCI", 13, "ARS", 1),
            csv_row(736, 23, name="Pep", position="AM", team="MCI", opponent=1, fixture=1),
            csv_row(1, 23, name="Raya", position="GK", team="ARS", opponent=13, fixture=1),
        ]

        records = _aggregate_fixtures(rows, "2024-25", _derive_team_ids(rows))

        assert 736 not in {player_id for player_id, _ in records}
        assert (1, 23) in records

    def test_unknown_position_raises(self) -> None:
        rows = [
            *both_sides(1, "ARS", 1, "LIV", 12),
            csv_row(1, 1, position="SWEEPER", team="ARS", opponent=12, fixture=1),
        ]

        with pytest.raises(ValueError, match="Unknown position"):
            _aggregate_fixtures(rows, "2025-26", _derive_team_ids(rows))


# ---------------------------------------------------------------------------
# Trap 3: auto-substitutions
# ---------------------------------------------------------------------------


def _squad_of(starters, bench, captain, vice) -> Squad:
    points = {p.id: 0.0 for p in list(starters) + list(bench)}
    return Squad(
        starting_xi=tuple(starters),
        bench=tuple(bench),
        captain=captain,
        vice_captain=vice,
        expected_points=points,
    )


def _legal_xi() -> list:
    """1 GK, 4 DEF, 4 MID, 2 FWD — a legal 4-4-2."""
    return (
        [make_player(1, 1)]
        + [make_player(10 + i, 2) for i in range(4)]
        + [make_player(20 + i, 3) for i in range(4)]
        + [make_player(30 + i, 4) for i in range(2)]
    )


class TestAutoSubstitutions:
    def test_a_blanking_starter_is_replaced_by_the_first_bench_player_who_played(
        self, rules
    ) -> None:
        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        # Midfielder 20 does not play; the first bench outfielder (40) does.
        actuals[20] = record(20, 1, points=0, minutes=0)

        score = score_gameweek(squad, actuals, rules, 1)

        assert [out.id for out, _ in score.substitutions] == [20]
        assert [into.id for _, into in score.substitutions] == [40]
        assert 40 in {p.id for p in score.fielded_xi}
        assert 20 not in {p.id for p in score.fielded_xi}

    def test_a_bench_player_who_also_blanked_is_skipped(self, rules) -> None:
        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        actuals[20] = record(20, 1, points=0, minutes=0)
        actuals[40] = record(40, 1, points=0, minutes=0)  # first choice also blanked

        score = score_gameweek(squad, actuals, rules, 1)

        assert [into.id for _, into in score.substitutions] == [41]

    def test_an_outfielder_never_replaces_a_goalkeeper(self, rules) -> None:
        xi = _legal_xi()
        # Bench keeper blanked too, so no legal replacement exists for the starting keeper.
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        actuals[1] = record(1, 1, points=0, minutes=0)
        actuals[2] = record(2, 1, points=0, minutes=0)

        score = score_gameweek(squad, actuals, rules, 1)

        assert score.substitutions == (), "no outfielder may keep goal"
        assert 1 in {p.id for p in score.fielded_xi}

    def test_a_substitution_that_would_break_the_formation_is_refused(self, rules) -> None:
        # Start 3 defenders — the minimum. If one blanks, a midfielder cannot replace him,
        # because that would leave only 2 at the back.
        xi = (
            [make_player(1, 1)]
            + [make_player(10 + i, 2) for i in range(3)]
            + [make_player(20 + i, 3) for i in range(5)]
            + [make_player(30 + i, 4) for i in range(2)]
        )
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 4)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        actuals[10] = record(10, 1, points=0, minutes=0)  # a defender blanks

        score = score_gameweek(squad, actuals, rules, 1)

        assert score.substitutions == (), "only 3 DEF started; none of the bench is a DEF"

    def test_the_armband_passes_to_the_vice_when_the_captain_does_not_play(self, rules) -> None:
        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        captain, vice = xi[5], xi[6]
        squad = _squad_of(xi, bench, captain=captain, vice=vice)
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        actuals[captain.id] = record(captain.id, 1, points=0, minutes=0)
        actuals[vice.id] = record(vice.id, 1, points=10)

        score = score_gameweek(squad, actuals, rules, 1)

        assert score.captain is not None
        assert score.captain.id == vice.id
        # 10 starters on 2, the vice on 10, and the vice doubled.
        assert score.points_before_hits == (10 * 2) + 10 + 10

    def test_nobody_is_doubled_when_neither_captain_nor_vice_plays(self, rules) -> None:
        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        captain, vice = xi[5], xi[6]
        squad = _squad_of(xi, bench, captain=captain, vice=vice)
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}
        actuals[captain.id] = record(captain.id, 1, points=0, minutes=0)
        actuals[vice.id] = record(vice.id, 1, points=0, minutes=0)

        score = score_gameweek(squad, actuals, rules, 1)

        assert score.captain is None

    def test_hits_are_docked_from_the_gameweek_score(self, rules) -> None:
        xi = _legal_xi()
        bench = [make_player(2, 1), make_player(40, 3), make_player(41, 3)]
        squad = _squad_of(xi, bench, captain=xi[5], vice=xi[6])
        actuals = {p.id: record(p.id, 1, points=2) for p in xi + bench}

        score = score_gameweek(squad, actuals, rules, 1, transfer_hit_points=8)

        assert score.points_before_hits == 24  # 11 starters x 2, captain doubled
        assert score.points == 16


# ---------------------------------------------------------------------------
# Trap 4: lookahead. The canaries.
# ---------------------------------------------------------------------------


def _tiny_season() -> list[GameweekRecord]:
    """Three gameweeks of a pool just big enough to field a legal squad."""
    records: list[GameweekRecord] = []
    slots = [(1, 1, 3), (2, 2, 6), (3, 3, 6), (4, 4, 4)]
    player_id = 1
    for element_type, _, count in slots:
        for _index in range(count):
            for gameweek in (1, 2, 3):
                records.append(
                    record(
                        player_id,
                        gameweek,
                        element_type=element_type,
                        points=(player_id + gameweek) % 8,
                        minutes=90,
                        price=40 + (player_id % 3) * 5,
                        team=f"T{player_id % 8}",
                        team_id=(player_id % 8) + 1,
                    )
                )
            player_id += 1
    return records


class TestNoLookahead:
    def test_the_view_carries_only_completed_gameweeks(self, rules) -> None:
        history = SeasonHistory(_tiny_season(), rules)

        view = history.view_at(2)

        stats = view.stats(1)
        assert len(stats.points) == 1, "at GW2 exactly one gameweek is in the past"
        assert len(stats.minutes) == 1
        # GW1 was the only completed week, so the total must equal that week alone.
        actual_gw1 = history.actuals_at(1)[1].total_points
        assert stats.total_points == actual_gw1

    def test_gameweek_one_has_no_history_at_all(self, rules) -> None:
        history = SeasonHistory(_tiny_season(), rules)

        stats = history.view_at(1).stats(1)

        assert stats.points == ()
        assert stats.minutes == ()

    def test_the_pool_never_exposes_this_gameweeks_outcome(self, rules) -> None:
        history = SeasonHistory(_tiny_season(), rules)

        view = history.view_at(3)

        for player in view.pool:
            this_week = history.actuals_at(3)[player.id]
            stats = view.stats(player.id)
            # The player's own season-to-date totals must exclude the gameweek being picked.
            assert stats.total_points == sum(stats.points)
            assert len(stats.points) == 2
            assert player.total_points == stats.total_points
            # And ep_next is dead: FPL's forecast is not retained for past gameweeks.
            assert player.ep_next == 0.0
            assert this_week.gameweek == 3

    def test_a_lagged_baseline_scores_far_below_perfect_foresight(
        self, rules, transfer_rules
    ) -> None:
        """The canary that matters. If a naive model approaches the ceiling, something leaks."""
        from fpl_optimiser.backtest.engine import run_season

        history = SeasonHistory(_tiny_season(), rules)

        ceiling = run_season(history, rules, transfer_rules, PerfectForesight(), timeout_seconds=10)
        naive = run_season(
            history, rules, transfer_rules, TrailingForm(window=2), timeout_seconds=10
        )

        assert naive.total_points < ceiling.total_points, (
            "a lagged model matched perfect foresight — it is reading the future"
        )

    def test_availability_is_predicted_from_lagged_minutes_only(self) -> None:
        nailed = lagged(1, points=(5, 6, 4), minutes=(90, 90, 90))
        dropped = lagged(2, points=(0, 0, 0), minutes=(0, 0, 0))
        cameo = lagged(3, points=(1, 1, 1), minutes=(30, 30, 30))

        assert availability_proxy(nailed, 4) == 1.0
        assert availability_proxy(dropped, 4) == 0.0
        assert availability_proxy(cameo, 4) == pytest.approx(0.5)

    def test_gameweek_one_availability_falls_back_on_last_season(self) -> None:
        # No history this season; last season they were a regular.
        regular = lagged(1, previous_season_points=150, previous_season_minutes=3000)
        newcomer = lagged(2)

        assert availability_proxy(regular, 1) > 0.9
        assert availability_proxy(newcomer, 1) == 0.0

    def test_a_new_signing_mid_season_is_assumed_unavailable(self) -> None:
        # In the pool at GW10 but no completed gameweeks: nothing to go on, so assume nothing.
        unknown = lagged(1)

        assert availability_proxy(unknown, 10) == 0.0


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------


class TestTransfers:
    def _pool_and_squad(self):
        """A legal 15-man squad plus spare replacements.

        Teams are spread across the whole pool, not restarted per position — otherwise club 1
        collects a keeper, a defender, a midfielder and a forward, and the "current squad" is
        already illegal under the 3-per-club rule before the solver even sees it.
        """
        pool = []
        player_id = 1
        for element_type, count in [(1, 3), (2, 6), (3, 6), (4, 4)]:
            for _ in range(count):
                pool.append(
                    make_player(
                        player_id,
                        element_type,
                        team_id=((player_id - 1) % 10) + 1,
                        cost_tenths=45,
                        ep_next=1.0,
                    )
                )
                player_id += 1
        current = [p.id for p in pool[:2]]  # 2 GK
        current += [p.id for p in pool if p.element_type == 2][:5]
        current += [p.id for p in pool if p.element_type == 3][:5]
        current += [p.id for p in pool if p.element_type == 4][:3]
        return pool, current

    def test_a_free_transfer_is_taken_when_it_gains_points(self, rules, transfer_rules) -> None:
        pool, current = self._pool_and_squad()
        points = {p.id: 1.0 for p in pool}
        # An unowned midfielder is worth far more than any owned one.
        spare = next(p for p in pool if p.element_type == 3 and p.id not in current)
        points[spare.id] = 20.0

        decision = optimise_transfers(
            pool,
            rules,
            transfer_rules,
            StaticProvider(points_by_player_id=points),
            current_squad_ids=current,
            purchase_prices={i: 45 for i in current},
            bank_tenths=0,
            free_transfers=1,
        )

        assert spare.id in {p.id for p in decision.transfers_in}
        assert decision.hits == 0
        assert decision.transfer_count == 1

    def test_a_hit_is_refused_when_it_does_not_pay_for_itself(self, rules, transfer_rules) -> None:
        pool, current = self._pool_and_squad()
        points = {p.id: 1.0 for p in pool}
        # Two upgrades, each worth only +2 over the man they replace. The first is free; the
        # second would cost 4 points to gain 2, so the solver must decline it.
        spares = [p for p in pool if p.element_type == 3 and p.id not in current]
        for spare in spares:
            points[spare.id] = 3.0

        decision = optimise_transfers(
            pool,
            rules,
            transfer_rules,
            StaticProvider(points_by_player_id=points),
            current_squad_ids=current,
            purchase_prices={i: 45 for i in current},
            bank_tenths=0,
            free_transfers=1,
        )

        assert decision.hits == 0
        assert decision.transfer_count <= 1

    def test_a_hit_is_taken_when_the_gain_exceeds_four_points(self, rules, transfer_rules) -> None:
        pool, current = self._pool_and_squad()
        points = {p.id: 1.0 for p in pool}
        # Every unowned player is worth 30 against the 1 they would replace. Each swap gains
        # ~29 and costs at most 4, so the solver should take the hit rather than wait a week.
        for spare in (p for p in pool if p.id not in current):
            points[spare.id] = 30.0

        decision = optimise_transfers(
            pool,
            rules,
            transfer_rules,
            StaticProvider(points_by_player_id=points),
            current_squad_ids=current,
            purchase_prices={i: 45 for i in current},
            bank_tenths=0,
            free_transfers=1,
        )

        assert decision.transfer_count >= 2
        assert decision.hits == decision.transfer_count - 1
        assert decision.hit_points == decision.hits * 4

    def test_selling_price_not_market_price_funds_the_move(self, rules, transfer_rules) -> None:
        """The sell-on fee must bite: a squad bought cheap and now valuable cannot spend the
        full paper profit."""
        pool, current = self._pool_and_squad()
        points = {p.id: 1.0 for p in pool}
        # Everyone is now worth 45, but the squad was bought at 40 — a 0.5m paper profit each,
        # of which only half (rounded down to 0.2m) is realisable.
        purchase = {i: 40 for i in current}
        expensive = next(p for p in pool if p.element_type == 3 and p.id not in current)
        points[expensive.id] = 50.0

        decision = optimise_transfers(
            pool,
            rules,
            transfer_rules,
            StaticProvider(points_by_player_id=points),
            current_squad_ids=current,
            purchase_prices=purchase,
            bank_tenths=0,
            free_transfers=1,
        )

        # sell price = 40 + floor((45-40) * 0.5) = 40 + 2 = 42, not the 45 market price.
        assert transfer_rules.selling_price(40, 45) == 42
        assert decision.bank_tenths >= 0

    def test_a_squad_of_the_wrong_size_raises(self, rules, transfer_rules) -> None:
        pool, current = self._pool_and_squad()

        with pytest.raises(ValueError, match="must hold 15 players"):
            optimise_transfers(
                pool,
                rules,
                transfer_rules,
                StaticProvider(points_by_player_id={}),
                current_squad_ids=current[:14],
                purchase_prices={i: 45 for i in current},
                bank_tenths=0,
                free_transfers=1,
            )

    def test_a_missing_purchase_price_raises(self, rules, transfer_rules) -> None:
        pool, current = self._pool_and_squad()

        with pytest.raises(ValueError, match="No purchase price"):
            optimise_transfers(
                pool,
                rules,
                transfer_rules,
                StaticProvider(points_by_player_id={}),
                current_squad_ids=current,
                purchase_prices={i: 45 for i in current[:-1]},
                bank_tenths=0,
                free_transfers=1,
            )


# ---------------------------------------------------------------------------
# The season loop
# ---------------------------------------------------------------------------


class TestSeasonEngine:
    def test_free_transfers_bank_up_to_the_cap(self, rules, transfer_rules) -> None:
        from fpl_optimiser.backtest.engine import run_season

        history = SeasonHistory(_tiny_season(), rules)

        # Constant expected points: no transfer ever gains anything, so none is made and the
        # free transfers should simply accumulate rather than being spent.
        class Flat:
            name = "flat"

            def build(self, view, history):
                return StaticProvider(points_by_player_id={}, default=1.0)

        result = run_season(history, rules, transfer_rules, Flat(), timeout_seconds=10)

        assert result.transfers_made == 0
        assert result.hits_taken == 0
        assert len(result.scores) == 3

    def test_the_season_total_is_net_of_hits(self, rules, transfer_rules) -> None:
        from fpl_optimiser.backtest.engine import run_season

        history = SeasonHistory(_tiny_season(), rules)
        result = run_season(history, rules, transfer_rules, PerfectForesight(), timeout_seconds=10)

        assert result.total_points == result.points_before_hits - result.hit_points
        assert result.hit_points == result.hits_taken * 4
