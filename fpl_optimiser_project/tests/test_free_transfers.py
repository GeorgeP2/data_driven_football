"""The free-transfer count the planner prices every hit against.

It used to be a hand-typed `--free-transfers 1`, which is the worst kind of wrong input: the
planner believes it, quotes a -4 on a move that is actually free, and nothing anywhere errors.
The count is fully derivable from the public history endpoint, so these tests pin the replay —
in particular the two rules that are easy to implement backwards.
"""

from __future__ import annotations

from typing import Any

import pytest
from fpl_optimiser import live
from fpl_optimiser.rules import TransferRules

RULES = TransferRules(sell_on_fee=0.5, max_free_transfers=5)
"""The live 2026-27 rules: one a week, bank up to five."""

OLD_RULES = TransferRules(sell_on_fee=0.5, max_free_transfers=2)
"""Before 2024-25 the cap was two, which is why it is read from the API and not assumed."""


def _history(
    transfers: dict[int, int], chips: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "current": [
            {"event": gw, "event_transfers": count} for gw, count in sorted(transfers.items())
        ],
        "chips": chips or [],
    }


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    """Serve a canned history in place of the endpoint."""

    def _serve(payload: dict[str, Any]) -> None:
        monkeypatch.setattr(live, "_get", lambda *args, **kwargs: payload)

    return _serve


class TestTheReplay:
    def test_the_first_gameweek_opens_on_one(self, api) -> None:
        api(_history({}))

        assert live.free_transfers_for(1, gameweek=1, transfer_rules=RULES) == 1

    def test_the_opening_gameweek_does_not_bank(self, api) -> None:
        # GW1 is unlimited squad creation, so a quiet GW1 leaves you on one in GW2, not two.
        # This is the off-by-one that inflated the count for a whole season.
        api(_history({1: 0}))

        assert live.free_transfers_for(1, gameweek=2, transfer_rules=RULES) == 1

    def test_an_unused_transfer_banks(self, api) -> None:
        api(_history({1: 0, 2: 0}))

        assert live.free_transfers_for(1, gameweek=3, transfer_rules=RULES) == 2

    def test_georges_actual_season_through_gw4(self, api) -> None:
        # Entry 552800, 2026-27: nothing in GW1-2, two moves in GW3, one in GW4. Cross-checked
        # against `entry/552800/` -> last_deadline_total_transfers == 3.
        api(_history({1: 0, 2: 0, 3: 2, 4: 1}))

        assert live.free_transfers_for(1, gameweek=5, transfer_rules=RULES) == 1

    def test_the_bank_is_capped(self, api) -> None:
        api(_history({gw: 0 for gw in range(1, 11)}))

        assert live.free_transfers_for(1, gameweek=11, transfer_rules=RULES) == 5

    def test_the_cap_comes_from_the_rules_not_a_constant(self, api) -> None:
        # Same history, older cap: the number must move with the season's rules.
        api(_history({gw: 0 for gw in range(1, 11)}))

        assert live.free_transfers_for(1, gameweek=11, transfer_rules=OLD_RULES) == 2

    def test_only_gameweeks_before_the_one_planned_are_counted(self, api) -> None:
        # GW4's two transfers have not happened yet when planning GW4.
        api(_history({1: 0, 2: 0, 3: 0, 4: 2}))

        assert live.free_transfers_for(1, gameweek=4, transfer_rules=RULES) == 3


class TestTheTwoRulesThatAreEasyToGetBackwards:
    def test_a_hit_floors_the_count_at_zero_rather_than_going_negative(self, api) -> None:
        # One banked, four taken: that is a -12, and it leaves you on zero, not on minus two.
        # Carrying the negative forward would suppress the next two weeks' transfers entirely.
        api(_history({1: 0, 2: 4}))

        assert live.free_transfers_for(1, gameweek=3, transfer_rules=RULES) == 1

    def test_a_wildcard_freezes_the_allowance(self, api) -> None:
        # A wildcard week reports a full slate of transfers, spends none of the bank — and
        # earns no new one either. Letting it accrue is the second off-by-one.
        api(_history({1: 0, 2: 0, 3: 11}, chips=[{"name": "wildcard", "event": 3}]))

        assert live.free_transfers_for(1, gameweek=4, transfer_rules=RULES) == 2

    def test_a_free_hit_freezes_it_too(self, api) -> None:
        api(_history({1: 0, 2: 0, 3: 11}, chips=[{"name": "freehit", "event": 3}]))

        assert live.free_transfers_for(1, gameweek=4, transfer_rules=RULES) == 2

    def test_a_bench_boost_is_not_a_transfer_chip(self, api) -> None:
        # Only wildcard and free hit are exempt; a bench boost week spends transfers normally.
        api(_history({1: 0, 2: 0, 3: 2}, chips=[{"name": "bboost", "event": 3}]))

        assert live.free_transfers_for(1, gameweek=4, transfer_rules=RULES) == 1


class TestBadInput:
    def test_a_gameweek_below_one_is_rejected(self, api) -> None:
        api(_history({}))

        with pytest.raises(ValueError, match="gameweek must be >= 1"):
            live.free_transfers_for(1, gameweek=0, transfer_rules=RULES)

    def test_a_missing_entry_names_the_team(self, api) -> None:
        api({})

        with pytest.raises(RuntimeError, match="No history found for team 999"):
            live.free_transfers_for(999, gameweek=5, transfer_rules=RULES)

    def test_an_entry_that_joined_late_does_not_accrue_before_it_started(self, api) -> None:
        # Joining at GW10 does not hand you ten banked transfers.
        api(_history({10: 0, 11: 0}))

        assert live.free_transfers_for(1, gameweek=12, transfer_rules=RULES) == 2


class TestAgainstRealEntriesThatPaidHits:
    """The evidence, not the rulebook.

    Every case below is a completed gameweek where a real entry paid a known hit. The hit pins
    the free-transfer count exactly — `cost == 4 * (transfers - free)` — so each of these fails
    on the pre-fix code, which counted one too many. Encoded as literals because the downloaded
    histories are `*.json` and the repo blanket-ignores those.
    """

    def test_entry_6589563_paid_four_for_two_transfers_in_gw2(self, api) -> None:
        # 2 transfers, -4. That is one free and one bought, so GW2 opened on one.
        api(_history({1: 0, 2: 2}))

        assert live.free_transfers_for(6589563, gameweek=2, transfer_rules=RULES) == 1

    def test_entry_2512675_paid_twelve_for_four_transfers_in_gw2(self, api) -> None:
        # 4 transfers, -12: one free, three bought. Same conclusion, larger margin.
        api(_history({1: 0, 2: 4}))

        assert live.free_transfers_for(2512675, gameweek=2, transfer_rules=RULES) == 1

    def test_entry_825485_went_into_gw6_on_one_after_two_chip_weeks(self, api) -> None:
        # Wildcard GW4, free hit GW5, then 3 transfers in GW6 for -8 — one free, two bought.
        # If chip weeks accrued, it would have gone into GW6 on three and paid nothing.
        api(
            _history(
                {1: 0, 2: 1, 3: 1, 4: 0, 5: 0},
                chips=[
                    {"name": "wildcard", "event": 4},
                    {"name": "freehit", "event": 5},
                ],
            )
        )

        assert live.free_transfers_for(825485, gameweek=6, transfer_rules=RULES) == 1
