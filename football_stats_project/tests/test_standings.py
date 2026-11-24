import pandas as pd
from football_stats.standings import (
    COLUMNS,
    clean_results,
    is_complete,
    season_code,
    season_label,
    standings,
    table_now,
)

# Three teams, double round robin: A wins every match, B beats C twice.
MATCHES = [
    ("2020-08-01", "A", "B", 2, 0),
    ("2020-08-08", "C", "A", 0, 1),
    ("2020-08-15", "B", "C", 3, 1),
    ("2020-08-22", "B", "A", 0, 1),
    ("2020-08-29", "A", "C", 4, 0),
    ("2020-09-05", "C", "B", 0, 1),
]


def results(matches=MATCHES, season="2020-21"):
    df = pd.DataFrame(matches, columns=["date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "season", season)
    return df


def row(table, team, gameweek):
    return table[(table["team"] == team) & (table["gameweek"] == gameweek)].iloc[0]


def test_season_names():
    assert season_code(1999) == "9900"
    assert season_code(2024) == "2425"
    assert season_label(1999) == "1999-00"


def test_one_row_per_team_and_gameweek():
    table = standings(results())
    assert list(table.columns) == COLUMNS
    assert len(table) == 3 * 4
    assert not table.duplicated(["season", "team", "gameweek"]).any()


def test_points_accumulate():
    table = standings(results())
    assert [row(table, "A", gw)["points"] for gw in range(1, 5)] == [3, 6, 9, 12]
    assert [row(table, "B", gw)["points"] for gw in range(1, 5)] == [0, 3, 3, 6]
    assert [row(table, "C", gw)["points"] for gw in range(1, 5)] == [0, 0, 0, 0]


def test_positions_and_final_position():
    table = standings(results())
    # after one match each: A won 2-0, B lost 0-2 (gd -2), C lost 0-1 (gd -1)
    assert row(table, "A", 1)["position"] == 1
    assert row(table, "C", 1)["position"] == 2
    assert row(table, "B", 1)["position"] == 3
    assert row(table, "B", 4)["position"] == 2
    assert table.groupby("team")["final_position"].unique().map(list).to_dict() == {
        "A": [1],
        "B": [2],
        "C": [3],
    }


def test_deduction_applies_from_the_gameweek_it_was_imposed():
    # A loses 10 points between its second and third matches
    table = standings(results(), deductions=[("2020-21", "A", "2020-08-10", -10)])
    assert [row(table, "A", gw)["points"] for gw in range(1, 5)] == [3, 6, -1, 2]
    assert [row(table, "A", gw)["position"] for gw in range(1, 5)] == [1, 1, 3, 2]
    assert row(table, "A", 1)["final_position"] == 2
    assert row(table, "B", 1)["final_position"] == 1
    # a deduction in another season changes nothing
    other = standings(results(), deductions=[("2019-20", "A", "2019-08-10", -10)])
    pd.testing.assert_frame_equal(other, standings(results(), deductions=[]))


def test_goal_difference_accumulates():
    table = standings(results())
    assert [row(table, "A", gw)["goal_difference"] for gw in range(1, 5)] == [2, 3, 4, 8]
    assert [row(table, "C", gw)["goal_difference"] for gw in range(1, 5)] == [-1, -3, -7, -8]


def test_gap_to_17th():
    # 18 teams, one match each: T01 beats T02 1-0, T03 beats T04 2-0, ... T17 beats T18 9-0
    names = [f"T{i:02d}" for i in range(1, 19)]
    matches = [("2020-08-01", names[i], names[i + 1], i // 2 + 1, 0) for i in range(0, 18, 2)]
    table = standings(results(matches))
    # nine winners on 3 points fill 1st to 9th, so 17th has 0 points
    assert set(table.loc[table["position"] <= 9, "gap_to_17th"]) == {3}
    assert set(table.loc[table["position"] >= 10, "gap_to_17th"]) == {0}
    # with a 5-point deduction the bottom team is 5 behind 17th
    docked = standings(results(matches), deductions=[("2020-21", "T18", "2020-08-01", -5)])
    assert row(docked, "T18", 1)["position"] == 18
    assert row(docked, "T18", 1)["gap_to_17th"] == -5
    # fewer than 17 teams: no 17th place to measure against
    assert standings(results())["gap_to_17th"].isna().all()


def test_table_now_ranks_teams_on_unequal_matches_played():
    # A has played three, B three, C two
    table = table_now(results(MATCHES[:4]))
    assert list(table["team"]) == ["A", "B", "C"]
    assert list(table["gameweek"]) == [3, 3, 2]
    assert list(table["points"]) == [9, 3, 0]
    assert list(table["position"]) == [1, 2, 3]
    # only the latest season counts
    both = pd.concat([results(MATCHES[:4], season="2019-20"), results()], ignore_index=True)
    assert set(table_now(both)["gameweek"]) == {4}


def test_margins_measure_the_distance_to_the_line():
    # six teams, three matches: winners by 3, 2 and 1 goals, so the table is A C E / F D B
    matches = [
        ("2020-08-01", "A", "B", 3, 0),
        ("2020-08-01", "C", "D", 2, 0),
        ("2020-08-01", "E", "F", 1, 0),
    ]
    table = standings(results(matches)).set_index("team")
    assert list(table.index) == ["A", "C", "E", "F", "D", "B"]
    # leader level on points with 2nd; everyone else measured against the leader
    assert list(table["title_margin"]) == [0, 0, 0, -3, -3, -3]
    # top four hold F on 0 points; 5th is D on 0: lead over 5th, or gap to 4th
    assert list(table["top4_margin"]) == [3, 3, 3, 0, 0, 0]
    assert list(table_now(results(matches))["title_margin"]) == [0, 0, 0, -3, -3, -3]


def test_last_season_finish():
    # C is replaced by D in the second season
    second = [
        (d.replace("2020", "2021"), h.replace("C", "D"), a.replace("C", "D"), hg, ag)
        for d, h, a, hg, ag in MATCHES
    ]
    both = pd.concat([results(), results(second, season="2021-22")], ignore_index=True)
    table = standings(both)
    first = table[table["season"] == "2020-21"]
    assert first["last_season_finish"].isna().all()
    last = table[table["season"] == "2021-22"].groupby("team")["last_season_finish"].first()
    assert last["A"] == 1
    assert last["B"] == 2
    assert pd.isna(last["D"])
    now = table_now(both).set_index("team")["last_season_finish"]
    assert now["A"] == 1
    assert pd.isna(now["D"])


def test_goals_scored_splits_equal_goal_difference():
    table = standings(results([("2020-08-01", "A", "B", 1, 1), ("2020-08-01", "C", "D", 2, 2)]))
    assert list(table["team"]) == ["C", "D", "A", "B"]
    assert list(table["position"]) == [1, 2, 3, 4]


def test_unfinished_season_has_no_final_position():
    table = standings(results(MATCHES[:4]))
    assert table["final_position"].isna().all()
    # A and B have played three, C only two: gameweek 3 is not a full table yet
    assert table.loc[table["gameweek"] == 3, "position"].isna().all()
    assert table.loc[table["gameweek"] <= 2, "position"].notna().all()


def test_seasons_are_ranked_separately():
    both = pd.concat([results(), results(season="2021-22")], ignore_index=True)
    table = standings(both)
    assert len(table) == 2 * 3 * 4
    assert (table.groupby(["season", "gameweek"])["position"].max() == 3).all()


def test_clean_results_drops_unplayed_matches():
    raw = pd.DataFrame(
        {
            "Date": ["16/08/2024", "17/08/24", "18/08/2024"],
            "HomeTeam": ["Man United", "Ipswich", "Arsenal"],
            "AwayTeam": ["Fulham", "Liverpool", "Wolves"],
            "FTHG": [1, 0, None],
            "FTAG": [0, 2, None],
        }
    )
    out = clean_results(raw, "2024-25")
    assert list(out["home"]) == ["Man Utd", "Ipswich"]
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-08-16", "2024-08-17"]
    assert not is_complete(out)
    assert is_complete(results())
