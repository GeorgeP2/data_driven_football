import pandas as pd
import pytest
from football_stats.provisional import match_names, provisional_results

TEAMS = {1: "Arsenal", 2: "Coventry City", 3: "Man Utd", 4: "Spurs"}


def results():
    return pd.DataFrame(
        {
            "season": "2026-27",
            "date": pd.to_datetime(["2026-08-15", "2026-08-15"]),
            "home": ["Arsenal", "Man Utd"],
            "away": ["Coventry", "Spurs"],
            "hg": [1, 0],
            "ag": [0, 0],
        }
    )


def fixture(home, away, score=(None, None), finished=False, provisional=False):
    return {
        "team_h": home,
        "team_a": away,
        "team_h_score": score[0],
        "team_a_score": score[1],
        "finished": finished,
        "finished_provisional": provisional,
        "kickoff_time": "2026-11-22T14:00:00Z",
    }


def test_match_names():
    known = ["Arsenal", "Coventry", "Man Utd", "Man City"]
    assert match_names(["Arsenal", "Coventry City"], known) == {
        "Arsenal": "Arsenal",
        "Coventry City": "Coventry",
    }
    with pytest.raises(ValueError, match="Sheffield Utd"):
        match_names(["Sheffield Utd"], known)


def test_only_finished_matches_not_already_in_the_results():
    fixtures = [
        fixture(1, 2, (1, 0), finished=True),  # already in the results
        fixture(2, 1, (3, 2), provisional=True),  # finished today
        fixture(4, 3, (2, 0)),  # still in play
        fixture(3, 1),  # not started
    ]
    extra = provisional_results(fixtures, TEAMS, results())
    assert extra.to_dict("records") == [
        {
            "season": "2026-27",
            "date": pd.Timestamp("2026-11-22"),
            "home": "Coventry",
            "away": "Arsenal",
            "hg": 3,
            "ag": 2,
        }
    ]
