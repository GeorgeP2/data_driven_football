"""Utilities for loading and analysing Fantasy Premier League entry history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

from fpl_optimiser.leagues.history import pull_fpl_history

REQUIRED_HISTORY_COLUMNS = {
    "event",
    "points",
    "total_points",
    "team_id",
    "name",
}


def _validate_team_map(team_map: dict[int, str]) -> None:
    if not team_map:
        raise ValueError("team_map must not be empty")
    invalid_items = [
        (team_id, name)
        for team_id, name in team_map.items()
        if not isinstance(team_id, int) or not isinstance(name, str)
    ]
    if invalid_items:
        raise ValueError("team_map must map int team_id values to string names")


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")


def load_histories(
    team_map: dict[int, str],
    output_dir: str = ".",
    refresh: bool = True,
) -> pd.DataFrame:
    """Load history for each team into one DataFrame.

    Args:
        team_map: Map of FPL team_id to display name.
        output_dir: Directory containing generated CSV files.
        refresh: If True, call the FPL API before reading each CSV.
    """
    _validate_team_map(team_map)
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for team_id, name in team_map.items():
        if refresh:
            pull_fpl_history(team_id, output_dir=output_dir)

        csv_path = base_path / f"fpl_entry_{team_id}_history.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing history CSV for team_id={team_id}: {csv_path}")

        team_df = pd.read_csv(csv_path)
        team_df["team_id"] = team_id
        team_df["name"] = name
        frames.append(team_df)

    history_df = pd.concat(frames, ignore_index=True)
    _validate_columns(history_df, REQUIRED_HISTORY_COLUMNS)

    history_df["event"] = pd.to_numeric(history_df["event"], errors="raise").astype(int)
    history_df["points"] = pd.to_numeric(history_df["points"], errors="raise").astype(float)
    history_df["total_points"] = pd.to_numeric(history_df["total_points"], errors="raise").astype(
        float
    )

    history_df = history_df.sort_values(["team_id", "event"]).reset_index(drop=True)
    return history_df


def add_features(df: pd.DataFrame, ewma_span: int = 5) -> pd.DataFrame:
    """Add derived columns used by exploratory analysis and simulation."""
    _validate_columns(df, REQUIRED_HISTORY_COLUMNS)
    if ewma_span < 1:
        raise ValueError("ewma_span must be >= 1")

    feature_df = df.copy()
    feature_df = feature_df.sort_values(["team_id", "event"]).reset_index(drop=True)

    feature_df["leaders_points"] = feature_df.groupby("event")["total_points"].transform("max")
    feature_df["points_from_leader"] = feature_df["leaders_points"] - feature_df["total_points"]
    feature_df["ewma_points"] = feature_df.groupby("team_id")["points"].transform(
        lambda values: values.ewm(span=ewma_span, adjust=False).mean()
    )
    feature_df["running_mean_points"] = feature_df.groupby("team_id")["points"].transform(
        lambda values: values.expanding().mean()
    )

    return feature_df


WEEKLY_POINTS_SD = 16.0
"""Week-to-week spread of one manager's score. In the 2025-26 Boys league every manager's
weekly standard deviation was 15-18 at a mean of 55-60 — more than double the sqrt(mean) ≈ 7.5
a Poisson allows, which is why the Poisson version called a 20-point lead at GW3 a certainty."""

PRIOR_GAMEWEEKS = 8
"""How many gameweeks of league-average scoring a manager's own record is weighed against.
Three good weeks are mostly luck; thirty are mostly the manager. Also caps how confident the
rate estimate can be early on: its standard error is WEEKLY_POINTS_SD / sqrt(played + prior)."""


def _validate_simulation_args(
    total_gws: int, n_iter: int, weekly_sd: float, prior_gameweeks: int
) -> None:
    if total_gws < 1:
        raise ValueError("total_gws must be >= 1")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if weekly_sd <= 0:
        raise ValueError("weekly_sd must be > 0")
    if prior_gameweeks < 0:
        raise ValueError("prior_gameweeks must be >= 0")


def _remaining_points(
    rng: np.random.Generator,
    observed_mean: float,
    games_observed: int,
    remaining_games: int,
    n_iter: int,
    *,
    league_mean: float,
    weekly_sd: float,
    prior_gameweeks: int,
) -> np.ndarray:
    """Draw one manager's points over the remaining games, `n_iter` times.

    Two sources of uncertainty a fixed-rate Poisson ignores. The manager's true weekly rate is
    only *estimated*: it is shrunk toward the league mean by `prior_gameweeks` worth of prior
    and drawn with a standard error that falls as games are played. And week-to-week scores
    vary far more than a Poisson allows, so the noise is normal at `weekly_sd`.
    """
    if remaining_games == 0:
        return np.zeros(n_iter, dtype=float)
    weight = games_observed + prior_gameweeks
    if weight == 0:
        raise ValueError("Cannot project with no games observed and no prior")
    shrunk_mean = (games_observed * observed_mean + prior_gameweeks * league_mean) / weight
    rate = rng.normal(shrunk_mean, weekly_sd / np.sqrt(weight), size=n_iter)
    total = rng.normal(remaining_games * rate, np.sqrt(remaining_games) * weekly_sd, size=n_iter)
    return np.clip(total, 0.0, None)


def simulate_season_finish(
    df: pd.DataFrame,
    team_map: dict[int, str],
    total_gws: int = 38,
    n_iter: int = 10000,
    seed: int = 42,
    recent_window: int = 5,
    weekly_sd: float = WEEKLY_POINTS_SD,
    prior_gameweeks: int = PRIOR_GAMEWEEKS,
) -> pd.DataFrame:
    """Simulate final points totals, with the manager's rate shrunk and uncertain."""
    _validate_team_map(team_map)
    _validate_columns(df, REQUIRED_HISTORY_COLUMNS)
    _validate_simulation_args(total_gws, n_iter, weekly_sd, prior_gameweeks)
    if recent_window < 1:
        raise ValueError("recent_window must be >= 1")

    rng = np.random.default_rng(seed)
    league_mean = float(df.loc[df["team_id"].isin(team_map), "points"].mean())
    iterations = np.arange(n_iter)
    frames: list[pd.DataFrame] = []

    for team_id, name in team_map.items():
        team_df = df.loc[df["team_id"] == team_id].sort_values("event")
        if team_df.empty:
            raise ValueError(f"No rows found for team_id={team_id}")

        current_points = float(team_df["total_points"].iloc[-1])
        games_played = int(team_df["event"].iloc[-1])
        remaining_games = max(total_gws - games_played, 0)
        recent = team_df["points"].tail(recent_window)

        remaining_points = _remaining_points(
            rng,
            float(team_df["points"].mean()),
            len(team_df),
            remaining_games,
            n_iter,
            league_mean=league_mean,
            weekly_sd=weekly_sd,
            prior_gameweeks=prior_gameweeks,
        )
        remaining_points_recent = _remaining_points(
            rng,
            float(recent.mean()),
            len(recent),
            remaining_games,
            n_iter,
            league_mean=league_mean,
            weekly_sd=weekly_sd,
            prior_gameweeks=prior_gameweeks,
        )

        frames.append(
            pd.DataFrame(
                {
                    "iteration": iterations,
                    "team_id": team_id,
                    "name": name,
                    "current_points": current_points,
                    "remaining_points": remaining_points,
                    "final_points": current_points + remaining_points,
                    "remaining_points_recent": remaining_points_recent,
                    "final_points_recent": current_points + remaining_points_recent,
                }
            )
        )

    sim_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["iteration", "team_id"], kind="stable")
        .reset_index(drop=True)
    )
    sim_df["final_rank"] = sim_df.groupby("iteration")["final_points"].rank(
        ascending=False, method="min"
    )
    sim_df["final_rank_recent"] = sim_df.groupby("iteration")["final_points_recent"].rank(
        ascending=False, method="min"
    )
    return sim_df


def simulate_win_prob_by_gw(
    df: pd.DataFrame,
    team_map: dict[int, str],
    total_gws: int = 38,
    n_iter: int = 10000,
    seed: int = 42,
    weekly_sd: float = WEEKLY_POINTS_SD,
    prior_gameweeks: int = PRIOR_GAMEWEEKS,
    rate_column: str = "running_mean_points",
    rate_games: int | None = None,
) -> pd.DataFrame:
    """Estimate title win probability as of each gameweek, from the record to that point.

    Args:
        rate_column: The per-row estimate of a manager's weekly scoring rate. The season
            running mean by default; `ewma_points` gives a form-weighted rate that reacts to a
            change in a manager's level within a few weeks rather than half a season.
        rate_games: How many games of evidence `rate_column` is worth when shrinking toward
            the league mean. None means every game played so far, right for a running mean.
            An EWMA forgets, so pass its span — otherwise a five-week form line would be
            trusted as if it were thirty weeks of data.
    """
    _validate_team_map(team_map)
    required = REQUIRED_HISTORY_COLUMNS.union({rate_column})
    _validate_columns(df, required)
    _validate_simulation_args(total_gws, n_iter, weekly_sd, prior_gameweeks)
    if rate_games is not None and rate_games < 1:
        raise ValueError("rate_games must be >= 1 or None")

    rng = np.random.default_rng(seed)
    team_items = list(team_map.items())
    league_df = df.loc[df["team_id"].isin(team_map)]
    gameweeks = sorted(pd.unique(league_df["event"]))

    output_rows: list[dict[str, float | int]] = []
    for game_week in gameweeks:
        gw_rows = league_df.loc[league_df["event"] == game_week]
        # What the league as a whole has averaged up to this gameweek — the shrinkage target.
        league_mean = float(league_df.loc[league_df["event"] <= game_week, "points"].mean())
        remaining_games = max(total_gws - int(game_week), 0)
        final_points = np.zeros((n_iter, len(team_items)), dtype=float)

        for idx, (team_id, _) in enumerate(team_items):
            team_row = gw_rows.loc[gw_rows["team_id"] == team_id].tail(1)
            if team_row.empty:
                current_points, running_mean, games_observed = 0.0, league_mean, 0
            else:
                current_points = float(team_row["total_points"].iloc[-1])
                running_mean = float(team_row[rate_column].iloc[-1])
                games_played = int(
                    (league_df["team_id"] == team_id).__and__(league_df["event"] <= game_week).sum()
                )
                games_observed = (
                    games_played if rate_games is None else min(games_played, rate_games)
                )
            final_points[:, idx] = current_points + _remaining_points(
                rng,
                running_mean,
                games_observed,
                remaining_games,
                n_iter,
                league_mean=league_mean,
                weekly_sd=weekly_sd,
                prior_gameweeks=prior_gameweeks,
            )

        winners = np.argmax(final_points, axis=1)
        row: dict[str, float | int] = {"game_week": int(game_week)}
        for idx, (_, name) in enumerate(team_items):
            row[f"{name}_win_prob"] = float((winners == idx).mean())
        output_rows.append(row)

    return pd.DataFrame(output_rows).sort_values("game_week").reset_index(drop=True)


def win_prob_deltas(win_prob_df: pd.DataFrame, n_iter: int, alpha: float = 0.05) -> pd.DataFrame:
    """Week-on-week change in each manager's win probability, zeroed where it is noise.

    Each probability is the mean of `n_iter` Bernoulli draws, so two consecutive weeks are two
    samples and their difference has sampling error of order sqrt(2 p (1 - p) / n_iter). A Welch
    t-test on the two samples (reconstructed from their means, since a 0/1 sample's variance is
    p (1 - p)) decides whether the change is distinguishable from a rerun with another seed;
    if not, the delta is reported as 0.

    Returns one row per (game_week, name) with the raw delta, the p-value, and the zeroed delta.
    """
    if n_iter < 2:
        raise ValueError("n_iter must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    prob_columns = [c for c in win_prob_df.columns if c.endswith("_win_prob")]
    if not prob_columns:
        raise ValueError("win_prob_df has no *_win_prob columns")

    ordered = win_prob_df.sort_values("game_week").reset_index(drop=True)
    rows: list[dict[str, float | int | str]] = []
    for column in prob_columns:
        name = column[: -len("_win_prob")]
        probs = ordered[column].to_numpy(dtype=float)
        for index in range(1, len(ordered)):
            before, after = probs[index - 1], probs[index]
            delta = after - before
            std_before = np.sqrt(before * (1.0 - before))
            std_after = np.sqrt(after * (1.0 - after))
            if std_before == 0.0 and std_after == 0.0:
                # Both weeks were unanimous: any difference is exact, no difference is exact.
                p_value = 0.0 if delta != 0.0 else 1.0
            else:
                p_value = float(
                    ttest_ind_from_stats(
                        after,
                        std_after,
                        n_iter,
                        before,
                        std_before,
                        n_iter,
                        equal_var=False,
                    ).pvalue
                )
            rows.append(
                {
                    "game_week": int(ordered.loc[index, "game_week"]),
                    "name": name,
                    "delta": delta,
                    "p_value": p_value,
                    "delta_significant": delta if p_value <= alpha else 0.0,
                }
            )
    return pd.DataFrame(rows)


def add_required_run_rate(df: pd.DataFrame, total_gws: int = 38) -> pd.DataFrame:
    """Points per week each manager needs, from here, to catch the leader by the last gameweek.

    Catching up means matching the leader's future scoring *and* closing the gap, so the
    required rate is the leader's running mean plus the gap spread over the games left. It is
    NaN on the final gameweek (nothing left to play) and for the leader it is their own mean.
    Compare it with the manager's own running mean: the week the required rate climbs past
    anything they have ever averaged is the week the race was really over.
    """
    _validate_columns(df, REQUIRED_HISTORY_COLUMNS.union({"running_mean_points"}))
    if total_gws < 1:
        raise ValueError("total_gws must be >= 1")
    out = df.sort_values(["event", "team_id"]).copy()

    leader_index = out.groupby("event")["total_points"].idxmax()
    leader_mean = out.loc[leader_index, ["event", "running_mean_points"]].rename(
        columns={"running_mean_points": "leader_running_mean"}
    )
    out = out.merge(leader_mean, on="event", how="left")
    if out["leader_running_mean"].isna().any():
        raise ValueError("Every gameweek must have a leader; the merge left gaps")

    gap = out.groupby("event")["total_points"].transform("max") - out["total_points"]
    remaining = (total_gws - out["event"]).astype(float)
    out["required_rate"] = out["leader_running_mean"] + gap / remaining.where(remaining > 0)
    return out.sort_values(["team_id", "event"]).reset_index(drop=True)
