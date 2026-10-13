"""Render backtest results."""

from __future__ import annotations

from collections.abc import Sequence

from fpl_optimiser.backtest.engine import SeasonResult


def format_results(results: Sequence[SeasonResult], season: str) -> str:
    """A table of every provider's season, best first."""
    if not results:
        return "No results."

    ranked = sorted(results, key=lambda r: -r.total_points)
    ceiling = max(r.total_points for r in ranked)

    lines = [
        f"Season {season} — {len(ranked[0].scores)} gameweeks",
        "",
        f"  {'provider':<20} {'points':>7} {'gross':>7} {'hits':>6} "
        f"{'transfers':>10} {'subs':>6} {'% of best':>10}",
        f"  {'-' * 20} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 10} {'-' * 6} {'-' * 10}",
    ]
    for result in ranked:
        share = 100 * result.total_points / ceiling if ceiling else 0.0
        lines.append(
            f"  {result.provider:<20} {result.total_points:>7} "
            f"{result.points_before_hits:>7} {-result.hit_points:>6} "
            f"{result.transfers_made:>10} {result.auto_subs:>6} {share:>9.1f}%"
        )

    lines.append("")
    lines.append(
        "  points = net (after hits). gross = before hits. subs = auto-substitutions made."
    )
    return "\n".join(lines)
