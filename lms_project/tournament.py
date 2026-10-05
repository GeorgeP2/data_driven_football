#!/usr/bin/env python3
"""
Last-Man-Standing tournament: strategies play EACH OTHER in the same pool, on real EPL results
and real pre-match odds (same data and no-look-ahead rules as backtest.py).

Every (season, start gameweek) is one pool. Each entrant has one seat, pays a stake, picks one
unused team per gameweek (draw = out) and sees what the others have used. If everyone still in
goes out in the same week, the pool resets: everyone rebuys, the pot rolls over, used lists clear.
A sole survivor takes the pot; pools still alive at season end split it.

Entrants are named by config, so you can field several versions of each strategy:
  random                 uniformly random unused team
  crowd_k<k>             picks with probability ~ P(win)^k (a model of typical colleagues)
  greedy[_w<w>]          best P(win) this week; w = weight on odds vs Dixon-Coles (default 1)
  greedy_eps<e>_h<H>     among teams within e of the best P(win), take the one whose best
                         fixture in the next H weeks is weakest (greedy that conserves)
  fav_p<p>[_k<k>][_w<w>] backs this week's favourite, but with probability p skips it and picks
                         another unused team: uniformly at random, or ~ P(win)^k if k is given
  plan_h<H>[_w<w>]       rolling-horizon assignment over H weeks, play week 1
  mc_h<H>_k<k>[_w<w>]    plan shortlist scored by Monte Carlo pot equity, assuming the other
                         players pick ~ P(win)^k   (backtest.py's plan_mc)

Usage
  python tournament.py                                    # default 14-entrant field, 2002/03-2025/26
  python tournament.py --entrants greedy plan_h2 plan_h4 mc_h4_k3 crowd_k3 crowd_k3 crowd_k3
  python tournament.py --pool-size 8 --lineups 5          # random 8-player lineups from the roster
  python tournament.py --field crowd_k1.5 crowd_k1.5 crowd_k1.5 crowd_k3 crowd_k3 crowd_k3 \
      crowd_k3 crowd_k3 crowd_k3 crowd_k3 --pool-size 4      # 10 colleague-like seats + 4 contestants

Results depend heavily on the field: near-identical entrants (e.g. several favourite-backers)
survive and die together, so none of them becomes the sole survivor.
"""
import argparse
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import backtest as B

DEFAULT_ROSTER = [
    "random", "crowd_k1.5", "crowd_k3",
    "greedy", "greedy_w0.5", "greedy_eps0.03_h4", "greedy_eps0.08_h4",
    "plan_h2", "plan_h4", "plan_h8",
    "mc_h2_k3", "mc_h4_k1.5", "mc_h4_k3", "mc_h4_k5",
]


# --------------------------------------------------------------------------- #
# Entrants
# --------------------------------------------------------------------------- #
@dataclass
class Entrant:
    name: str
    fn: Callable
    w: float = 1.0  # odds weight -> which Season object the strategy sees


def strat_random(S, g, used, state):
    """Uniformly random unused team with a match this week (seat's own rng: reproducible)."""
    ok = np.flatnonzero(B._avail(S, used) & (S.Pnow[g] > 0))
    return int(state["rng_me"].choice(ok)) if len(ok) else None


def make_greedy_eps(eps, H):
    def f(S, g, used, state):
        P = S.forecast(g, H)
        avail = B._avail(S, used)
        p0 = P[0] * avail
        if p0.max() <= 0:
            return None
        near = np.flatnonzero(p0 >= p0.max() - eps)
        future = P[1:, near].max(axis=0) if P.shape[0] > 1 else np.zeros(len(near))
        return int(near[np.argmin(future)])
    return f


def make_fav(p_skip, k=0.0):
    def f(S, g, used, state):
        p = S.Pnow[g] * B._avail(S, used)
        if p.max() <= 0:
            return None
        fav = int(p.argmax())
        rng = state["rng_me"]
        if rng.random() >= p_skip:
            return fav
        others = np.flatnonzero(p > 0)
        others = others[others != fav]
        if len(others) == 0:
            return fav
        wts = np.power(p[others], k)
        return int(rng.choice(others, p=wts / wts.sum()))
    return f


def parse_entrant(name, mc_top, mc_sims):
    opts = dict(re.findall(r"_([a-z]+)([0-9.]+)", name))
    w = float(opts.get("w", 1.0))
    kind = name.split("_")[0]
    if name == "random":
        return Entrant(name, strat_random)
    if kind == "crowd":
        return Entrant(name, B.make_crowd(float(opts["k"])))
    if kind == "greedy" and "eps" in opts:
        return Entrant(name, make_greedy_eps(float(opts["eps"]), int(opts.get("h", 4))), w)
    if kind == "greedy":
        return Entrant(name, B.strat_greedy, w)
    if kind == "fav":
        return Entrant(name, make_fav(float(opts["p"]), float(opts.get("k", 0.0))), w)
    if kind == "plan":
        return Entrant(name, B.make_plan(int(opts["h"])), w)
    if kind == "mc":
        return Entrant(name, B.make_plan_mc(int(opts["h"]), mc_top, mc_sims, k=float(opts["k"])), w)
    raise ValueError(f"unknown entrant {name!r} (see --help)")


# --------------------------------------------------------------------------- #
# One pool
# --------------------------------------------------------------------------- #
def run_pool(seasons, entrants, start, seed):
    """seasons: {odds_weight: Season}. Returns per-seat result dicts."""
    S0 = next(iter(seasons.values()))
    n, T = len(entrants), S0.T
    used = np.zeros((n, T), bool)
    alive = np.ones(n, bool)
    stake = np.ones(n)
    out_week = np.full(n, np.inf)  # week knocked out in the current cycle
    weeks_won = np.zeros(n, int)
    pot, resets, winner = float(n), 0, None
    rngs = [np.random.default_rng([seed, i]) for i in range(n)]

    for g in range(start, S0.G + 1):
        used[used.all(axis=1)] = False  # all teams used -> list resets
        picks = {}
        for i in np.flatnonzero(alive):
            others = np.arange(n) != i
            state = {"rng_me": rngs[i], "N": n, "pot": pot,
                     "alive_opp": alive[others], "used_opp": used[others]}
            e = entrants[i]
            picks[i] = e.fn(seasons[e.w], g, set(np.flatnonzero(used[i])), state)
        for i, t in picks.items():  # simultaneous: apply after everyone has picked
            if t is not None and S0.win[g, t]:
                used[i, t] = True
                weeks_won[i] += 1
            else:
                alive[i] = False
                out_week[i] = g
        if not alive.any():  # everyone out together -> reset + rebuy
            resets += 1
            stake += 1.0
            pot += n
            alive[:] = True
            used[:] = False
            out_week[:] = np.inf
        elif alive.sum() == 1:
            winner = int(np.flatnonzero(alive)[0])
            break

    payout = np.zeros(n)
    if winner is not None:
        payout[winner] = pot
    else:
        payout[alive] = pot / alive.sum()
    # finishing position for head-to-head: sole winner > still alive at the end > later exit
    finish = np.where(alive, S0.G + 1.0, out_week)
    if winner is not None:
        finish[winner] = S0.G + 2.0
    return [{"seat": i, "entrant": entrants[i].name, "net": payout[i] - stake[i], "stake": stake[i],
             "sole_win": i == winner, "share": payout[i] / pot, "finish": finish[i],
             "resets": resets, "weeks_won": weeks_won[i]} for i in range(n)]


# --------------------------------------------------------------------------- #
# Worker: one season, all start gameweeks
# --------------------------------------------------------------------------- #
def process_season(job):
    code, hist, cur, cfg = job
    roster = [parse_entrant(nm, cfg["mc_top"], cfg["mc_sims"]) for nm in cfg["entrants"]]
    field = [parse_entrant(nm, cfg["mc_top"], cfg["mc_sims"]) for nm in cfg["field"]]
    weights = sorted({e.w for e in roster + field})
    seasons = {w: B.Season(code, hist, cur, cfg["refit_every"], cfg["hist_matches"], w) for w in weights}
    G = next(iter(seasons.values())).G
    rows = []
    for start in range(1, min(cfg["max_start"], G - 4) + 1):
        n_lineups = 1 if cfg["pool_size"] >= len(roster) else cfg["lineups"]
        for lu in range(n_lineups):
            seed = zlib.crc32(f"{code}-{start}-{lu}".encode())
            if cfg["pool_size"] >= len(roster):
                seats = list(range(len(roster)))
            else:
                seats = sorted(np.random.default_rng(seed).choice(len(roster), cfg["pool_size"], replace=False))
            for r in run_pool(seasons, field + [roster[s] for s in seats], start, seed):
                rows.append({"season": code, "start": start, "lineup": lu, "pool": f"{code}-{start}-{lu}", **r})
    return rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def leaderboard(df):
    rows = []
    for name, d in df.groupby("entrant"):
        rows.append({"entrant": name, "ROI": d.net.sum() / d.stake.sum(),
                     "ROI_se": B.boot_se(d, "net", "stake"),
                     "P(sole win)": d.sole_win.mean(), "pot share": d.share.mean(),
                     "weeks won": d.weeks_won.mean(), "pools": d["pool"].nunique()})
    return pd.DataFrame(rows).sort_values("ROI", ascending=False).reset_index(drop=True)


def head_to_head(df, order):
    """H[a, b] = share of shared pools where a finished ahead of b (ties count half)."""
    wins = pd.DataFrame(0.0, index=order, columns=order)
    games = pd.DataFrame(0.0, index=order, columns=order)
    pos = {nm: i for i, nm in enumerate(order)}
    W, G = np.zeros((len(order), len(order))), np.zeros((len(order), len(order)))
    for _, p in df.groupby("pool"):  # seat vs seat, so duplicate entrants count once per seat
        e = p.entrant.map(pos).values
        v = p.finish.values
        cmp = (v[:, None] > v[None, :]) + 0.5 * (v[:, None] == v[None, :])
        np.add.at(W, (e[:, None], e[None, :]), cmp)
        np.add.at(G, (e[:, None], e[None, :]), 1.0)
    wins.loc[:, :], games.loc[:, :] = W, G
    h = wins / games.replace(0, np.nan)
    for nm in order:
        h.loc[nm, nm] = np.nan
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrants", nargs="+", default=DEFAULT_ROSTER)
    ap.add_argument("--first", default="0203")
    ap.add_argument("--last", default="2526")
    ap.add_argument("--max-start", type=int, default=30, help="pools start at gameweeks 1..this")
    ap.add_argument("--field", nargs="*", default=[],
                    help="entrants seated in every pool in addition to the sampled contestants")
    ap.add_argument("--pool-size", type=int, default=99,
                    help="contestant seats per pool; smaller than the roster -> random lineups")
    ap.add_argument("--lineups", type=int, default=5, help="random lineups per start (with --pool-size)")
    ap.add_argument("--mc-top", type=int, default=4)
    ap.add_argument("--mc-sims", type=int, default=800)
    ap.add_argument("--refit-every", type=int, default=6)
    ap.add_argument("--hist-matches", type=int, default=900)
    ap.add_argument("--cache-dir", default="fd_cache")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="tournament_runs.csv")
    args = ap.parse_args()

    for nm in args.entrants + args.field:  # fail fast on typos
        parse_entrant(nm, args.mc_top, args.mc_sims)
    if len(set(args.entrants)) < len(args.entrants) or set(args.entrants) & set(args.field):
        print("! duplicate entrant names share one row in the results", file=sys.stderr)

    base = B.load_base(args.first, args.last, args.cache_dir)
    cfg = vars(args)
    jobs = [(c, h, cu, cfg) for c, h, cu in base]
    print(f"Tournament: {len(args.entrants)} entrants, {len(jobs)} seasons, starts 1..{args.max_start}, "
          f"{args.jobs} workers ...", flush=True)
    rows = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(process_season, jobs):
                rows += r
    else:
        for j in jobs:
            rows += process_season(j)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 50)
    lb = leaderboard(df)
    print(f"\n=== LEADERBOARD ({df['pool'].nunique()} pools; ROI = net profit per unit staked, "
          f"SE = season bootstrap) ===")
    print(lb.round(3).to_string(index=False))
    print(f"\nsanity: total net across all seats = {df.net.sum():.6f} (pools are zero-sum)")

    h = head_to_head(df, list(lb.entrant))
    print("\n=== HEAD TO HEAD: row finished ahead of column (share of shared pools, ties = 1/2) ===")
    short = {nm: nm.replace("greedy", "g").replace("crowd", "c").replace("_", "") for nm in h.columns}
    print(h.rename(columns=short).round(2).to_string(na_rep="  -"))
    print(f"\nPer-seat detail saved to {args.out}")


if __name__ == "__main__":
    main()
