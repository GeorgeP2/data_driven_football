#!/usr/bin/env python3
"""
Backtest Last-Man-Standing strategies on real EPL history (football-data.co.uk, 2000/01 onward).

What is replayed
  * REAL results and REAL pre-match bookmaker odds for every gameweek.
  * What the strategy is allowed to know at gameweek g:
      - odds for gameweek g only (as in real life, odds only exist a round or so ahead);
      - a Dixon-Coles model fitted on matches BEFORE g (refit every --refit-every gameweeks)
        for gameweeks g+1 ... g+H-1.   No look-ahead.
  * Opponents are simulated bots: each alive opponent picks an unused team with
    probability proportional to P(win)^k (k = --k, higher = crowd piles on favourites).
    Their picks are random, the match results are real.

Two tests
  SOLO  : strategy plays alone from every start gameweek; measures pure survival skill
          (weeks until first loss, capped), no opponent modelling involved.
  POOL  : strategy vs 13 bots with your pool rules (draw = out, no team reuse, everyone
          eliminated -> reset + rebuy, lists reset). Measures net profit per stake.
          'crowd' strategy (plays like the bots) should come out at ~0 ROI: sanity check.

Strategies
  random  : uniformly random unused team
  crowd   : same behaviour as the opponents
  greedy  : best win probability available this week
  plan    : rolling-horizon assignment (maximise product of survival probs), play week 1
  plan_mc : plan shortlist + Monte Carlo pot-equity scoring (slow; use --mc)

Usage
  python -m lms.backtest    # all seasons 2000/01 .. last complete, downloads + caches CSVs
  python -m lms.backtest --first 1415 --last 2425 --jobs 4
  python -m lms.backtest --mc --mc-starts 1    # include the slow Monte Carlo strategy
  python -m lms.backtest --k 2 --horizon 6     # sensitivity checks
  python -m lms.backtest --odds-weight 0.6 0.8 1.0   # sweep the odds/model blend
"""

import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football.football_data import canon, read_fd_csv
from lms import optimiser as L

BASE = "https://football-data.co.uk/mmz4281/{s}/E0.csv"
ODDS_SETS = [
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("GBH", "GBD", "GBA"),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def season_codes(first, last):
    """'0001' ... '2425' (inclusive), as two-digit year pairs."""
    f, la = int(first[:2]), int(last[:2])

    # years 00..99 wrap: treat >=90 as 19xx
    def y(v):
        return v + (1900 if v >= 90 else 2000)

    return [f"{a % 100:02d}{(a + 1) % 100:02d}" for a in range(y(f), y(la) + 1)]


def load_season(code, cache_dir):
    path = os.path.join(cache_dir, f"E0_{code}.csv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        read_fd_csv(BASE.format(s=code)).to_csv(path, index=False)
    df = read_fd_csv(path)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date", kind="stable").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "date": df["Date"],
            "home": df["HomeTeam"].map(canon),
            "away": df["AwayTeam"].map(canon),
            "hg": df["FTHG"].astype(int),
            "ag": df["FTAG"].astype(int),
        }
    )
    o = np.full((len(df), 3), np.nan)
    for h, d, a in ODDS_SETS:
        if all(c in df.columns for c in (h, d, a)):
            v = df[[h, d, a]].apply(pd.to_numeric, errors="coerce").to_numpy()
            ok = np.isnan(o[:, 0]) & ~np.isnan(v).any(axis=1) & (v > 1).all(axis=1)
            o[ok] = v[ok]
    out[["odds_h", "odds_d", "odds_a"]] = o
    out["season"] = code
    return out


def assign_gw(df, gap_days=2, min_matches=5):
    """Gameweek per match (df sorted by date), 0 = dropped.
    Matches are clustered by date (a new cluster starts after a blank day), and a cluster that
    holds a weekend plus a midweek round is split by each team's 1st/2nd/... appearance in it.
    Rounds with fewer than `min_matches` matches (isolated rearranged fixtures) are dropped, as
    an LMS pool would not treat them as a gameweek. Counting matches per team over the whole
    season instead lets every postponement push later rounds out of step."""
    dates = pd.to_datetime(df.date).to_numpy()
    keys, cluster, cnt = [], 0, {}
    for i, (h, a) in enumerate(zip(df.home, df.away, strict=True)):
        if i and (dates[i] - dates[i - 1]) / np.timedelta64(1, "D") >= gap_days:
            cluster, cnt = cluster + 1, {}
        sub = max(cnt.get(h, 0), cnt.get(a, 0)) + 1
        cnt[h] = cnt[a] = sub
        keys.append((cluster, sub))
    sizes = pd.Series(keys).value_counts()
    kept = sorted(k for k in sizes.index if sizes[k] >= min_matches)
    num = {k: g for g, k in enumerate(kept, start=1)}
    return [num.get(k, 0) for k in keys]


# --------------------------------------------------------------------------- #
# Per-season prepared state
# --------------------------------------------------------------------------- #
class Season:
    def __init__(self, code, hist, cur, refit_every, hist_matches, odds_weight=1.0):
        self.code = code
        cur = cur.copy()
        cur["gw"] = assign_gw(cur)
        played = cur  # every match, incl. dropped catch-ups, is still history for the model
        cur = cur[cur.gw > 0].reset_index(drop=True)
        self.cur = cur
        self.teams = sorted(set(cur.home) | set(cur.away))
        self.T = len(self.teams)
        tix = {t: i for i, t in enumerate(self.teams)}
        self.G = int(cur.gw.max())
        h = cur.home.map(tix).to_numpy()
        a = cur.away.map(tix).to_numpy()
        gw = cur.gw.to_numpy()
        M = len(cur)

        # actual outcomes: win[g, t]
        self.win = np.zeros((self.G + 1, self.T + 1), dtype=bool)  # extra col = "no pick"
        hg, ag = cur.hg.to_numpy(), cur.ag.to_numpy()
        self.win[gw[hg > ag], h[hg > ag]] = True
        self.win[gw[ag > hg], a[ag > hg]] = True

        # odds probs per match (Shin de-vig)
        po = np.full((M, 3), np.nan)
        for i, row in enumerate(cur[["odds_h", "odds_d", "odds_a"]].values):
            if not np.isnan(row).any():
                po[i] = L.shin_devig(row)
        self.has_odds = ~np.isnan(po[:, 0])

        # model versions, refit at gw 1, 1+R, ...
        self.refit_gws = list(range(1, self.G + 1, refit_every))
        self.version = np.zeros(self.G + 2, dtype=int)
        for v, rg in enumerate(self.refit_gws):
            self.version[rg:] = v
        self.version[0] = 0
        first_date = cur.groupby("gw").date.min()
        self.model_match = []  # per version: (M,3) model probs
        self.Pmodel = []  # per version: (G+1, T) win probabilities
        for rg in self.refit_gws:
            cutoff = first_date[rg]
            past = pd.concat([hist, played[played.date < cutoff][hist.columns]]).tail(hist_matches)
            dc = L.DixonColes().fit(past[["date", "home", "away", "hg", "ag"]])
            mm = np.array([dc.predict(x, y) for x, y in zip(cur.home, cur.away, strict=True)])
            self.model_match.append(mm)
            self.Pmodel.append(self._to_win_matrix(mm, gw, h, a))

        # what the strategy sees for the *current* gameweek: odds blended with model
        self.now_match = np.zeros((M, 3))
        for i in range(M):
            mm = self.model_match[self.version[gw[i]]][i]
            self.now_match[i] = (
                (odds_weight * po[i] + (1 - odds_weight) * mm) if self.has_odds[i] else mm
            )
            self.now_match[i] /= self.now_match[i].sum()
        self.Pnow = self._to_win_matrix(self.now_match, gw, h, a)
        self.gw, self.h, self.a = gw, h, a

    def _to_win_matrix(self, probs, gw, h, a):
        P = np.zeros((self.G + 1, self.T))
        for i in range(len(gw)):
            P[gw[i], h[i]] = max(P[gw[i], h[i]], probs[i, 0])
            P[gw[i], a[i]] = max(P[gw[i], a[i]], probs[i, 2])
        return P

    def forecast(self, g, H):
        """Win-prob matrix for gameweeks g..g+H-1.

        Week g uses the odds blend, later weeks the model only.
        """
        v = self.version[g]
        rows = [self.Pnow[g]] + [self.Pmodel[v][g + j] for j in range(1, H) if g + j <= self.G]
        return np.vstack(rows)

    def fp_rows(self, g, H):
        v = self.version[g]
        sel = (self.gw >= g) & (self.gw < g + H)
        idx = np.flatnonzero(sel)
        p = np.where((self.gw[idx] == g)[:, None], self.now_match[idx], self.model_match[v][idx])
        return pd.DataFrame(
            {
                "gw": self.gw[idx],
                "home": [self.teams[i] for i in self.h[idx]],
                "away": [self.teams[i] for i in self.a[idx]],
                "p_h": p[:, 0],
                "p_d": p[:, 1],
                "p_a": p[:, 2],
                "src": "bt",
            }
        )


# --------------------------------------------------------------------------- #
# Strategies: f(S, g, used, state) -> team index or None
# --------------------------------------------------------------------------- #
def _avail(S, used):
    m = np.ones(S.T, bool)
    if used:
        m[list(used)] = False
    return m


def crowd_pick(p, used_mask, k, rng):
    w = np.power(np.clip(p, 0, None), k) * used_mask
    if w.sum() <= 0:
        return None
    return int(np.searchsorted(np.cumsum(w / w.sum()), rng.random()).clip(max=len(p) - 1))


def make_crowd(k):
    def f(S, g, used, state):
        return crowd_pick(S.Pnow[g], _avail(S, used), k, state["rng_me"])

    return f


_RAND = np.random.default_rng(7)


def strat_random(S, g, used, state):
    """Uniformly random unused team that has a match this week."""
    ok = np.flatnonzero(_avail(S, used) & (S.Pnow[g] > 0))
    return int(_RAND.choice(ok)) if len(ok) else None


def strat_greedy(S, g, used, state):
    p = S.Pnow[g] * _avail(S, used)
    return int(p.argmax()) if p.max() > 0 else None


def make_plan(H):
    def f(S, g, used, state):
        P = S.forecast(g, H)
        cols = np.flatnonzero(_avail(S, used))
        if len(cols) == 0:
            return None
        W = min(P.shape[0], len(cols))
        cost = -np.log(np.clip(P[:W][:, cols], 1e-6, 1.0))
        r, c = linear_sum_assignment(cost)
        t = int(cols[c[np.argmin(r)]])
        return t if P[0, t] > 0 else strat_greedy(S, g, used, state)

    return f


def make_plan_mc(H, top_n, n_sims, skill_edge=1.0, k=3.0):
    def f(S, g, used, state):
        P = S.forecast(g, H)
        avail = _avail(S, used)
        if avail.sum() == 0:
            return None
        cands = []
        for t in np.flatnonzero(avail & (P[0] > 0)):
            plan, surv = L.plan_after_pick(P, avail, int(t))
            cands.append((int(t), plan, surv))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[2])
        cands = cands[:top_n]
        N = state["N"]
        n_alive = int(state["alive_opp"].sum()) + 1
        opp_used = {
            f"o{j}": [S.teams[t] for t in np.flatnonzero(state["used_opp"][j])]
            for j in np.flatnonzero(state["alive_opp"])
        }
        pool = L.Pool(
            n_players=N,
            n_alive=n_alive,
            pot=state["pot"],
            skill_edge=skill_edge,
            opp_sharpness=k,
            opp_used=opp_used,
        )
        fp = S.fp_rows(g, H)
        gws = list(range(g, min(g + H, S.G + 1)))
        draws = L.sample_draws(
            fp, gws, S.teams, n_alive - 1, n_sims, seed=zlib.crc32(f"{S.code}-{g}".encode())
        )
        best, best_ev = None, -1e9
        for t, plan, _surv in cands:
            plan = plan[: len(gws)]
            st = L.simulate_candidate(fp, gws, S.teams, P, plan, pool, n_sims, draws=draws)
            if st["ev"] > best_ev:
                best, best_ev = t, st["ev"]
        return best

    return f


# --------------------------------------------------------------------------- #
# Simulators
# --------------------------------------------------------------------------- #
def run_solo(S, start, strategy, cap=12):
    used, survived = set(), 0
    for g in range(start, min(S.G, start + cap - 1) + 1):
        if len(used) >= S.T:
            used.clear()
        t = strategy(S, g, used, {})
        if t is None or not S.win[g, t]:
            return survived
        used.add(t)
        survived += 1
    return survived


def run_pool(S, start, strategy, seed, N=14, k=3.0):
    rng = np.random.default_rng(seed)
    state = {
        "rng_me": np.random.default_rng(seed + 10_000),
        "N": N,
        "used_opp": np.zeros((N - 1, S.T), bool),
        "alive_opp": np.ones(N - 1, bool),
        "pot": float(N),
    }
    used_me, me_alive, resets = set(), True, 0
    stake_me, weeks_me, me_streak_open = 1.0, 0, True
    ended, payout = False, 0.0
    for g in range(start, S.G + 1):
        if len(used_me) >= S.T:
            used_me.clear()
        uo = state["used_opp"]
        ex = uo.all(axis=1)
        uo[ex] = False
        pick_me = strategy(S, g, used_me, state) if me_alive else None

        w = np.power(S.Pnow[g], k)[None, :] * (~uo)
        tot = w.sum(axis=1, keepdims=True)
        has = tot[:, 0] > 0
        cum = np.cumsum(np.where(tot > 0, w / np.where(tot > 0, tot, 1), 0), axis=1)
        pick = (rng.random((N - 1, 1)) > cum).sum(axis=1).clip(max=S.T - 1)
        pick = np.where(has, pick, S.T)  # S.T -> no valid pick -> out
        live = state["alive_opp"]
        for j in np.flatnonzero(live & has):
            uo[j, pick[j]] = True
        state["alive_opp"] = live & S.win[g][pick]

        if me_alive:
            ok = pick_me is not None and S.win[g, pick_me]
            if ok:
                used_me.add(pick_me)
                if me_streak_open:
                    weeks_me += 1
            else:
                me_alive = False
                me_streak_open = False
        n_alive = int(state["alive_opp"].sum()) + int(me_alive)
        if n_alive == 0:  # reset + rebuy
            resets += 1
            stake_me += 1.0
            state["pot"] += N
            state["alive_opp"][:] = True
            state["used_opp"][:] = False
            used_me.clear()
            me_alive = True
        elif n_alive == 1:
            payout = state["pot"] if me_alive else 0.0
            ended = True
            break
    if not ended and me_alive:
        payout = state["pot"] / (int(state["alive_opp"].sum()) + 1)
    return {
        "net": payout - stake_me,
        "stake": stake_me,
        "sole_win": payout >= state["pot"] - 1e-9,
        "resets": resets,
        "weeks_first": weeks_me,
    }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def process_season(args):
    code, hist, cur, cfg = args
    S = Season(code, hist, cur, cfg["refit_every"], cfg["hist_matches"], cfg["odds_weight"])
    strategies = {
        "random": strat_random,
        "crowd": make_crowd(cfg["k"]),
        "greedy": strat_greedy,
        "plan": make_plan(cfg["horizon"]),
    }
    if cfg["mc"]:
        strategies["plan_mc"] = make_plan_mc(
            cfg["horizon"], cfg["mc_top"], cfg["mc_sims"], k=cfg["k"]
        )
    solo_rows, pool_rows = [], []
    for name, strat in strategies.items():
        if name == "plan_mc":
            starts = list(range(1, 1 + cfg["mc_starts"]))
        else:
            starts = list(range(1, min(S.G - 4, cfg["max_start"]) + 1))
        if name not in ("crowd", "plan_mc"):  # random, greedy, plan all get solo runs
            for s in starts:
                solo_rows.append(
                    {
                        "season": code,
                        "strategy": name,
                        "start": s,
                        "survived": run_solo(S, s, strat, cap=cfg["cap"]),
                    }
                )
        pool_starts = [1] if name == "plan_mc" else list(range(1, 1 + cfg["pool_starts"]))
        for s in pool_starts:
            for rep in range(cfg["reps"]):
                r = run_pool(
                    S,
                    s,
                    strat,
                    seed=zlib.crc32(f"{code}-{s}-{rep}".encode()),
                    N=cfg["n"],
                    k=cfg["k"],
                )
                pool_rows.append({"season": code, "strategy": name, "start": s, "rep": rep, **r})
    return solo_rows, pool_rows


# --------------------------------------------------------------------------- #
def boot_se(df, num, den, n=1000, seed=0):
    g = df.groupby("season")[[num, den]].sum()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        vals.append(s[num].sum() / s[den].sum())
    return float(np.std(vals))


def load_base(first, last, cache_dir):
    """[(season code, history before it, season matches)] for each complete test season."""
    test = season_codes(first, last)
    pre = season_codes(f"{(int(first[:2]) - 3) % 100:02d}00", f"{(int(first[:2]) - 1) % 100:02d}00")
    codes = pre + test
    frames = {}
    for c in codes:
        try:
            frames[c] = load_season(c, cache_dir)
        except Exception as e:
            print(f"! skip season {c}: {e}", file=sys.stderr)
    base = []
    for c in test:
        if c not in frames:
            continue
        cur = frames[c]
        if len(cur) < 370:
            print(f"! season {c} incomplete ({len(cur)} matches), skipped", file=sys.stderr)
            continue
        prior = [frames[p] for p in codes[: codes.index(c)] if p in frames]
        if not prior:
            print(f"! season {c} has no history, skipped", file=sys.stderr)
            continue
        hist = pd.concat(prior)[["date", "home", "away", "hg", "ag"]].reset_index(drop=True)
        base.append((c, hist, cur))
    return base


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--first", default="0001", help="first season to TEST (default 0001 = 2000/01)")
    ap.add_argument("--last", default="2526", help="last season to test (must be complete)")
    ap.add_argument("--cache-dir", default="data/fd_cache")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument(
        "--odds-weight",
        type=float,
        nargs="+",
        default=[1.0],
        help="weight(s) on odds vs model for the current week; several values = sweep",
    )
    ap.add_argument("--k", type=float, default=3.0, help="opponent favourite-chasing sharpness")
    ap.add_argument("--n", type=int, default=14, help="pool size")
    ap.add_argument("--refit-every", type=int, default=6)
    ap.add_argument("--hist-matches", type=int, default=900)
    ap.add_argument("--max-start", type=int, default=28, help="latest solo start gameweek")
    ap.add_argument("--cap", type=int, default=12, help="solo survival cap in weeks")
    ap.add_argument("--pool-starts", type=int, default=1, help="pool start gameweeks 1..this")
    ap.add_argument("--reps", type=int, default=40, help="opponent-randomness repeats per pool")
    ap.add_argument("--mc", action="store_true")
    ap.add_argument("--mc-top", type=int, default=4)
    ap.add_argument("--mc-sims", type=int, default=800)
    ap.add_argument("--mc-starts", type=int, default=1)
    ap.add_argument("--out", default="outputs/backtest_runs")
    args = ap.parse_args()

    base = load_base(args.first, args.last, args.cache_dir)
    for w in args.odds_weight:
        cfg = {**vars(args), "odds_weight": w}
        jobs = [(c, h, cu, cfg) for c, h, cu in base]
        tag = f"{args.out}_w{w:g}"
        print(
            f"\n##### odds_weight = {w:g}: backtesting {len(jobs)} seasons "
            f"on {args.jobs} workers ...",
            flush=True,
        )
        report(jobs, args, tag)


def report(jobs, args, tag):
    solo, pool = [], []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for sr, pr in ex.map(process_season, jobs):
                solo += sr
                pool += pr
    else:
        for j in jobs:
            sr, pr = process_season(j)
            solo += sr
            pool += pr
    solo, pool = pd.DataFrame(solo), pd.DataFrame(pool)
    os.makedirs(os.path.dirname(tag) or ".", exist_ok=True)
    solo.to_csv(f"{tag}_solo.csv", index=False)
    pool.to_csv(f"{tag}_pool.csv", index=False)

    pd.set_option("display.width", 160)
    print(
        f"\n=== SOLO survival (weeks until first loss, capped at {args.cap}; "
        "all start gameweeks) ==="
    )
    t = solo.groupby("strategy").survived.agg(
        mean="mean",
        p_ge5=lambda x: (x >= 5).mean(),
        p_ge8=lambda x: (x >= 8).mean(),
        p_ge_cap=lambda x: (x >= args.cap).mean(),
        runs="count",
    )
    t["se_mean"] = [
        solo[solo.strategy == s].groupby("season").survived.mean().std()
        / np.sqrt(solo[solo.strategy == s].season.nunique())
        for s in t.index
    ]
    print(t.round(3).to_string())

    print(f"\n=== POOL vs {args.n - 1} bots (k={args.k:.1f}): net profit per unit staked ===")
    rows = []
    for s, d in pool.groupby("strategy"):
        rows.append(
            {
                "strategy": s,
                "ROI": d.net.sum() / d.stake.sum(),
                "ROI_se": boot_se(d, "net", "stake"),
                "P(sole win)": d.sole_win.mean(),
                "resets/pool": d.resets.mean(),
                "weeks_first_out": d.weeks_first.mean(),
                "pools": len(d),
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nPer-run detail saved to {tag}_solo.csv / {tag}_pool.csv")


if __name__ == "__main__":
    main()
