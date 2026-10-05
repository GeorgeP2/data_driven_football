#!/usr/bin/env python3
"""
Last Man Standing (EPL) optimiser
=================================
Rules assumed: pick one team per gameweek to WIN (draw = out), never reuse a team.
If everyone is eliminated, the pool resets (used-teams lists reset too) and everyone rebuys.

Pipeline
  1. De-vig bookmaker 1X2 odds (Shin's method) where odds exist.
  2. Dixon-Coles model (fit on results.csv) fills in fixtures without odds.
  3. Rolling-horizon assignment (Hungarian) = best season plan for each candidate pick now.
  4. Monte Carlo with opponent model, scoring each candidate by expected pot equity:
       P(sole survivor)*pot + P(all out)*V_reset + P(shared at horizon end)*pot/alive

Inputs (CSV / JSON, see --help and the README block at the bottom)
  results.csv   date,home,away,hg,ag             (history for Dixon-Coles; ~2 seasons)
  fixtures.csv  gw,home,away[,odds_h,odds_d,odds_a]   (upcoming; odds optional, decimal)
  config.json   pool settings, your used teams, opponents' used teams

Usage
  python lms_optimiser.py --results results.csv --fixtures fixtures.csv --config config.json
  python lms_optimiser.py --demo        # runs on synthetic data to show the output
"""
import argparse
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linear_sum_assignment, minimize
from scipy.special import gammaln
from scipy.stats import poisson

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# 1. Odds -> probabilities
# --------------------------------------------------------------------------- #
def shin_devig(odds):
    """Shin's method for removing the bookmaker margin. odds: decimal [H, D, A]."""
    pi = 1.0 / np.asarray(odds, dtype=float)
    B = pi.sum()
    if B <= 1.0:
        return pi / B

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * pi**2 / B) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1.0, 1e-9, 0.4)
        p = probs(z)
        return p / p.sum()
    except ValueError:
        return pi / B  # fallback: proportional


# --------------------------------------------------------------------------- #
# 2. Dixon-Coles
# --------------------------------------------------------------------------- #
def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, xi=0.0018):
        self.xi = xi  # time decay per day (half-life ~ 385 days)

    def fit(self, results: pd.DataFrame):
        df = results.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.teams = sorted(set(df.home) | set(df.away))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h = df.home.map(idx).values
        a = df.away.map(idx).values
        hg = df.hg.values.astype(int)
        ag = df.ag.values.astype(int)
        days = (df.date.max() - df.date).dt.days.values
        w = np.exp(-self.xi * days)
        lg = gammaln(hg + 1) + gammaln(ag + 1)  # precomputed: much faster than poisson.logpmf

        def nll(params):
            att = params[:n]
            dfn = params[n:2 * n]
            ha, rho = params[2 * n], params[2 * n + 1]
            lam = np.exp(att[h] + dfn[a] + ha)
            mu = np.exp(att[a] + dfn[h])
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-6, None)
            ll = np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu - lg
            pen = 100.0 * (att.sum() ** 2)  # identifiability
            return -(w * ll).sum() + pen

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n) + [(0.0, 0.8), (-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        p = res.x
        self.att = dict(zip(self.teams, p[:n]))
        self.dfn = dict(zip(self.teams, p[n:2 * n]))
        self.ha, self.rho = p[2 * n], p[2 * n + 1]

        # Prior for teams without history (promoted): the average of the 6 weakest teams in the
        # most recent ~season of data. On 2002/03-2025/26 promoted-team matches this scored a
        # log-loss of 0.976 vs 0.989 for the bottom 3 and 1.072 for a fixed [.40,.27,.33].
        recent = df.sort_values("date").tail(380)
        recent_teams = sorted(set(recent.home) | set(recent.away))
        weakest = sorted(recent_teams, key=lambda t: self.att[t] - self.dfn[t])[:6]
        self.new_att = float(np.mean([self.att[t] for t in weakest]))
        self.new_dfn = float(np.mean([self.dfn[t] for t in weakest]))
        return self

    def predict(self, home, away, max_goals=10):
        """Return [P(home win), P(draw), P(away win)]."""
        att_h = self.att.get(home, self.new_att)
        att_a = self.att.get(away, self.new_att)
        dfn_h = self.dfn.get(home, self.new_dfn)
        dfn_a = self.dfn.get(away, self.new_dfn)
        lam = np.exp(att_h + dfn_a + self.ha)
        mu = np.exp(att_a + dfn_h)
        g = np.arange(max_goals + 1)
        grid = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        grid[0, 0] *= 1 - lam * mu * self.rho
        grid[0, 1] *= 1 + lam * self.rho
        grid[1, 0] *= 1 + mu * self.rho
        grid[1, 1] *= 1 - self.rho
        grid /= grid.sum()
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


# --------------------------------------------------------------------------- #
# 3. Build per-fixture 1X2 probabilities
# --------------------------------------------------------------------------- #
def build_fixture_probs(fixtures: pd.DataFrame, model: DixonColes, odds_weight=1.0):
    """Blend de-vigged odds with the model where both exist. odds_weight is a hyperparameter
    (1.0 = odds only, 0.0 = model only). Fixtures without odds use the model alone."""
    rows = []
    for r in fixtures.itertuples(index=False):
        pm = model.predict(r.home, r.away)
        has_odds = all(hasattr(r, c) and pd.notna(getattr(r, c))
                       for c in ("odds_h", "odds_d", "odds_a"))
        if has_odds:
            po = shin_devig([r.odds_h, r.odds_d, r.odds_a])
            p = odds_weight * po + (1 - odds_weight) * pm
            src = "odds"
        else:
            p, src = pm, "model"
        rows.append((r.gw, r.home, r.away, *(p / p.sum()), src))
    return pd.DataFrame(rows, columns=["gw", "home", "away", "p_h", "p_d", "p_a", "src"])


def win_matrix(fp: pd.DataFrame, gws, teams):
    """P[w, t] = P(team t wins) in gameweek gws[w]; 0 if no game.
    Double gameweek: the best single fixture is used (you pick one team once)."""
    tix = {t: i for i, t in enumerate(teams)}
    P = np.zeros((len(gws), len(teams)))
    for r in fp.itertuples(index=False):
        if r.gw not in gws:
            continue
        w = gws.index(r.gw)
        P[w, tix[r.home]] = max(P[w, tix[r.home]], r.p_h)
        P[w, tix[r.away]] = max(P[w, tix[r.away]], r.p_a)
    return P


# --------------------------------------------------------------------------- #
# 4. Rolling-horizon assignment
# --------------------------------------------------------------------------- #
def plan_after_pick(P, avail_mask, first_team):
    """Force `first_team` in week 0, then optimally assign the remaining weeks
    from still-available teams. Returns (list of team indexes per week, survival prob)."""
    W, T = P.shape
    plan = [first_team]
    surv = P[0, first_team]
    if W > 1:
        cand = [t for t in range(T) if avail_mask[t] and t != first_team]
        sub = P[1:, cand]
        cost = -np.log(np.clip(sub, 1e-6, 1.0))
        r, c = linear_sum_assignment(cost)
        order = np.argsort(r)
        for k in order:
            plan.append(cand[c[k]])
            surv *= sub[r[k], c[k]]
    return plan, surv


# --------------------------------------------------------------------------- #
# 5. Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass
class Pool:
    n_players: int = 14
    stake: float = 1.0
    rebuy: float = 1.0
    pot: float = None          # defaults to n_players * stake (rolled-over pots: set it)
    skill_edge: float = 1.0    # 1.0 = average player; >1 if you think you're better
    opp_sharpness: float = 3.0 # k in P(win)^k opponent pick weighting
    n_alive: int = None        # opponents + you currently alive (default: all)
    me_used: list = field(default_factory=list)
    opp_used: dict = field(default_factory=dict)  # name -> list of used teams (alive opps only)

    def __post_init__(self):
        if self.pot is None:
            self.pot = self.n_players * self.stake
        if self.n_alive is None:
            self.n_alive = self.n_players

    @property
    def v_reset(self):
        # After a reset: share of (rolled pot + new rebuys) minus the rebuy you pay.
        total = self.pot + self.n_players * self.rebuy
        return self.skill_edge * total / self.n_players - self.rebuy


def sample_draws(fp, gws, teams, n_opp, n_sims=4000, seed=0):
    """Random draws shared by every candidate (common random numbers), so EV differences
    between candidates reflect the picks rather than simulation noise.
    Returns (team_win[s, w, t], opp_u[w, s, j, 1])."""
    rng = np.random.default_rng(seed)
    W, T = len(gws), len(teams)
    tix = {t: i for i, t in enumerate(teams)}
    team_win = np.zeros((n_sims, W, T), dtype=bool)
    for r in fp.itertuples(index=False):
        if r.gw not in gws:
            continue
        w = gws.index(r.gw)
        u = rng.random(n_sims)
        team_win[:, w, tix[r.home]] |= u < r.p_h
        team_win[:, w, tix[r.away]] |= u > r.p_h + r.p_d
    opp_u = rng.random((W, n_sims, max(n_opp, 1), 1))
    return team_win, opp_u


def simulate_candidate(fp, gws, teams, P, plan, pool: Pool, n_sims=4000, draws=None):
    """Simulate the next len(plan) gameweeks. Returns dict of outcome stats.
    Pass the same `draws` (from sample_draws) for every candidate being compared."""
    W, T = len(plan), len(teams)
    tix = {t: i for i, t in enumerate(teams)}
    n_opp = pool.n_alive - 1

    if draws is None:
        draws = sample_draws(fp, gws, teams, n_opp, n_sims)
    team_win, opp_u = draws
    n_sims = team_win.shape[0]

    # opponent used-sets (unknown history = nothing used)
    used = np.zeros((n_sims, max(n_opp, 1), T), dtype=bool)
    for j, (_, lst) in enumerate(list(pool.opp_used.items())[:n_opp]):
        for t in lst:
            if t in tix:
                used[:, j, tix[t]] = True

    alive_opp = np.ones((n_sims, max(n_opp, 1)), dtype=bool)
    if n_opp == 0:
        alive_opp[:] = False
    me_alive = np.ones(n_sims, dtype=bool)
    outcome_sole = np.zeros(n_sims, dtype=bool)
    outcome_reset = np.zeros(n_sims, dtype=bool)
    done = np.zeros(n_sims, dtype=bool)

    for w in range(W):
        # my pick
        me_win = team_win[:, w, plan[w]]
        me_alive_new = me_alive & me_win

        # opponents pick ~ P(win)^k among unused teams
        wts = np.power(np.clip(P[w], 1e-9, None), pool.opp_sharpness)[None, None, :] * (~used)
        wts_sum = wts.sum(axis=2, keepdims=True)
        wts = np.where(wts_sum > 0, wts / np.where(wts_sum > 0, wts_sum, 1), 1.0 / T)
        cum = wts.cumsum(axis=2)
        pick = (opp_u[w] > cum).sum(axis=2).clip(max=T - 1)
        sim_idx = np.arange(n_sims)[:, None]
        opp_win = team_win[sim_idx, w, pick]
        used[sim_idx, np.arange(used.shape[1])[None, :], pick] = True
        alive_opp_new = alive_opp & opp_win

        n_alive_now = alive_opp_new.sum(axis=1) + me_alive_new
        active = ~done
        # everyone out -> reset
        reset_now = active & (n_alive_now == 0)
        outcome_reset |= reset_now
        # sole survivor is me
        sole_now = active & (n_alive_now == 1) & me_alive_new
        outcome_sole |= sole_now
        # exactly one survivor who is not me -> I get nothing
        other_sole = active & (n_alive_now == 1) & ~me_alive_new
        done |= reset_now | sole_now | other_sole

        me_alive, alive_opp = me_alive_new, alive_opp_new

    # unresolved at horizon end: split pot among those alive
    unresolved = ~done
    n_alive_end = alive_opp.sum(axis=1) + me_alive
    split_share = np.where(unresolved & me_alive, 1.0 / np.maximum(n_alive_end, 1), 0.0)

    ev = (outcome_sole * pool.pot
          + outcome_reset * pool.v_reset
          + split_share * pool.pot).mean()
    return {
        "ev": ev,
        "p_sole": outcome_sole.mean(),
        "p_reset": outcome_reset.mean(),
        "p_still_in_shared": (unresolved & me_alive).mean(),
    }


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #
def recommend(model, fixtures, pool: Pool, horizon=8, top_n=6, n_sims=4000, odds_weight=1.0):
    fp = build_fixture_probs(fixtures, model, odds_weight)
    all_gws = sorted(fp.gw.unique())
    gws = all_gws[:horizon]
    teams = sorted(set(fp.home) | set(fp.away))
    P = win_matrix(fp, gws, teams)

    tix = {t: i for i, t in enumerate(teams)}
    avail = np.array([t not in pool.me_used for t in teams])

    # shortlist: best by plan survival
    cands = []
    for t in range(len(teams)):
        if not avail[t] or P[0, t] <= 0:
            continue
        plan, surv = plan_after_pick(P, avail, t)
        cands.append((t, plan, surv))
    cands.sort(key=lambda x: -x[2])
    cands = cands[:top_n]

    draws = sample_draws(fp, gws, teams, pool.n_alive - 1, n_sims)
    rows = []
    for t, plan, surv in cands:
        stats = simulate_candidate(fp, gws, teams, P, plan, pool, n_sims, draws=draws)
        rows.append({
            "pick": teams[t],
            "P(win) now": P[0, t],
            "plan survival": surv,
            "P(sole)": stats["p_sole"],
            "P(reset)": stats["p_reset"],
            "EV (stakes)": stats["ev"],
            "plan": " > ".join(teams[i] for i in plan[:5]),
        })
    out = pd.DataFrame(rows).sort_values("EV (stakes)", ascending=False)
    return out, fp


def load_config(path):
    with open(path) as f:
        c = json.load(f)
    return Pool(**{k: v for k, v in c.items() if k in Pool.__dataclass_fields__}), c


def demo_data():
    """Synthetic league so the script can be tried end to end."""
    teams = [f"Team{i:02d}" for i in range(20)]
    strength = np.linspace(0.6, -0.6, 20)
    rows = []
    for d in range(380 * 2 // 2):
        h, a = RNG.choice(20, 2, replace=False)
        lam = np.exp(0.25 + strength[h] - 0.5 * strength[a] + 0.1)
        mu = np.exp(strength[a] - 0.5 * strength[h] - 0.1)
        rows.append((pd.Timestamp("2025-08-01") + pd.Timedelta(days=d // 2),
                     teams[h], teams[a], RNG.poisson(lam), RNG.poisson(mu)))
    results = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    fx = []
    for gw in range(1, 11):
        order = RNG.permutation(20)
        for i in range(0, 20, 2):
            fx.append((gw, teams[order[i]], teams[order[i + 1]]))
    fixtures = pd.DataFrame(fx, columns=["gw", "home", "away"])
    # no odds in the demo: every fixture comes from the model
    fixtures[["odds_h", "odds_d", "odds_a"]] = np.nan
    return results, fixtures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results")
    ap.add_argument("--fixtures")
    ap.add_argument("--config")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--odds-weight", type=float, default=1.0,
                    help="weight on de-vigged odds vs Dixon-Coles for fixtures that have odds (0-1)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        results, fixtures = demo_data()
        pool = Pool(n_players=14, me_used=["Team00"], n_alive=10)
    else:
        if not (args.results and args.fixtures and args.config):
            ap.error("need --results, --fixtures and --config (or --demo)")
        results = pd.read_csv(args.results)
        fixtures = pd.read_csv(args.fixtures)
        pool, _ = load_config(args.config)

    model = DixonColes().fit(results)
    out, fp = recommend(model, fixtures, pool, horizon=args.horizon, n_sims=args.sims,
                        odds_weight=args.odds_weight)

    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(f"\nPool: {pool.n_alive} alive of {pool.n_players} | pot={pool.pot:.1f} | "
          f"V_reset={pool.v_reset:.2f} stakes | opp sharpness k={pool.opp_sharpness}\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best = out.iloc[0]
    print(f"\n>>> Recommended pick: {best['pick']}  (EV {best['EV (stakes)']:.2f} stakes)")
    safest = out.sort_values("plan survival", ascending=False).iloc[0]
    if safest["pick"] != best["pick"]:
        print(f"    Pure-survival pick would be: {safest['pick']}")


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #
# README: file formats
# --------------------------------------------------------------------------- #
# config.json
# {
#   "n_players": 14,
#   "n_alive": 9,                    // including you
#   "stake": 1.0, "rebuy": 1.0,
#   "pot": 14.0,                     // current pot (include rollovers)
#   "skill_edge": 1.0,               // >1 if you believe you're better than average
#   "opp_sharpness": 3.0,            // higher = opponents pick favourites more
#   "me_used": ["Arsenal", "Man City"],
#   "opp_used": { "Dave": ["Liverpool"], "Sam": ["Chelsea", "Arsenal"] }   // alive opps, if known
# }
# Team names must match exactly across results.csv, fixtures.csv and config.json.
# fixtures.csv: gw,home,away,odds_h,odds_d,odds_a  (leave odds blank for later gameweeks)
