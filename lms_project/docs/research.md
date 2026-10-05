# Winning a Premier League Last Man Standing pool

*Research notes, 5 October 2026. Code is in the parent `lms_project/` folder; all results are
reproducible (fixed seeds). Next steps: [further-research.md](further-research.md).*

## Summary

- **Surviving longest is not the same as winning the pot.** In head-to-head tournaments the
  strategy that survives the most weeks (always back the biggest available favourite) loses money,
  because it shares its fate with everyone else backing the same team.
- **The best strategy found is Monte Carlo pot-equity scoring** (`mc_h4_k5`). It shortlists strong
  picks, simulates the next 4 gameweeks including the rivals' likely picks, and chooses the pick that
  maximises expected share of the pot. Against a realistic office field it returned
  **+0.73 to +0.98 per stake**, against **−0.11 to −0.18** for always backing the favourite.
- **Planning ahead on its own does not help much.** Even the best simple strategy survives only about
  2.6 weeks on average, so saving teams for weeks you rarely reach costs more than it gains.
- **Being different is worth something only if you are still picking a strong team.** A bot that
  skips the favourite 25% of the time for another strong team beat always backing the favourite.
  Skipping to a random team lost money.
- **Use bookmaker odds for the current week.** Our statistical model is useful for future weeks but
  can be badly wrong early in a season (it rated Newcastle away at Coventry 73%; the market said 45%).

## 1. The problem

Pool rules: each gameweek every player still in picks one Premier League team, which must **win**
(a draw counts as a loss), and **no team can be used twice**. Losers are out. The last player left
wins the pot. **If everyone still in goes out in the same week, the pool resets:** everyone pays
again, the pot rolls over and used-team lists are cleared.

The reset rule changes the objective. Going out at the same time as everyone else is not a loss (you
are back in a bigger pool). The only real loss is going out while someone else survives. So the goal
is not "survive as long as possible" but **maximise expected share of the pot**:

```
EV(pick) = P(sole survivor) × pot  +  P(everyone out together) × V_reset  +  P(still sharing at end) × pot / survivors
```

## 2. Data

| Source | Used for |
|---|---|
| [football-data.co.uk](https://football-data.co.uk) `E0.csv`, 1999/00–2026/27 | results and pre-match 1X2 odds (market average, falling back to individual bookmakers) |
| [FPL API](https://fantasy.premierleague.com/api/fixtures/) | the full 2026/27 fixture list with gameweek numbers |
| OddsPortal | live odds for gameweek 6 (football-data has no odds during an international break) |

Data issues found and fixed:
- **Silently dropped matches.** 1999/00, 2003/04 and 2004/05 lost 45–169 matches each to rows
  with trailing empty fields, so two seasons were skipped entirely. The reader now keeps them.
- **Gameweek reconstruction.** Numbering each team's k-th match as gameweek k sounds robust but
  isn't. One postponed match pushes its teams ahead, every later opponent inherits the shift, and
  seasons came out with **40–49 "gameweeks"**, most with fewer than 10 matches. Matches are now
  grouped by date, and isolated catch-up fixtures (2–7% of matches) are dropped. This gives
  **37–39 gameweeks** of about 10 matches each.

## 3. Methods

**Win probabilities.**
- *Current week:* bookmaker odds with the margin removed (Shin's method). Closing odds already
  account for team news, injuries and rotation.
- *Future weeks:* a time-weighted Dixon-Coles model (Poisson goals with a low-score correction and a
  half-life of about 1 year), refitted every 6 gameweeks on data before that date only (no look-ahead).
- *Promoted teams with no history:* they get the average rating of the 6 weakest teams from last
  season. On 336 early-season matches involving such teams (2002/03–2025/26), log-loss:

  | Prior for an unseen team | Log-loss |
  |---|---|
  | fixed 40% / 27% / 33% (original) | 1.072 |
  | average of 3 weakest teams | 0.989 |
  | **average of 6 weakest teams (chosen)** | **0.976** |
  | average of 9 weakest teams | 0.975 |
  | bookmaker odds (reference) | 0.960 |

**Strategies.** Names below are as used in `tournament.py`.

| Name | Rule |
|---|---|
| `random` | uniformly random unused team |
| `crowd_k<k>` | picks with probability proportional to P(win)^k; a model of typical colleagues (higher k = chases favourites harder) |
| `greedy` | highest P(win) this week (`_w0.5` blends odds 50/50 with the model) |
| `greedy_eps<e>_h<H>` | among teams within e of the best P(win), keep the one with the better upcoming fixture |
| `fav_p<p>[_k<k>]` | backs the favourite, but with probability p picks another team (random, or weighted to strong teams with k) |
| `plan_h<H>` | best assignment of teams to the next H gameweeks (maximises the product of win probabilities, solved with the Hungarian algorithm); play week 1 and re-solve each week |
| `mc_h<H>_k<k>` | shortlist from the planner, then score each candidate by simulated pot EV, assuming rivals pick ~ P(win)^k |

The Monte Carlo scores every candidate on the **same random draws** (common random numbers), so
differences between candidates reflect the picks rather than simulation noise.

**Evaluation.** Replays use real results and real pre-match odds. Each strategy sees only this
week's odds and a model fitted on earlier matches. ROI = net profit per unit staked (including
rebuys after resets). Standard errors are bootstrapped over seasons.

## 4. Results

### 4.1 Survival alone, and against bots (`backtest.py`)

Planning horizon sweep, 2002/03–2025/26 (24 seasons). Solo = weeks until first loss.
Pool = one strategy against 13 `crowd_k3` bots, 960 pools.

| Strategy | Solo: mean weeks | Pool: ROI per stake |
|---|---|---|
| greedy | **2.62** | 1.56 ± 0.56 |
| plan, horizon 2 | 2.53 | 1.90 ± 0.61 |
| plan, horizon 3 | 2.44 | 1.18 ± 0.50 |
| plan, horizon 4 | 2.35 | 1.68 ± 0.62 |
| plan, horizon 5 | 2.34 | 1.39 ± 0.45 |
| plan_mc, horizon 4 | – | **2.00 ± 0.50** |
| crowd (sanity check, should be ≈ 0) | – | −0.08 ± 0.16 |

Paired differences (same bots, same seeds): plan_mc − greedy = **+0.44 ± 0.51**; plan − greedy =
+0.13 ± 0.40. Against bots alone, nothing is distinguishable from greedy. Planning with model-only
probabilities (odds weight 0) gave the same pattern, so the gap is not caused by mixing odds and
model.

**Why planning doesn't help:** with greedy, only about 4% of runs reach week 8, so a planner that
maximises "survive all 8 weeks" is optimising an event that almost never happens. It pays for it by
taking a lower win probability now.

### 4.2 Tournament: strategies play each other (`tournament.py`)

Each season × start gameweek (1–30) is one pool, 2002/03–2025/26.

**Format A: one seat for each of 14 configs (720 pools).**

| Entrant | ROI per stake | P(sole win) | Weeks survived |
|---|---|---|---|
| mc_h4_k5 | **+1.25 ± 0.41** | 14.4% | 3.9 |
| crowd_k3 | +1.19 ± 0.31 | 14.7% | 2.5 |
| crowd_k1.5 | +0.70 ± 0.24 | 10.3% | 1.9 |
| greedy_w0.5 | +0.26 ± 0.31 | 8.9% | 4.5 |
| random | +0.07 ± 0.18 | 6.5% | 1.0 |
| mc_h2_k3 | +0.03 ± 0.24 | 7.8% | 4.6 |
| plan_h8 | −0.14 ± 0.20 | 6.5% | 4.0 |
| plan_h4 | −0.49 ± 0.22 | 2.8% | 4.4 |
| plan_h2 | −0.63 ± 0.10 | 1.4% | 4.7 |
| greedy | **−0.77 ± 0.05** | 1.0% | **4.8** |

Greedy survives the longest and loses the most. Most of this field are near-copies that back the
favourite: they survive together and go out together, so none of them is ever the sole survivor.
The pot goes to whoever picked differently. Format A measures how much you differ from the field
rather than skill, so Format B is the main result.

**Format B: a realistic office pool.** Each pool has 10 colleague-like seats (3 × `crowd_k1.5`,
7 × `crowd_k3`) plus 4 contestants drawn at random from 12 configs; 7,200 pools. (Abridged:
the full table is printed by the command in section 9.)

| Entrant | ROI per stake | P(sole win) |
|---|---|---|
| **mc_h4_k5** | **+0.73 ± 0.19** | 11.6% |
| mc_h2_k3 | +0.54 ± 0.24 | 9.8% |
| plan_h8 | +0.36 ± 0.18 | 8.9% |
| greedy_w0.5 | +0.31 ± 0.21 | 8.4% |
| mc_h4_k3 | +0.27 ± 0.13 | 8.1% |
| plan_h2 / plan_h4 | ≈ −0.05 | 5.5% |
| greedy | −0.18 ± 0.12 | 6.3% |
| random | −0.73 ± 0.05 | 1.6% |

`mc_h4_k5` leads greedy by about 4 standard errors. In the head-to-head table the strategies that
use odds are all close to 50/50 at outlasting each other. The money goes to the one that avoids
sharing a fate with everyone else, not the one that survives longest.

**Assumed opponent sharpness works as a "how different should I be" setting.** The field's real
sharpness was 1.5–3, yet the MC that assumed k = 5 did best. Assuming rivals chase favourites harder
makes it differ from them more readily.

### 4.3 Does skipping the favourite at random help? (Format B, 7,200 pools)

| Entrant | ROI per stake |
|---|---|
| mc_h4_k5 | **+0.98 ± 0.22** |
| plan_h8 | +0.53 ± 0.20 |
| fav_p0.25_k3 (skip 25%, to a strong team) | **+0.24 ± 0.11** |
| fav_p0.5_k3 | +0.09 ± 0.08 |
| fav_p0.1_k3 | −0.02 ± 0.09 |
| greedy (never skip) | −0.11 ± 0.12 |
| fav_p0.25 (skip 25%, to a random team) | −0.30 ± 0.06 |
| fav_p0.5 | −0.36 ± 0.07 |

Skipping helps only when the replacement is a strong team. Skipping at random is still far behind
the MC strategy, which skips **when** it is worth it: when the favourite is heavily backed and a
nearly-as-strong alternative exists.

## 5. Recommended strategy

Use `lms_optimiser.py` each week with `--horizon 4` and `"opp_sharpness": 5` in `config.json`.
Enter every rival's used teams and refresh the odds close to the deadline.

Rules of thumb without the tool:
1. Back a team with roughly a 65% or better chance to win.
2. If most of the pool will be on the same favourite and a nearly-as-strong team is available, take
   that team instead, especially while many players are still in.
3. Save teams that have an easy home fixture coming up.

## 6. Live application: our 2026/27 pool, PL gameweek 6 (10–12 Oct)

Pool state from the shared sheet after its first round (PL gameweek 5): 18 entrants (17 paid),
11 still in. Of the 10 who picked and survived, 8 used Man City, 1 Newcastle and 1 Everton. The 7
who went out had picked Nott'm Forest (5) or Leeds (2). I assumed the unpaid entrant who hasn't
picked yet is still in, and treated the pot as 17 stakes.

Odds as of 5 October (OddsPortal average), recommended setup:

| Pick | Win probability (from odds) | EV (stakes) |
|---|---|---|
| Chelsea v Bournemouth | 56% | 1.597 |
| Man Utd v Spurs | 57% | 1.586 |
| **Arsenal v Leeds** | **70%** | 1.578 |
| Newcastle at Coventry | 45% | 1.497 |

The top three are within noise of each other, and a 2-week horizon or less-sharp opponents
reorders them. **Recommendation: Arsenal.** It is never more than about 1% behind on EV, it is
clearly best when rivals are assumed to chase favourites less hard, and it gives the best chance of
surviving the week. The other picks stay close only because the 7 other City players can't use City
again and are likely to pile onto Arsenal.

## 7. Limitations

- **The simulated colleagues are model bots** of the same family the MC assumes. Recording the
  pool's real picks would allow testing against how people actually behave.
- The tests used 14-seat pools; ours has 18. A bigger pool usually rewards differentiation more.
- The Dixon-Coles model is weak early in a season and for newly promoted teams. It matters only
  for future weeks, but it drives the planner's choices.
- Sharpness values above 5, and horizons other than 2 and 4 for the MC, have not been tested.
- In the tournament every player sees the others' used teams. In the live tool, a rival whose used
  teams aren't entered in `config.json` is treated as having all teams available.

## 8. Further research

The detailed plan is in [further-research.md](further-research.md). In priority order:

1. **Opponents fitted to the pool's real picks**, to replace the model bots (the main limitation above).
2. **Tune the MC for our pool:** sharpness k up to 20, horizons 1–6, 18 seats.
3. **Headroom benchmark:** an "oracle" MC that knows the opponents' exact behaviour, to show how
   much any better method could gain.
4. **Data augmentation:** shuffle gameweek order within seasons (24 × 38! ≈ 10⁴⁶ possible seasons,
   though all built from the same 912 real rounds) and re-draw results from the odds. The odds are
   well calibrated across 9,120 matches, so re-drawn results are realistic.
5. **A hybrid that changes strategy as the game goes on,** with an **exact endgame solver** for
   2–4 players.
6. **A learned value function** to replace the MC's crude valuation at the end of its horizon.
7. **Deep RL with self-play,** only if steps 3–6 show the gains are there. It should see odds-based
   features only (no team names), and be validated on held-out seasons.

## 9. Reproducing

Run from the `lms_project/` folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# football-data CSVs are cached in fd_cache/ (download with curl if Python's SSL certificates are missing)
.venv/bin/python backtest.py --first 0203 --last 2526 --horizon 4 --mc             # section 4.1
.venv/bin/python tournament.py                                                      # format A
.venv/bin/python tournament.py --entrants random greedy greedy_w0.5 greedy_eps0.03_h4 greedy_eps0.08_h4 \
    plan_h2 plan_h4 plan_h8 mc_h2_k3 mc_h4_k1.5 mc_h4_k3 mc_h4_k5 \
    --field crowd_k1.5 crowd_k1.5 crowd_k1.5 crowd_k3 crowd_k3 crowd_k3 crowd_k3 crowd_k3 crowd_k3 crowd_k3 \
    --pool-size 4 --lineups 10                                                      # format B
.venv/bin/python lms_optimiser.py --results results.csv --fixtures fixtures.csv --config config.json --horizon 4
```

| File | Purpose |
|---|---|
| `fetch_data.py` | results, fixtures and odds from football-data (+ a schedule CSV) |
| `lms_optimiser.py` | weekly pick recommender |
| `backtest.py` | solo and pool-versus-bots replay |
| `tournament.py` | strategies against each other |
| `fd_common.py` | team names and CSV reading |
