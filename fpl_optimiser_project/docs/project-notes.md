# FPL optimiser — project notes

A running record of what this is, what has been built, what the numbers actually say, and — the
part most worth reading — the ideas that *failed measurement* so nobody spends a week rediscovering
them. For how to run the tools, see the project `README.md`; this is the engineering log behind them.

---

## What it is

A Fantasy Premier League system in three layers:

1. **A squad optimiser** (`optimise.py`, `model.py`) — a 0-1 integer program solved with OR-Tools
   CP-SAT. Given expected points per player, it picks the optimal legal 15 (budget, 2/5/5/3, ≤3
   per club, valid XI, captain). This half is *solved*: exact and fast.
2. **A forecasting model** (`backtest/points_model.py`) — expected points per player per gameweek,
   built component by component from FPL's scoring rules. This is where the edge is, and where
   almost all the difficulty lives.
3. **A season simulator** (`backtest/`) — plays a completed season under the real transfer rules
   (one free transfer a week bankable to five, −4 per hit, sell-on fee, auto-subs, chips), so any
   change can be scored against what actually happened.

Plus a live **weekly advisor** (`cli_advise.py`) that runs the same code against the live API, and
a **forecast evaluator** (`cli_evaluate.py`) — see "The two things that made this trustworthy".

---

## The two things that made this trustworthy

Everything good in this project traces back to two pieces of measurement infrastructure. Build
these first on any similar problem.

### 1. The noise band (`cli_backtest`, jittered seeds)

A season total is **one** observation, and the system has ~10 hyperparameters. Fitting 10 knobs to
1 number is not model selection, it is curve-fitting. So we measured the noise: run the *same*
model over the *same* season with a ±1% jitter that should change nothing.

> **Result: standard deviation ≈ 30 points.** Any change worth less than ~60 points cannot be
> distinguished from luck.

This one number invalidated an earlier "+20 vs a real manager" claim (it was noise) and became the
bar every subsequent idea had to clear. Most did not.

### 2. Forecast evaluation (`backtest/evaluate.py`, `cli_evaluate.py`)

A season total is one data point; the forecast underneath makes **~30,000** predictions with known
answers. Scoring the forecast directly (rank correlation vs actual points, minutes-model
calibration, head-to-head vs FPL's own `ep_next`) gives three orders of magnitude more signal, runs
in seconds, and cannot be moved by a lucky captain. This is what let us iterate on the model
honestly.

---

## What is built and measured

| Component | File | Verdict |
|---|---|---|
| CP-SAT squad optimiser | `optimise.py`, `model.py` | Exact, fast, done |
| Multi-gameweek transfer planner | `backtest/planner.py` | **+105 pts** (biggest lever) |
| Expected-points model | `backtest/points_model.py` | Beats FPL given equal info (see below) |
| Minutes model + calibration | `backtest/points_model.py` | The bottleneck; calibrated |
| Team strength from xG | `backtest/strength.py` | Small MAE gain, neutral on rank |
| Suspensions | `backtest/suspension.py` | **Works** — a fact, not a forecast |
| Chip timing | `backtest/chips.py` | +12 pts, but overfit and fragile; wildcard branch is dead in the live path — see `wildcard-timing.md` |
| Season-aware rules | `rules.py` | Correctness fix (see traps) |
| Fixture calendar (European) | `backtest/calendar.py` | Built; signal ≈ 0 (see failures) |
| Live advisor | `cli_advise.py`, `live.py` | Works end-to-end |

### The headline result, stated honestly

On 2025-26 the full system scores ~2,125 against a real manager's 2,106. **That is not a win** —
it is inside the noise band. What the season simulation *reliably* shows is where points come from:
the planning horizon is worth ~105, the forecast ~50 over naive baselines, chips ~12. Perfect
foresight (the ceiling) is ~4,675, more than twice as far away again.

### The most important single finding

Our forecast scores **0.707** rank correlation vs FPL's `ep_next` at **0.752** — we appear to lose.
But given equal information about who is fit:

```
  ours (blind to injuries)          0.707
  FPL ep_next (sees injury flags)   0.752
  ours + availability oracle        0.868
```

**Our scoring model is better than FPL's.** The entire gap is that `ep_next` is computed with access
to injury flags, which are not retained historically so the backtest cannot see them. **In a live
season we get them** — so the backtest structurally *understates* the live system, and the biggest
lever is availability information, not the scoring model.

---

## Traps that would have shipped silently (all now fixed)

These are the bugs that fail *flatteringly* — no error, just a better-looking number. Each has a
regression test.

- **Availability lookahead.** "Skip players with 0 minutes" is reading the answer. Availability
  must be predicted from lagged minutes only.
- **Double/blank gameweeks.** Players have one row *per fixture*, so they double in DGWs and vanish
  in blanks. Must aggregate; a missing player scored zero, not null.
- **The sell-on fee.** You keep only half your profit on a sale. Ignore it and the simulated
  manager gets invisibly richer than a real one.
- **Phantom defensive-contribution points.** That scoring rule did not exist before 2025-26, but
  the positional prior would award it in older backtests. Fixed via `rules_for_season`, which
  **raises** on an unknown season rather than defaulting to today's rules.
- **Wrong-season transfer cap.** `TransferRules` read the *live* cap (5) and applied it to all
  seasons; before 2024-25 it was 2. Same fix.
- **The yellow-card cutoff.** A 5th yellow only bans if it arrives inside the club's first 19
  matches (100% within, 10% past). Implemented from memory without the cutoff, it would have
  benched 81 players a season who then played — worse than no data.
- **Chip evaluation after transfers.** The free hit measured the squad *after* the planner had
  already dodged the blank, so it never fired. Must measure the squad you *hold*.
- **The live horizon was one gameweek, always.** `load_live_history` built records for the
  completed gameweeks plus *one* upcoming one, so `[gw for gw in history.gameweeks if gw >=
  gameweek][:horizon]` could only ever return a single element. **`--horizon` was inert live**:
  the five-gameweek planner — the biggest lever in the table above at +105 points — ran as a
  one-week planner against the API, with no error and no warning. It is why live runs churned
  transfers to chase one good fixture, the exact failure this project attributes to a one-week
  planner. Fixed by building `horizon` future gameweeks *and* giving `view_at` a `known_through`
  cutoff: without the second half, the unplayed placeholders read as completed matches in which
  every player logged zero minutes, which is the cold start below arriving from the other
  direction. The backtest could not catch this either — it replays a finished season, so at GW5
  it genuinely holds GW4, and `known_through` defaults to exactly that.

- **The live advisor opened every season blind.** `forecast_minutes` falls back on
  `previous_season_minutes` when a player has no completed gameweeks — but `live.py` built its
  `SeasonHistory` without passing `previous_season` at all, while `cli_backtest.py` passed it.
  So at GW1 every one of 555 players had zero prior minutes, forecast a **0.00** chance of
  starting, and the advisor returned an essentially arbitrary squad. It recovered over the first
  few gameweeks as real minutes accumulated — which is exactly what "it takes a few weeks to
  start working" felt like from the outside. **The backtest could never catch this**, because
  the backtest is the one path that was wired correctly: same class of blind spot as the injury
  flags, pointing the other way (flags make the backtest *understate* the live system; this made
  it *overstate* the opening weeks).

---

- **The carry-over kept last season's team ids.** `load_previous_season` re-keyed *players* by
  name and left `team_id` as the archive's — but FPL reissues team ids alphabetically each
  summer, so every club below a promoted side inherits its neighbour's number. The strength
  prior is keyed by team id, so in 2026-27 Ipswich carried Liverpool's record, Man City
  Newcastle's, Sunderland Wolves' and Bournemouth Burnley's. No error; the live advisor simply
  rated every fixture against the wrong defence and spent GW2-4 stacking Brentford at
  Bournemouth and Arsenal at Sunderland, with Haaland at 3.9 xP. Diagnosed by regressing actual
  on forecast by position (calibration was fine, both seasons) and then rebuilding the live
  GW4 forecast from the backtest's inputs. The backtest had the same scramble, and a worse one —
  it passed the previous season with no player re-key at all, and of 784 element ids present in
  both 2024-25 and 2025-26 exactly one belonged to the same player. Both paths now go through
  `backtest/carryover.py`. The whole-pool rank correlation cannot see the fix (0.7485 → 0.7487):
  that metric is dominated by who plays at all. Treat it as a correctness fix, not a tuned one.
- **The live history cache outlived its gameweek.** `_fetch_histories` cached per season and
  refetched only on `--refresh`, so a file written on 1 September served every run for the
  next fortnight and GW4 was planned having never seen GW3. Now keyed by the last completed
  gameweek as well, so a newly finished gameweek is a cache miss.

## The GW1 cliff, and why there is no crossover

Measured gameweek by gameweek on the rows FPL also forecast, over two independently cold-started
seasons (2025-26 from 2024-25, 2024-25 from 2023-24):

```
                  ours   ep_next
  GW1 (2025-26)  0.009     0.468     n=690
  GW1 (2024-25)  0.027     0.443     n=616
  GW2+            ~0.73     ~0.79    both seasons
```

**At GW1 the model has no signal at all.** Not a weak forecast — a rank correlation of 0.009 is a
coin. With no current-season minutes, every per-90 rate is shrunk to its positional mean and the
pool collapses into near-ties. One round of real football restores it to ~0.73.

Two things fall out of this, and the second was a surprise:

1. **The advisor now falls back on `ep_next` for gameweeks before `MODEL_FROM_GAMEWEEK` (2).**
   This is the whole fix for "it takes a few weeks to start working".
2. **There is no crossover.** Scored across both full seasons, the model is ahead of `ep_next` in
   only ~5 of ~30 usable gameweeks, always narrowly. The intuition that it overtakes once it has
   data is **false on this metric** — so the handover is at GW2 and no later, because a later one
   would be fitting noise.

The standing caveat still applies and cuts the other way: this comparison is blind to injury
flags, which the live advisor *does* get and which the availability-oracle experiment valued at
about +0.16 — far more than the ~0.04 the model trails by from GW2. So the honest reading is
**GW1 is settled (use `ep_next`), GW2+ is genuinely too close to call from backtest evidence
alone.** Do not read the ~0.04 as "FPL is better"; read it as "not separable without the flags".

## The squad this produced, before the fallback existed

Worth recording because it looked plausible: with the carry-over wired but no `ep_next` fallback,
the GW1 optimiser returned five defenders (three of them Brentford), a **goalkeeper as captain**,
and £8.5m unspent, with Haaland ranked 18th behind five defenders. The tell was that the model
rated defenders *most confidently* (top 8 by xP were all GKP/DEF) exactly where the evaluation
says it ranks them *worst* (DEF 0.515 vs FWD 0.611 at cold start). Confidence and accuracy
pointing opposite ways is the signature of a level miscalibration, not an edge.

## Ideas that failed measurement — do not re-attempt without new data

This is the most valuable section. Every one of these was well-motivated and sounded obviously
good. The data said otherwise. The pattern: **clever modelling was worth ~nothing; the only things
that paid were an arithmetic certainty (suspensions) and searching further ahead (the horizon).**

- **A risk margin on transfer hits.** The live advisor proposes a four-transfer −12 every week,
  so the planner's hit penalty was swept (4, 6, 8, 12; the season still charged the real 4)
  over both seasons with the fixed carry-over and the live configuration (horizon 5, discount
  0.85). Net points: 2025-26 — 2197 / 2238 / 2171 / 2178, taking 3 / 1 / 1 / 1 hits; 2024-25 —
  2230 / 2211 / 2215 / 2132, taking 8 / 3 / 0 / 0. The seasons disagree in sign at 6 and both
  fall away by 12, all inside the ±30 noise band. The reason it cannot pay: **the planner is not
  hit-hungry on its own squads** — 3 and 8 hits a season, 12 and 32 points, is the entire prize.
  The weekly −12 live is an artefact of evaluating a *human* squad carrying zero-xP bench players
  and an injured starter, which the planner's own squads never hold; the fix for that is to not
  hold dead slots (or pass the real `--free-transfers`), not a margin. Default left at 4.
- **Price-rise modelling.** Largest price move all of 2025-26 was £1.5m; mean spread £0.19m; four
  players rose ≥£1.0m. A rounding error against a £100m budget.
- **Fixture congestion / European rest.** Built a full multi-competition calendar
  (`backtest/calendar.py`) — correct, cached, tested. As a club-wide discount it moved rank
  correlation by **+0.0001**. Reason: a club-wide discount preserves the *within-club* ordering,
  which is most of what the optimiser reads. Rotation is a *per-player* decision, and the fixture
  list cannot say *which* player is rested.
- **Per-slot bench weights.** Theory said the first substitute is worth 10× the third, so weight
  them differently. Backtest: the shaped bench *lost* up to 107 points. The theory missed that a
  bench player is not just blank-insurance, they are **next week's starter** — the planner rotates
  them in without a transfer, and a flat weight quietly pays for that depth. Default reverted to
  flat 0.25; the mechanism is kept but unused.
- **BPS bonus curve (first attempt).** Applied the within-match S-curve to a player's *average*
  BPS. Jensen's inequality: the average is not the realisation, so a convex curve systematically
  under-predicts. The curve that works is lagged-rate-to-outcome, which is far flatter.
- **A start-probability floor.** A 5% floor is *better calibrated* (fringe players do start ~1 game
  in 20) and improved the Brier score — but *hurt* the points forecast, because it hands ~15,000
  non-players a small noisy forecast where zero was right. Calibration is a means, not the goal.

---

## What actually worked

- **Suspensions.** Deterministic within the yellow-card cutoff (96% precision); red cards
  discounted not excluded (85%, appealable). Removes ~56 confident "ghost" starters a season. Needs
  no backtest to justify — a banned player scores zero.
- **The planning horizon.** 1→3 weeks +73, 3→5 weeks +26. The single biggest lever, and an
  engineering win (bigger solve) rather than a modelling one.
- **Minutes calibration.** The model claimed 98% start probability when 86% started. Capping at the
  measured ceiling (0.88) improved MAE, Brier *and* rank correlation.

---

## Where to go next

Ranked by expected value, given everything above.

1. **Player-level European match logs (FBref/Understat).** The single highest-value lever. The
   fixture calendar failed because it is club-level; knowing *Bruno Fernandes* played 90 minutes on
   Thursday — not merely that United did — is what breaks the within-club symmetry and predicts
   rotation. Harder scrape than a CSV, but it is the thing the data is missing. The calendar is
   already the scaffolding for it.

2. **Availability is the whole game — chase it, carefully.** The oracle experiment proved perfect
   availability is worth +0.16 rank correlation, ~100× any scoring-model tweak. Predicted-lineup
   feeds run ~85-90%, just inside the zone where they help. **Hard constraint: below ~82% accuracy
   an external source makes the system *worse* than ignorance** (a false "he's out" removes a good
   player entirely). Validate any feed against a held-out season *before* wiring it in.

3. **Chips as exact search, not thresholds.** The current chip heuristics are the most overfit part
   of the codebase (the first version *lost* 33 points). The decision space is tiny — 8 chips, 38
   gameweeks, one active at a time — so this is exact dynamic programming or plain enumeration.
   **Not** MCTS: we are evaluation-limited, not search-limited, and the squad solve is already
   exact. Deeper search over a weak forecast is deeper garbage.

4. **Finish the multi-season, held-out validation (task #34, still open).** Tune knobs on
   2022-23…2024-25, report on 2025-26 untouched. This is the only version of "it beats a real
   manager" worth stating. `rules_for_season` already makes it correct; it is ~40 min compute per
   season and just needs running.

### Explicitly not worth doing

- **MCTS / deeper tree search.** Depth is exhausted: 5→8 weeks extrapolates to inside the noise
  band, because the planner discounts future weeks (forecast decays) and the squad solve is already
  optimal at each step. We are evaluation-limited. (Confirm cheaply by running horizon 8 once; if it
  beats 5 by >30 points, revisit.)
- **More scoring-model components.** xG, BPS, DC are in and collectively near their ceiling. The gap
  to perfect is *availability*, not scoring.

---

## The one-line lesson

Build the ruler before the thing you measure with it. The noise band and the forecast evaluator
were worth more than every modelling idea combined — because they are what turned "this feels
better" into "this is +71 or −107", and the difference between those two was, more often than not,
the opposite of what the theory predicted.
