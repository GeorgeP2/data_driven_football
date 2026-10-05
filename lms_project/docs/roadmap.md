# Roadmap

*Breaks [further-research.md](further-research.md) into milestones and atomic tasks. 5 October 2026.*

Each milestone has a motivation, a plan and acceptance criteria. Each task is one reviewable change
with its own "done when". Task IDs (`T1.3`) are stable so they can be referenced in commits.

## Order and dependencies

```
M0 Benchmark harness ──┬─> M1 Real opponents + live log ──┐
                       ├─> M2 Data augmentation ──────────┼─> M3 Tune + headroom ──> [gate] ──> M5 Advanced strategies
                       │                                  │
                       └─> M4 Better forecasts (parallel) ┘
M6 Robustness (parallel, any time after M0)
```

- **M0 first**: every later acceptance criterion is stated in its metric, and the tournament
  currently reports ROI, not fair-share multiple or paired differences.
- **M1 starts now** in parallel with M0: the season is under way and pick data that isn't logged
  each week is lost.
- **Gate after M3**: M5 only proceeds if the headroom benchmark shows a meaningful gap.

---

## M0. Benchmark harness

**Motivation.** The adoption rules in further-research.md (paired comparison, season-bootstrap SE,
held-out real seasons, 1.5× fair share) can't be checked with the tournament as it stands. Without
one agreed measurement, every later milestone would argue from a different number.

**Plan.** Add the fair-share metric, a paired comparison against a reference entrant, a held-out
season split and an 18-seat preset to `lms.tournament`. Reproduce the current baseline with it.

**Acceptance criteria.**
- One command produces, for every entrant, its fair-share multiple and its paired difference vs
  `mc_h4_k5` with a season-bootstrap SE.
- Rerunning the office-pool format reproduces ≈ 1.6× for `mc_h4_k5` (within its SE).
- Two runs with the same seed give identical output.
- `make check` passes, with tests for the new metric and the pairing.

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T0.1 | Add `fair_share_multiple` per seat (sole-win indicator ÷ (1 / seats at start)) to the per-seat rows and the leaderboard | Leaderboard shows the column; unit test on a hand-built 3-pool frame |
| T0.2 | Confirm all entrants in a pool share lineups, seeds and result realisations (common random numbers); fix if not | Test: swapping entrant order doesn't change any seat's outcome for a fixed seed |
| T0.3 | Add `--reference ENTRANT`: report each entrant's paired difference in fair-share multiple vs the reference, SE from bootstrapping real seasons | New table printed; unit test that a reference vs itself gives 0 ± 0 |
| T0.4 | Add `--train-seasons` / `--test-seasons` (default: 18 / 6 split, fixed list committed in code) | Running with `--test-seasons` only touches the 6 held-out seasons |
| T0.5 | Add an 18-seat office-pool preset (`--preset pool18`: 17 crowd-like field seats + 1 contestant) | Preset documented in `--help`; smoke test runs it on one season |
| T0.6 | Re-run the baseline with the harness; record numbers in research.md | research.md has a "baseline (harness v1)" table with command and seed |

---

## M1. Real opponents and the live log (items 1 and 10, live part)

**Motivation.** Every result so far uses bots from the same P(win)^k family the MC assumes, so the
headline result is partly circular. The pool's real picks are the only way to break that, and the
same weekly data is the only fully out-of-sample test of the optimiser.

**Plan.** Log every pick and result each week to a git-ignored file. Once enough weeks exist, fit a
choice model, wrap it as a tournament entrant and rerun the benchmark with fitted bots in the
colleague seats.

**Acceptance criteria.**
- Pick log is complete from GW1 to the current week, with a committed `.example` schema and no
  personal data in git.
- Fitted model beats the best single-k crowd bot on held-out weeks by log-likelihood.
- Tournament reports `mc_h4_k5` vs greedy with fitted bots in the field, paired, with SE.
- **Decision recorded:** does `mc_h4_k5` still lead against fitted bots? (yes → result holds for
  our pool; no → M3 retunes against fitted bots before anything else)

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T1.1 | Define the pick-log schema (`week, player_id, team, alive_before, result, optimiser_pick`) as `data/picks.csv` + committed `lms_project/picks.example.csv` (outside `data/`, which is git-ignored); anonymised player IDs | Schema in README; example file committed; real file git-ignored |
| T1.2 | Backfill GW1–current from the shared sheet | Row count = sum of players alive per week; check script passes |
| T1.3 | Weekly logging routine (one command or checklist) that also stores the optimiser's recommendation | Used for one live week end to end |
| T1.4 | Feature builder: per (player, week, team) P(win), home/away, big-club flag, last-week popularity, team lost last week, player's past favourite-rate | Unit test on a 2-week toy log |
| T1.5 | Fit a multinomial logit (conditional logit over available teams); compare to best-k crowd bot by held-out log-likelihood | Fit summary + likelihood comparison saved to `outputs/` |
| T1.6 | Add `fitted` entrant to `lms.tournament` that samples picks from the fitted model | `parse_entrant("fitted", …)` works; smoke test |
| T1.7 | Run the benchmark with fitted bots in the field seats; record the decision | research.md section with table, command, decision |

**Note.** With ~18 players × few weeks the fit will be thin early on. Pool across players first;
add per-player effects only if held-out likelihood improves.

---

## M2. Data augmentation (item 4)

**Motivation.** 24 seasons (912 rounds) is too few to tune more than a couple of parameters or to
train anything. Shuffling rounds and re-drawing results removes luck of the order and gives near
unlimited episodes, which M3 and M5 depend on.

**Plan.** Add round shuffling within a season, look-ahead-free forecasts for shuffled seasons, and
result re-drawing from de-vigged odds (plus a calibration-preserving variant). Measure the noise
reduction it actually buys.

**Acceptance criteria.**
- `--shuffle-rounds N` and `--redraw-results K` work in `lms.tournament`, alone and combined.
- No look-ahead: a test shows forecasts for a shuffled week use no data from rounds later in the
  shuffled order.
- Error bars still bootstrap over real seasons (not over shuffles), checked by test.
- Paired SE for `mc_h4_k5` vs greedy shrinks by a documented factor (target ≈ 3×, i.e. ~10× less
  variance) at a stated compute cost.
- Final-verdict mode (`--test-seasons`, real results, no redraw) is unchanged.

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T2.1 | Function to permute rounds within one season (whole rounds only, never across seasons) | Property test: each team still plays exactly once per round; results unchanged |
| T2.2 | Forecast future weeks in shuffled seasons from their own odds plus noise sized to the real model–odds gap; measure that gap | Gap estimate in research.md; look-ahead test passes |
| T2.3 | `--shuffle-rounds N` flag; seeds per (season, ordering) shared across entrants | Smoke test; CRN test from T0.2 still passes |
| T2.4 | Result re-draw from Shin-de-vigged odds | Over many draws, empirical win rates match the odds (test) |
| T2.5 | Calibration-preserving variant: swap real results within favourite-probability bands | Flag option; test that band-level win rates match real data exactly |
| T2.6 | `--redraw-results K` flag | Smoke test; combined with shuffle |
| T2.7 | Variance study: paired SE vs N and K; choose defaults | Table/plot in `reports/figures/`; defaults set in code |
| T2.8 | Track the 50–75% favourite under-pricing: add the calibration table to research.md as a regenerable output | Script regenerates the table |

---

## M3. Tune the MC for our pool and estimate headroom (items 2 and 3)

**Motivation.** k > 5 is untested, and our pool has 18 seats, not 14. Before building anything
complex we need the ceiling: if an oracle barely beats `mc_h4_k5`, items 5–8 aren't worth it.

**Plan.** Sweep k and H on augmented training seasons with paired comparisons, confirm the winner
on held-out real seasons, then run an oracle MC that knows the true opponent policy and searches
deeper.

**Acceptance criteria.**
- Sweep results for k ∈ {3, 5, 8, 12, 20} × H ∈ {1, 2, 3, 4, 6} at 18 seats, against both crowd
  bots and fitted bots (if M1 is done), as paired differences vs `mc_h4_k5`.
- Any new default beats `mc_h4_k5` on held-out real seasons, clears 1.5× fair share, and stays
  ≥ 1× across opponent k = 1.5–20 and fitted bots (the three adoption rules).
- Headroom figure: oracle minus `mc_h4_k5` in fair-share multiple, with SE.
- **Gate decision recorded:** proceed to M5 only if the headroom gap is meaningfully above noise
  (threshold set before running: e.g. ≥ 0.15× fair share and > 2 SE).

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T3.1 | Sweep script over (k, H) using the harness on training seasons | Results CSV + heatmap in `reports/figures/` |
| T3.2 | Repeat the sweep with fitted bots in the field (after T1.6) | Second heatmap |
| T3.3 | Robustness grid for the top 1–3 candidates: opponent k = 1.5, 3, 5, 8, 20 and fitted | Table: min fair-share multiple across opponents |
| T3.4 | Held-out real-season confirmation of the chosen candidate | Paired difference vs `mc_h4_k5` on test seasons, with SE |
| T3.5 | Oracle entrant: true opponent k, H = 6, more candidates and sims | `oracle_…` entrant parses; smoke test |
| T3.6 | Headroom run and gate decision written up | research.md section with threshold, result, decision |

---

## M4. Better future-week probabilities (item 9)

**Motivation.** The planner relies on Dixon-Coles for weeks without odds, and it missed badly
(Newcastle at Coventry: 73% vs the market's 45%). We don't yet know how much this error costs.

**Plan.** First measure the cost with an upper bound (future weeks forecast from their real odds).
Only if the cost is material, fit ratings to past odds and use season-long markets for early-season
priors.

**Acceptance criteria.**
- Cost of forecast error stated in fair-share multiple: backtest with real future odds vs model.
- If a new forecaster is adopted: lower log-loss / Brier vs the market on held-out weeks than
  Dixon-Coles, and a paired tournament improvement for `mc_h4_k5` that uses it.

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T4.1 | Oracle-forecast mode: planner sees real odds for future weeks | Flag in backtest/tournament; smoke test |
| T4.2 | Cost study: oracle forecasts vs Dixon-Coles, paired | Number in research.md; **stop here if small** |
| T4.3 | Ratings fitted to past de-vigged odds instead of scores | Log-loss vs Dixon-Coles on held-out weeks |
| T4.4 | Early-season priors from title/relegation markets (data source documented) | Improvement on GW1–8 forecasts, esp. promoted teams |
| T4.5 | Swap into the planner and re-run the benchmark | Paired result recorded |

---

## M5. Advanced strategies (items 5–8, gated by M3)

**Motivation.** The best play plausibly differs by game state, the pot is decided with 2–4 players
left, and the MC values the end of its horizon crudely. Each of these is a specific weakness of
`mc_h4_k5`, but each is only worth building if M3 shows headroom.

**Plan.** Build in order of cost: endgame solver, then hybrid (which uses the solver), then learned
leaf value. Deep RL only if the three cheaper methods leave a gap to the oracle.

**Acceptance criteria (per new strategy).** The three adoption rules from further-research.md:
paired win over `mc_h4_k5` with season-bootstrap SE; edge holds on held-out real seasons; ≥ 1.5×
fair share in the office format and ≥ 1× across opponent k = 1.5–20 and fitted bots.

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T5.1 | Endgame solver: expectimax over picks and results for 2–4 alive, 1–3 weeks deep, MC value at leaves | Matches brute force on toy positions (test); runtime per decision stated |
| T5.2 | `endgame_…` entrant: `mc_h4_k5` above threshold, solver below | Benchmark result recorded |
| T5.3 | Hybrid entrant with 2–3 parameters: k(n_alive), H(week), solver threshold | `hybrid_…` parses; parameters chosen on training seasons only |
| T5.4 | Hybrid held-out evaluation | Adoption rules checked and recorded |
| T5.5 | Generate value-function training data from augmented simulated pools (features: week, alive, my vs rivals' remaining strength, pot) | Dataset in `outputs/`; generator tested |
| T5.6 | Train value model (gradient boosting first); report held-out calibration of predicted pot share | Calibration plot in `reports/figures/` |
| T5.7 | Plug value model into MC as leaf value; benchmark | Adoption rules checked |
| T5.8 | **Gate:** remaining gap between best of T5.2/T5.4/T5.7 and the oracle | Decision recorded; RL only if gap is still meaningful |
| T5.9 | RL environment wrapping the tournament (odds-based features only, no team names or round IDs) | Env passes a random-agent smoke test |
| T5.10 | Train against fitted bots + crowd mixture, shaped reward from value model | Training curve; held-out vs training gap reported (memorisation check) |
| T5.11 | Self-play population; final benchmark | Adoption rules checked |

---

## M6. Robustness (item 10, backtest part)

**Motivation.** The current edge over greedy is about 4 standard errors and rests on choices
(closing odds, how catch-up fixtures are dropped, the base rules) that may not match our pool.

**Plan.** Re-run the benchmark under each alternative, one at a time, and add the rule variants our
pool might adopt.

**Acceptance criteria.**
- A robustness table: `mc_h4_k5` vs greedy paired difference under each variant, with SE.
- The headline conclusion either holds under every variant or the exceptions are documented.

**Tasks.**

| ID | Task | Done when |
|---|---|---|
| T6.1 | Paired `mc_h4_k5` vs greedy with matched lineups (via harness) | Updated SE in research.md |
| T6.2 | Closing vs earlier pre-match odds | Row in robustness table |
| T6.3 | Alternative catch-up fixture handling | Row in robustness table |
| T6.4 | Rule variant: draws don't eliminate | Flag + row |
| T6.5 | Rule variant: used list resets after 20 teams | Flag + row |
| T6.6 | Rule variant: buy-backs | Flag + row |
| T6.7 | Double/blank gameweeks | Flag + row |
| T6.8 | Value of picking last (if picks become visible before the deadline) | Number in research.md |
| T6.9 | Correlated picks between rivals (clustered crowd bots) | Row in robustness table |
| T6.10 | End-of-season review of the live log (T1.3): recommendation vs actual vs result | research.md section |
