# Forecast log

Weekly record of what the live advisor predicted, so the forecast can be calibrated across the
season instead of judged on whether a captain hauled.

## Each week, before the deadline

```
PYTHONPATH=src python -m fpl_optimiser.cli_advise --team-id <id> --log-forecast outputs/forecast_log
```

Writes `pool_gw<NN>_<season>.json` — expected points for **every player** at **every horizon
depth**, ~3,300 predictions per run. Logging the whole pool rather than the chosen XI is the
point: 11 rows a week would take most of a season to say anything, and the XI is a biased
sample (it is exactly the players the model rates highest).

It also writes `records/plan_gw<NN>_<season>.json` — the planned transfers for every week of
the horizon. Only the first week is executed, but a move made as the first step of a chain
(selling cheap now to fund a bigger buy later) needs the rest of the plan to explain it. And it
saves the `bootstrap-static` the run saw to `outputs/snapshots/`, because FPL overwrites injury
flags in place and this is the only way a history of them comes to exist.

Optionally also hand-write `records/gw<NN>_<season>.json` — the *decision* record (XI, captain,
transfers, counterfactuals). That scores the advice, not the forecast, and is the file
`score.py` reads.

Then, from the repo root, `make sync` to archive the pool log and snapshot to S3, and commit
`records/`. Pool logs and snapshots cannot be regenerated once the deadline passes.

## Any time after a gameweek finishes

```
PYTHONPATH=src python -m fpl_optimiser.forecast.calibrate   # the forecast, all weeks pooled
PYTHONPATH=src python -m fpl_optimiser.forecast.score 5     # one week's decision vs actuals
```

## What the numbers mean

- **actual/xP** — 1.00 is unbiased. Above 1 the model under-forecasts.
- **rank correlation** — does it order players correctly. This is what the transfer planner
  actually consumes; calibration barely matters to it, ordering is everything.
- **head-to-head vs `ep_next`** — FPL's own forecast, free, and the bar to clear. Restricted to
  horizon depth 0 because FPL only publishes a next-gameweek number.
- **by horizon depth** — the planner trusts the week+4 forecast as much as week+0. Whether that
  is earned is a question nothing else in the project measures.

## Two traps in reading this

1. **`actual/xP` is confounded by how high-scoring the week was.** GW4 averaged 69 against a
   season running average around 63, so some of its 1.22 is a good week, not a biased model.
   Only the trend across many weeks separates the two.
2. **A week is ~650 *correlated* predictions, not 650 independent ones** — one wet Saturday
   moves every row. `project-notes.md` puts the season noise band at ±30 points; treat a single
   week's ratio as one observation.

## First reading (GW4 2026-27, out-of-sample, known through GW3)

| | |
|---|---|
| overall actual/xP | 1.22 |
| rank correlation | 0.7610 |
| vs FPL `ep_next` | 0.8386 — **ep_next ahead** |
| DEF / GKP / FWD / MID actual/xP | 1.38 / 1.46 / 1.25 / 1.07 |

`ep_next` out-ranking the model at GW4 is consistent with what `live.py:278` already says (the
gap closes but does not necessarily invert by GW2), but it is the first time it has been
measured on the live path rather than the backtest. If it holds over several weeks it is an
argument for raising `--model-from`, not a bug. One week is not several weeks.

## Second reading (GW5 2026-27, out-of-sample, known through GW4)

Scored 2026-09-21 by hand from `event/5/live/` while FPL still had the week flagged
`finished: false` (all ten fixtures played, bonus provisional). `calibrate.py` will pick GW5 up
on its own once the flag flips; the numbers below may move by a bonus point or two.

| | GW4 | GW5 |
|---|---|---|
| overall actual/xP | 1.22 | 1.26 |
| rank correlation | 0.7610 | 0.7556 |
| vs FPL `ep_next` | 0.8386 — ep_next ahead | 0.7466 — **model ahead** |
| DEF / GKP / FWD / MID actual/xP | 1.38 / 1.46 / 1.25 / 1.07 | 1.43 / 1.43 / 1.08 / 1.17 |

Two weeks running at 1.22–1.26 with DEF and GKP near 1.4 both times, and GW5 was a *low*-scoring
week (average 44 against 50/81/51/69 before it), so the good-week confounder from trap 1 does
not explain it. Still two observations. The head-to-head flipped: one week each.

Restricted to players who got minutes (n=301) the rank correlation collapses to 0.37 for the
model and 0.38 for `ep_next` — the whole-pool number is mostly who-plays, as `project-notes.md`
warns. The decision record for the week is in `gw05_2026-27.json`; `score.py 5` reproduces the
real 27 exactly.

## Files

| File | What it is |
|---|---|
| `outputs/forecast_log/pool_gw<NN>_<season>.json` | Forecast for every player, every horizon week. Generated; archived to S3. |
| `outputs/snapshots/bootstrap_gw<NN>_<season>_<utc>.json.gz` | The API payload the run planned from. Generated; archived to S3. |
| `records/gw<NN>_<season>.json` | The decision: XI, captain, transfers, counterfactuals. Hand-written; committed. |
| `records/plan_gw<NN>_<season>.json` | The planned transfers for every week of the horizon. Only week one is executed; the rest record why. Generated; committed. |
| `calibrate.py` | Scores every pool log against actuals. |
| `score.py` | Scores one week's decision, and flags if the fielded XI differed. |

`pool_gw04_2026-27.json` was generated retrospectively on 2026-09-18 with `--gameweek 4
--horizon 1` to prove the pipeline before GW5 was played. It is a legitimate out-of-sample
forecast — `known_through_gameweek` is 3 — not a fit to a known result.
