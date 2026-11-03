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

## Third reading (GW8 + GW9 2026-27, scored 2026-11-03)

Scored with `calibrate.py` from the committed `pool_gw08` and `pool_gw09` logs; both weeks were
`finished` and `data_checked`. The GW6 and GW7 logs are in S3, not on this machine, so they are
not in the pooled numbers.

| | GW4 | GW5 | GW8+GW9 |
|---|---|---|---|
| overall actual/xP | 1.22 | 1.26 | 1.19 (GW8 1.24, GW9 1.17) |
| rank correlation, all depths | 0.7610 | 0.7556 | 0.7366 |
| vs FPL `ep_next` (depth 0) | 0.8386 — ep_next ahead | 0.7466 — model ahead | 0.7500 vs 0.7247 — **model ahead, both weeks** |
| DEF / GKP / FWD / MID actual/xP | 1.38 / 1.46 / 1.25 / 1.07 | 1.43 / 1.43 / 1.08 / 1.17 | 1.41 / 1.33 / 1.07 / 1.07 |
| depth +1 (forecast a week early) | | | actual/xP 1.18, rank correlation 0.7116 |

Four readings, and DEF has sat at 1.4 in all four, across high-scoring weeks (GW4) and low
(GW5, GW8, GW9). Trap 1 no longer explains it. The head-to-head is now 3-1 to the model, and
the week-early forecast loses only 0.04 of rank correlation, so the planner's trust in depth +1
is roughly earned (depths +2 to +4 are still unscored).

Restricted to players who got minutes (n=613) the whole-pool correlation again collapses, to
0.32 for the model and 0.26 for `ep_next` — same shape as GW5.

### Where the defender points go missing

The GW9 forecast was rebuilt from the saved snapshot and the cached element summaries (it
reproduces the logged pool to four decimal places) and split into its scoring components, for
every defender who played 60+ minutes (n=80):

| component | model | actual | gap |
|---|---|---|---|
| appearance | 1.45 | 2.00 | +0.55 |
| clean sheet | 0.63 | 1.35 | +0.72 |
| defensive contribution | 0.13 | 0.60 | +0.47 |
| goals conceded | −0.49 | −0.34 | +0.15 |
| bonus | 0.20 | 0.25 | +0.05 |
| goals + assists | 0.36 | 0.22 | −0.14 |
| cards (not modelled) | 0.00 | −0.29 | −0.29 |
| **total** | **2.29** | **3.80** | **+1.51** |

The appearance gap is conditioning, not bias: the table only contains players who did play,
and the minutes model gave them 0.69 for the 60-minute point. Midfielders show the same +0.57.
What is defender-specific is the next two rows.

- **Defensive contribution is the clear one.** The per-90 rate is shrunk toward the positional
  mean (7.8), multiplied by the expected-minutes share (~0.75), then put through Poisson
  P(≥10). A mean of ~5.9 gives 8%; 30% of these defenders actually cleared the bar. Two things
  compound: the threshold is convex, so the minutes discount hits it twice (once on the share,
  again through the tail), and shrinking toward one DEF mean erases the centre-back /
  full-back split, which is most of the signal. Worth ~0.5 xP per starting defender per week.
- **Clean sheets are short but not yet separable from the week.** The model implied 0.22 per
  match (exp(−1.51) on the fixture ratings) against 0.34 realised; GW8 and GW9 averaged 45 and
  47, so some of that is two quiet weekends. Needs more weeks.
- **Yellow cards are not modelled** and cost defenders 0.29 and midfielders 0.19 per
  player-week. Small, steady, and in the other direction: it offsets, it does not explain.

What this means for the planner: the ordering *within* defenders is fine (rank correlation is
what it consumes), but a transfer compares a defender against a midfielder, and defenders are
systematically ~0.4 xP a week cheap in that comparison.

### The decisions

**GW9.** Virgil → Konsa (2.78 → 3.69 xP): 1 point against 10. Captain Haaland, 7.22 xP and
top of the pool on both the model and `ep_next`: 2 points. The fielded XI was exactly the
model's best XI from the 15 (3-5-2), so the score is like-for-like: forecast 54.9, actual 45
against a 47 average. Guéhi was fielded on 0.92 availability and did not play (dropped after
45 minutes the week before; the minutes model cannot see that), the auto-sub brought in Ajayi
for 2 with De Cuyper's 9 last on the bench. Best XI from the 15 in hindsight: 62.

**GW8.** The plan's three transfers were made as written (Saka → Tavernier, B.Fernandes →
Rudoni, Walle Egeli → Haaland, −8): the three players sold scored 0 between them, the three
bought 20, so the hit returned +12 net. Fielded XI 58 gross against a 53.4 forecast.

**Plan churn.** Of the four forward moves in `plan_gw08` (GW9–12), none survived the GW9
re-plan intact: the GW9 move became Virgil → Konsa (Evanilson, the planned buy, scored 9; Konsa
10), and the later weeks were all re-written. Vuskovic appears in both plans but for a
different player out. One data point, but it is the first measure of how much the far end of
the horizon is worth.

### Follow-up: the bonus term, per position

The GW9 component split had forwards who played 60+ minutes short 0.78 bonus a week, on
eighteen rows. Checked on the 2025-26 archive it is real but not what it looked like. Scored
as `P(60) * curve(rate)` against bonus actually awarded, the single curve ran

| | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| actual/forecast, 2025-26 | 0.69 | 0.72 | 0.84 | 1.45 |
| actual/forecast, 2024-25 | 0.67 | 0.67 | 0.64 | 1.41 |

Same split both seasons: the curve over-paid everyone except forwards. The reason is in what
bonus is — a rank within the match — so the *spread* of a player's BPS matters as much as its
mean, and a forward's is spikier. Fitting one curve per position on 2025-26 and checking on
2024-25 untouched gives 1.01 / 0.99 / 0.92 / 1.32. Forwards are still short on the held-out
season (their bonus runs higher some years), but the gap halved and nothing else moved.

The surprise is the top end. At a 50-BPS rate — Haaland — the old curve paid 2.7 bonus a match;
the measured rows scored about 1. So the fix *lowers* Haaland (7.22 → 6.23 xP in the rebuilt
GW9) while lifting mid-priced forwards, and the whole-pool GW9 ratios barely move. The bonus
term was not where the premium-forward gap lives; if it is anywhere it is in the goals term.
Whole-model rank correlation is unchanged on both seasons; Pearson and MAE improve by a hair.

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
