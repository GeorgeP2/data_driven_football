# 2026-27 season opener — findings

Investigation log, 24 July 2026. Started as "build my GW1 squad", turned into two bugs and a
measurement that overturned an assumption about the model.

For the running engineering log see `project-notes.md`; for how to run the tools see the project `README.md`.
This file is the record of one investigation and the decisions that came out of it.

---

## TL;DR

- **The GW1 squad stands** (below). Built by `fpl-optimise` on FPL's `ep_next`. Enter it now,
  re-run `--refresh` on **Friday 21 August before 17:30 UTC**.
- **Two silent bugs found and fixed** in the live advisor: it had no previous-season carry-over,
  and its history cache was not season-aware. Either alone made GW1 advice worthless.
- **The model must not pick the GW1 squad.** Measured over two seasons, its GW1 rank correlation
  is 0.009 and 0.027 — no signal at all. `ep_next` scores 0.468 and 0.443.
- **There is no crossover.** The model never durably overtakes `ep_next` on backtest evidence.
  This contradicts the standing assumption and is the most useful thing here.

---

## 1. The squad

`fpl-optimise --refresh`, 24 July 2026. £100.0m exactly, all rules satisfied.

```
Starting XI (5-4-1)                          Bench
  GKP  Raya          ARS  £ 6.0m  (C)         1. Pickford     EVE  £5.5m
  DEF  Gabriel       ARS  £ 8.0m  (V)         2. Andrews      COV  £4.5m
  DEF  Virgil        LIV  £ 6.5m              3. Bassette     COV  £4.5m
  DEF  O'Reilly      MCI  £ 6.5m              4. Walle Egeli  IPS  £4.5m
  DEF  Tarkowski     EVE  £ 6.0m
  DEF  Guéhi         MCI  £ 6.0m
  MID  B.Fernandes   MUN  £12.0m
  MID  Mbeumo        MUN  £ 8.0m
  MID  Cunha         MUN  £ 8.0m
  MID  Gibbs-White   NFO  £ 8.0m
  FWD  Welbeck       BHA  £ 6.0m
```

All five defenders start, so there is **no defender on the bench** and no like-for-like cover if
one is rested at the deadline.

### Value over replacement

For each player, the squad was re-solved with them banned. The drop in the solver's objective is
what they are actually worth over the next-best legal alternative.

| Player | Pos | £m | Cost of banning |
|---|---|---|---|
| Gabriel | DEF | 8.0 | **0.68** |
| Raya | GKP | 6.0 | **0.57** |
| Virgil | DEF | 6.5 | 0.17 |
| O'Reilly | DEF | 6.5 | 0.17 |
| B.Fernandes | MID | **12.0** | **0.15** |
| Mbeumo / Cunha / Gibbs-White | MID | 8.0 | 0.07 |
| Tarkowski, Guéhi, Welbeck, Pickford, Andrews, Bassette, Walle Egeli | — | — | **0.00** |

**Seven of the fifteen are worth nothing over their replacement.** They are tied with dozens of
alternatives and the solver broke the tie arbitrarily. Bruno at £12.0m — 12% of the budget —
contributes 0.15, because pre-season `ep_next` rates him 4.0, the same as Gabriel at £8.0m.

This is not the optimiser failing. It is reporting faithfully that a flat forecast cannot tell
these players apart: `ep_next` has only **25 distinct values across 555 players**, four of which
cover 60% of the game.

*Practical consequence: those seven slots are free rolls. Pick players you want to watch.*

---

## 2. Age is not modelled

No reference to age, `birth_date`, or date of birth anywhere in `src/fpl_optimiser/`. Not in the
squad optimiser, the points model, or the minutes model.

It could not be used today in any case: `birth_date` exists in the bootstrap payload but is
**null for 553 of 555 players**.

Age enters only implicitly, through form — a declining player eventually shows up as fewer
minutes. There is no age curve and no ageing prior, so the model learns a player has declined
only *after* it happens.

---

## 3. Two bugs in the live advisor

Both silent, both would have kept producing plausible-looking output.

### 3.1 No previous-season carry-over

`forecast_minutes` (`points_model.py:168`) falls back on `previous_season_minutes` when a player
has no completed gameweeks. But the two paths disagreed:

```
cli_backtest.py:114    SeasonHistory(records, rules, previous_season=previous)   ← wired
live.py:214            SeasonHistory(records, rules)                             ← not wired
```

`previous_season` defaults to `()`, so every player had zero prior minutes:

```
GW1 availability, nailed-on starter (3200 mins last season)
  live path    →  0.00
  backtest     →  1.00
after one played gameweek (live)  →  1.00
```

At GW1 all 555 players forecast a **0.00** chance of starting; the advisor's squad was arbitrary.
It recovered as real minutes accrued over the following weeks — which is exactly what "it takes a
few weeks before the mathematical approach starts working" felt like from outside.

**The backtest could never catch this**, because the backtest is the path that was wired
correctly. Same class of blind spot as the injury flags, pointing the other way.

### 3.2 The history cache was not season-aware

`_fetch_histories` reused `element_summaries.json` whenever the file existed, with no season
check. FPL renumbers element ids every summer, so at GW1 of 2026-27 it would have handed Raya
(now id 1) the gameweek history of whoever held id 1 in 2025-26.

Worse than the first bug: every player would then *appear* to have current-season minutes, so the
carry-over would never fire and the forecast would run on the wrong footballers entirely.

Cache files are now per-season (`element_summaries_2026-27.json`), matching how the seasons
archive already worked.

### 3.3 The cross-season join

FPL renumbers ids and the archive CSV carries no stable `code`, so last season's records are
re-keyed to this season's ids by **normalised full name, exact match only** — a wrong match hands
a player someone else's minutes, which is worse than the gap it fills.

```
2025-26 archive → 2026-27 pool:  450 of 555 matched (81.1%), 0 ambiguous
  of the 105 misses, 80 play for newly-promoted clubs (COV, HUL, IPS, LEE, SUN)
```

The rate is logged every run and **raises below 50%**. That floor guards a renamed column or
changed name format, which would otherwise silently return the system to the cold start it was
built to remove.

---

## 4. The measurement that changed the plan

Scored gameweek by gameweek on the rows FPL also forecast, over two independently cold-started
seasons (2025-26 from 2024-25, 2024-25 from 2023-24).

```
                  ours   ep_next    delta
  GW1 (2025-26)  0.009     0.468   -0.459    n=690
  GW1 (2024-25)  0.027     0.443   -0.415    n=616
  GW2+            ~0.73     ~0.79   -0.04    both seasons
```

### 4.1 GW1 is a cliff, not a slope

A rank correlation of 0.009 is not a weak forecast — it is a coin. With no current-season minutes
every per-90 rate is shrunk to its positional mean and the pool collapses into near-ties. One
round of real football restores it to ~0.73.

### 4.2 There is no crossover

The model is ahead of `ep_next` in only **~5 of ~30 usable gameweeks per season**, always
narrowly. The intuition that it overtakes once it has data is **false on this metric**.

This is why the handover is at GW2 and deliberately no later: there is no measured point at which
the model pulls ahead, so choosing a later handover would be fitting noise.

### 4.3 The caveat, which cuts the other way

This comparison is blind to injury flags, which the live advisor **does** get and which the
availability-oracle experiment valued at about **+0.16** — far more than the ~0.04 the model
trails by from GW2.

So the honest reading is: **GW1 is settled (use `ep_next`); GW2+ is not separable on backtest
evidence alone.** Do not read the ~0.04 as "FPL is better".

### 4.4 The defensive tilt is an artefact

Before the fallback existed, the model's GW1 squad was five defenders (three of them Brentford),
a **goalkeeper as captain**, and £8.5m unspent, with Haaland ranked 18th behind five defenders.

```
cold start (GW1-3)          full season
  GKP  0.553                  GKP  0.802
  DEF  0.515  ← worst         DEF  0.674  ← worst
  MID  0.499                  MID  0.744
  FWD  0.611  ← best          FWD  0.773
```

The model rated defenders *most confidently* exactly where it ranks them *worst*. Confidence and
accuracy pointing in opposite directions is the signature of a level miscalibration, not an edge.
Clean sheets, saves and defensive contribution are high-frequency, low-variance returns a per-90
model predicts confidently; goals are rare and get shrunk hard toward the positional prior.

---

## 5. What changed in the code

| File | Change |
|---|---|
| `live.py` | `normalise_name`, `season_label`, `previous_season_label`, `load_previous_season`, `early_season_forecast`, `MODEL_FROM_GAMEWEEK = 2`; per-season history cache; `load_live_history` takes `previous_season` |
| `cli_advise.py` | `forecast_gameweek` picks the forecast source per gameweek; `--previous-season`, `--no-carryover`, `--model-from` flags |
| `tests/test_carryover.py` | 25 tests |
| `README.md`, `project-notes.md` | Both findings documented |

**159 tests pass.** Nothing committed — `football/` is still untracked.

---

## 6. Actions

1. **Enter the squad above now.** FPL allows unlimited edits before the GW1 deadline, so there is
   no cost to a placeholder and some risk in forgetting.
2. **Re-run `fpl-optimise --refresh` on Friday 21 August, before 17:30 UTC.** Only 41 of 555
   players currently carry injury information; that fills in through pre-season and the Friday
   press conferences. The gain is mostly in *removing* players who are out, not in better ranking
   of the fit ones — `ep_next` will still be a pre-season prior, because GW1 has not been played
   by its own deadline.
3. **Use your own judgment on the seven zero-value slots.** It costs nothing measurable.
4. **From GW2, switch to `cli_advise`.** That is when the model takes over and this work pays.

---

## 7. Open threads

- **~105 promoted-club players have no prior at all** and keep the old behaviour rather than an
  `ep_next` blend. Whether to blend is a calibration question that deserves measuring first.
- **A stale 25MB `element_summaries.json`** sits in the live cache under the old season-blind
  name. Now ignored; safe to delete.
- **GW2+ remains unresolved.** Separating the model from `ep_next` there needs an evaluation that
  can see injury flags — which the archive does not retain. That is a data problem, not a
  modelling one, and it is the same conclusion the availability work reached.
- **Nothing is committed.**
