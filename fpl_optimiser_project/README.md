# FPL Optimiser

> **Category:** Optimisation & simulation · **Status:** 🚧 In progress

Picks the highest expected-points Fantasy Premier League squad that is legal under the game's
rules: 15 players, £100.0m, 2 goalkeepers / 5 defenders / 5 midfielders / 3 forwards, at most
three players from any one club, a valid starting XI, and a captain.

```bash
# from the repo root, with the venv active (make setup)
cd fpl_optimiser_project
PYTHONPATH=src python -m fpl_optimiser.cli
```

```
Starting XI (3-5-2)
  GKP  Raya                 ARS  £ 6.2m   8.80 xP (C)
  DEF  Gabriel              ARS  £ 7.3m   8.80 xP (V)
  ...
Cost £94.9m  |  XI expected points (captain doubled): 93.90  |  bench: 24.20
```

## How it works

Squad selection is a 0-1 integer program, solved with OR-Tools CP-SAT. Each player gets three
binary variables — in the 15, in the starting XI, wearing the armband — and the FPL rules become
linear constraints over them. The objective is:

```
sum over players of  ep[p] * ( start[p] + capt[p] + bench_weight * (squad[p] - start[p]) )
```

The captain term adds the captain's expected points a second time, which is what the armband
does. `bench_weight` is the one genuinely judgement-based knob: at zero, the solver has no reason
to prefer a good bench over a worthless one and will fill all four slots with the cheapest bodies
it can find to free up money for the XI. A small positive weight (the default is 0.1) buys some
insurance against a starter being dropped or injured at the deadline. Try `--bench-weight 0` and
watch the bench collapse.

Everything else falls out without a solver: the vice-captain is the next-best starter, and the
bench is ordered by expected points behind the reserve keeper, whom FPL fixes in the first slot.

Prices stay in the API's integer tenths of a million (`now_cost: 62` is £6.2m) all the way
through, so the budget constraint is exact rather than subject to float drift. The rules
themselves — squad size, budget, the per-club cap, the position quotas and the formation
bounds — are read from `bootstrap-static` rather than hardcoded, so the model tracks the game if
FPL ever changes it.

## The part that actually matters

**The optimiser is not where the edge is.** Given expected points per player, finding the best
squad is a solved problem that takes milliseconds. Forecasting those expected points is the hard
part, and the whole game.

The default provider (`EpNextProvider`) reads FPL's own published `ep_next`, discounted by the
player's availability — a 75%-fit player is worth 75% of their points, an injured one nothing.
That signal is weak: it is roughly recent form with a fixture nudge. The optimiser will faithfully
return the best squad *according to a mediocre forecast*.

So expected points is a pluggable callable, not a column baked into the solver:

```python
from fpl_optimiser import GameRules, StaticProvider, fetch_bootstrap, optimise_squad, parse_players

payload = fetch_bootstrap()
rules = GameRules.from_bootstrap(payload)
players = parse_players(payload, rules)

squad = optimise_squad(players, rules, StaticProvider({308: 9.1, 427: 7.4}))
```

Any callable taking a `Player` and returning a float will do. A real model — minutes model x
per-90 scoring rate x fixture difficulty, fitted on historical gameweek data — drops in behind
the same protocol without the solver knowing anything changed. That is the natural next build.

## Options

| Flag | Effect |
| --- | --- |
| `--budget 83.5` | Budget in millions (default: the game's £100.0m). |
| `--bench-weight 0..1` | How much bench points count (default 0.1). |
| `--lock ID [ID ...]` | Players that must be in the squad. |
| `--exclude ID [ID ...]` | Players to keep out. |
| `--ignore-availability` | Do not discount for injury/doubt. Answers "who would I pick in a healthy world?" — not for picking a real squad. |
| `--refresh` | Re-fetch instead of using the cached payload. |

`bootstrap-static` is cached under `~/.cache/fpl_optimiser/` for six hours, since prices and
injury flags move at most a couple of times a day.

## Tests

```bash
make check    # from the repo root
```

A squad optimiser fails *plausibly* — it hands back fifteen real footballers that quietly break a
rule — so the tests make each constraint bind rather than merely asserting an already-legal squad
is legal. There is a pool where the greedy pick would take four players from one club, one where
the best striker is unaffordable, one where all the value sits in midfield. Two independent
oracles guard the objective: a brute-force search over every legal formation checks the XI is the
best XI from the chosen 15, and a greedy points-per-million baseline that the solver must never
lose to.

## The weekly advisor

```bash
PYTHONPATH=src python -m fpl_optimiser.cli_advise --team-id <your-entry-id> --free-transfers 2
```

Pulls your actual squad, forecasts the upcoming gameweek, plans transfers over a five-week
horizon, and tells you what to do — the transfer, the XI, the captain, the chip. It recommends;
it does not act, because the model has no idea what a manager said in a press conference an hour
ago.

Two caveats worth knowing. **Purchase prices are not public** — only the authenticated
`/my-team/` endpoint has them — so today's market price stands in, which means the sell-on fee is
not applied and the planner thinks it has slightly more money than it does. And the advisor needs
an upcoming gameweek: between seasons it will tell you so rather than guess.

### The cold start, and why last season is loaded

At GW1 nobody has played, so the minutes model has nothing to predict from — and a player it
cannot see playing is worth nothing, which quietly makes the whole squad arbitrary. So the
advisor seeds itself from last season's archive: a player who logged 3,300 minutes is a starter,
one who logged 400 is not.

FPL renumbers element ids every season and the archive does not carry the stable `code`, so the
join runs on normalised full names — exact match only, because handing a player someone else's
minutes is worse than the gap it would fill. Joining 2025-26 onto 2026-27 matches **81% of the
pool with no ambiguity**; of the 19% that miss, most play for newly-promoted clubs and genuinely
have no Premier League history. Those keep the old behaviour rather than inheriting a guess.

The match rate is logged every run and **falls over below 50%**, because the failure it guards
against — a renamed column, a changed name format — is otherwise silent and returns you to
exactly the cold start this exists to remove.

`--previous-season YYYY-YY` overrides which season is carried; `--no-carryover` turns it off.

### GW1 uses FPL's forecast, not ours

The carry-over stops the model being *degenerate* at GW1. It does not make it *good*. Scored
gameweek by gameweek across two cold-started seasons:

```
                  ours   ep_next
  GW1 (2025-26)  0.009     0.468
  GW1 (2024-25)  0.027     0.443
  GW2+            ~0.73     ~0.79
```

A rank correlation of 0.009 is not a weak forecast, it is a coin: with no current-season minutes
every per-90 rate is shrunk to its positional mean and the pool becomes one big tie. So the
advisor uses FPL's `ep_next` before `--model-from` (default GW2) and its own model from there.

Note what this table does *not* show: a crossover. The model never durably overtakes `ep_next` —
it wins ~5 of ~30 usable gameweeks. But the comparison is blind to injury flags, which the live
advisor gets and which are worth more than the gap, so from GW2 the two are not separable on
backtest evidence. GW1 is the only week where the answer is unambiguous, and that is the only
week the fallback claims.

`--model-from 1` restores the old behaviour if you want to see it.

## How the model works

Three parts, in order of how much they matter.

**The minutes model** is the biggest lever, and it exists because of what the backtest found: the
naive baselines fielded five players in GW1 who never took the pitch. "Will they play?" is worth
more than "how well will they play?" — a brilliant player who is benched scores nothing. It
predicts starts from recent minutes with a fast decay (losing your place in a football team is
abrupt, and the model treats it that way), and in a live season it also reads FPL's injury flags,
which are not retained historically and are therefore the one advantage the advisor has over its
own backtest.

**Expected points** follows FPL's scoring rules component by component — appearance, goals,
assists, clean sheet, saves, defensive contribution, bonus, goals conceded — each as a per-90
rate, shrunk toward the positional average, then scaled by expected minutes and fixture
difficulty. It uses *expected* goals rather than actual goals: xG is a far steadier estimate of a
player's scoring than the handful of goals he happens to have scored.

**The planner** looks five gameweeks ahead and executes only the first, re-planning each week.
This is what stops the squad churning. A one-week planner panics at blank gameweeks — it sees
half the squad with no fixture, takes four transfers to chase players who do play, pays a
16-point hit, and churns back the following week. Over the 2025-26 run-in that cost 48 points.
Looking ahead cut total hits from 68 to 32.

## Backtest

```bash
PYTHONPATH=src python -m fpl_optimiser.cli_backtest --season 2025-26
```

Plays a completed season under the real transfer rules — pick at GW1, then one free transfer a
week (bankable up to five), −4 per extra, sell-on fee charged on profit, auto-substitutions
applied — and scores each expected-points model against what actually happened.

```
  system                                    net points   vs a real manager (2106)
  full system  (model + planner + chips)          2126                       +20
  model + 5-week planner                          2114                        +8
  model + 3-week planner                          2082                       -24
  model + greedy 1-week transfers                 2009                       -97
  points-per-game baseline                        1957                      -149
  trailing-form baseline                          1746                      -360
  random legal squad                               914                     -1192
  ---
  perfect foresight (the ceiling, not a target)   4675
```

**Do not read the +20 as a win. It is noise, and we can prove it.**

Run the *same* model over the *same* season with a ±1% jitter in the forecast — a perturbation
that should change nothing — and the season total lands anywhere between 2,097 and 2,175.
Standard deviation: 30 points.

```
  seed 0: 2139    seed 2: 2129    seed 4: 2175
  seed 1: 2155    seed 3: 2105    seed 5: 2097

  => a difference of less than ~60 points is not an improvement, it is a coin landing well.
```

That number is the single most useful thing in this repository. A season score is **one
observation**, and roughly ten hyperparameters were tuned against it — which is curve-fitting, not
model selection. Any future change that gains less than ~60 points has not been shown to do
anything at all.

What the table *does* show reliably is where the points are: the planning horizon is worth ~105
over greedy transfers, the forecast ~50 over the naive baselines. Perfect foresight is the
ceiling, and it is more than twice as far away again.

## Measuring the forecast, not the season

```bash
PYTHONPATH=src python -m fpl_optimiser.cli_evaluate --season 2024-25
```

A season total is one data point. The forecast underneath makes a prediction for **every player in
every gameweek** — about 30,000 of them, each with a known answer. That is three orders of
magnitude more evidence, it runs in seconds rather than ten minutes, and unlike a season score it
cannot be moved by a lucky captain pick.

It reports rank correlation against actual points (the metric that matters — the optimiser only
needs to know who out-scores whom), the minutes model's calibration, and a head-to-head against
**FPL's own `ep_next`**, which the archive snapshots weekly and which is a genuinely independent
yardstick.

### What it found

- **The minutes model was badly over-confident.** It said 98% chance of starting; 86% of those
  players started. Capping it at the measured ceiling (0.88) improved MAE, Brier *and* rank
  correlation at once.
- **Better calibration made the model worse.** Adding a 5% *floor* — which is genuinely better
  calibrated, since fringe players do start about one game in twenty — improved the Brier score and
  *hurt* the points forecast (rank correlation 0.707 → 0.676), because it hands ~15,000 non-playing
  players a small noisy forecast where zero was right. Calibration is a means, not the goal.
- **The bonus model fell to Jensen's inequality.** Bonus is a rank within a match (top three BPS
  take 3/2/1), so I applied that S-curve to each player's *average* BPS — which systematically
  under-predicts, because a player averaging 20 BPS sometimes has a 40-BPS game. The curve that
  matters is lagged-rate-to-outcome, and it is far flatter.

## Suspensions: the one thing that is not a forecast

A banned player has a zero percent chance of playing. That is arithmetic, and it is the only
component here that needs no backtest to justify it.

It exists because suspensions are one of the ways a "nailed on" starter silently fails to appear.
Of the 4,900 players the model was ≥80% sure would start in 2024-25, 692 did not — and 65 of those
were serving bans that were knowable with certainty from cards already shown.

**The cutoff is the entire rule, and getting it wrong would have made the system worse.** A fifth
yellow earns a one-match ban *only if it arrives inside the club's first 19 matches*:

```
  5th yellow within the cutoff  ->  missed the next gameweek   43/43  (100%)
  5th yellow past the cutoff    ->  missed the next gameweek    8/81  ( 10%)
```

Implemented from memory, without the cutoff, this would have confidently benched 81 players who
went on to play. A false "he is banned" removes a good player from selection entirely — which is
exactly the failure mode that makes bad availability data worse than none.

**Yellow and red cards are not the same kind of fact**, and the model treats them differently:

| | mechanism | lands | treatment |
|---|---|---|---|
| yellow accumulation | administrative, automatic | 96% | fixtures dropped outright |
| red card | appealable, downgradeable | 85% | heavy discount, not exclusion |

Wrongly ruling a player out is the expensive direction to be wrong in, so certainty earns an
exclusion and 85% earns a discount. In a double gameweek a one-match ban leaves the player
available for the *other* fixture, so bans are counted in matches rather than flagged — that
player is still worth owning.

Measured on 2024-25: 54 yellow-card bans applied, 52 correct (96% precision), and 56 confident
ghosts correctly removed from selection.

### Why we "lose" to FPL, and why it does not matter

Our forecast scores 0.707 on rank correlation against FPL's 0.752. That looks bad until you ask
where the gap is:

```
  ours (blind to injuries)          0.707
  FPL ep_next (sees injury flags)   0.752
  ours + availability oracle        0.868
```

Given equal information about who is fit, our model is **far better** than FPL's. The entire gap
is that `ep_next` is computed with access to the injury flags, and those flags are not retained
historically — so the backtest can never see them.

**In a live season we get them.** Which means the backtest systematically *understates* the live
system, and the biggest remaining lever is not the model at all: it is better availability
information — press conferences, predicted line-ups, rotation risk. A data problem, not a
modelling one.

**You cannot backtest `EpNextProvider`.** FPL's own forecast is a live-only field; the API does
not retain past values, and the community archive's snapshot of it is zeroed for 27 of 2025-26's
38 gameweeks. Backtest providers therefore build their forecasts from lagged history only.

**What it is for.** Auto-substitutions are the only mechanism by which a good bench pays off, so
this is what lets you tune `bench_weight` with evidence instead of taste. Sweeping it over
2025-26: a weight of 0 scores 1835, and 0.05–0.1 scores ~1960. A worthless bench really does
cost you. Beyond that the curve is noise — one season is a single sample, so do not read an
optimum into it.

### Things the backtest caught that would otherwise have shipped

Every one of these looked fine in code review and was wrong in fact. This is what the harness is
*for*.

- **The chips lost points.** The first version scored 2,080 against 2,108 without them. It fired
  the triple captain in GW2 — when the model is running on priors and is least trustworthy — and
  a wildcard in GW38, which is worth nothing. Worse, the wildcard rebuild used the
  *single-gameweek* solver: it built a squad to win one week and then fall apart, when a wildcard
  is precisely the decision you keep. (A free hit is the opposite — it reverts, so one week is
  the right horizon. Same chip mechanic, opposite optimisation.) Fixed, chips are worth +12.
- **The league average was circular.** Team strength was shrunk toward a league mean computed
  from the same handful of matches it was meant to calibrate, so a high-scoring opening weekend
  convinced the model that three goals a game was normal.
- **The minutes model forgot too slowly.** A player benched three weeks running still rated a 30%
  chance of starting — which is how you end up fielding ghosts.

### Still not modelled

The transfer planner holds prices fixed across its horizon (predicting price rises is a separate
forecasting problem, and being wrong about it is worse than ignoring it). Chip *timing* is a set
of explicit thresholds rather than part of the integer program — auditable, but not provably
optimal.

## Scope

The picker itself is single-gameweek: no chip logic, no multi-gameweek horizon. It picks a squad
from scratch, which is what you want for a wildcard or a season opener. The backtest adds the
season-long transfer simulation on top.

## Mini-league pages

Separate from the optimiser, `fpl_optimiser.leagues` and `notebooks/01_league_simulation.ipynb`
forecast *mini-league standings* from manager history (a Monte Carlo over each manager's weekly
scores) and render one stats page per league. They have nothing to say about which players to
own. See [docs/league-pages.md](docs/league-pages.md).

## Data

| Source | What | Use |
|--------|------|-----|
| [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Live players, prices, fixtures, entry histories and picks | Optimiser, advisor, league pages |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Per-gameweek archive of past seasons | Backtest, forecast evaluation, GW1 carry-over |

Everything is fetched by the code on demand. API and archive caches go to `~/.cache/fpl_optimiser/`;
league histories and picks go to `data/`, run outputs to `outputs/` (both git-ignored). The FPL API
is unofficial and undocumented, so it is used for personal analysis only.

## Docs

- [docs/project-notes.md](docs/project-notes.md) — engineering log, including the ideas that failed measurement
- [docs/gw1-2026-27-findings.md](docs/gw1-2026-27-findings.md) — the season-opener investigation
- [docs/wildcard-timing.md](docs/wildcard-timing.md) — design note for wildcard timing
- [docs/forecast-log.md](docs/forecast-log.md) — weekly forecast logging and calibration
- [docs/league-pages.md](docs/league-pages.md) — the mini-league pages

## Reproduce

```bash
# from the repo root, with the venv active (make setup)
cd fpl_optimiser_project
PYTHONPATH=src python -m fpl_optimiser.cli                                  # best squad this week
PYTHONPATH=src python -m fpl_optimiser.cli_advise --team-id <your-entry-id> # weekly advice
PYTHONPATH=src python -m fpl_optimiser.cli_backtest --season 2025-26        # season simulation
PYTHONPATH=src python -m fpl_optimiser.cli_evaluate --season 2024-25        # forecast scoring
PYTHONPATH=src python -m fpl_optimiser.forecast.calibrate                   # live calibration
PYTHONPATH=src python -m fpl_optimiser.leagues.export && \
  PYTHONPATH=src python -m fpl_optimiser.leagues.build                      # league pages
pytest tests
```

## Skills demonstrated

- Integer programming (OR-Tools CP-SAT) with the game's rules read from the API, not hardcoded
- Component-wise expected-points model: minutes, per-90 rates with shrinkage, fixture difficulty
- Rolling-horizon transfer planning under real transfer, hit and sell-on rules
- Season backtesting with a measured noise band, and forecast evaluation on ~30,000 predictions
- Monte Carlo mini-league simulation with significance-tested week-on-week changes
