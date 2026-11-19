# victus-gta6-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/godfreycsh/victus-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/godfreycsh/victus-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**Victus Forecast** — a forecasting program for make.fun's
gta6-trophy-rarity-forecast-01: for every PS5 trophy of Grand Theft Auto VI,
forecast five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its Exophase earn
rate at the case's target, scored by mean pinball loss on the logit scale.
The program must beat the house persistence baseline (median = last read's
rate, spread = 0.5·√horizon-days) to take a prize.

## The model (main.py, Python stdlib only, deterministic)

The baseline's skeleton is kept, but everything it fixes by fiat is instead
estimated from the case's own history:

- **Drift view.** Per-trophy logit slope (recent 1.5-day window and full
  span, least squares), corrected for the fact that a window average
  overstates a decaying trend's endpoint value, shrunk toward the
  cross-trophy median slope by an empirical-Bayes weight (a slope counts
  only as far as it rises above its own estimation noise, vol²/Sxx), capped,
  and extrapolated over an exponentially decaying horizon
  (τ = 2.5 days — launch-week trends decelerate).
- **Count view.** Earners and players-tracked grow at different rates and
  the rate is their ratio: per-trophy earner growth (shrunk the same way)
  and shared player growth, both endpoint-corrected on the count decay
  (τ = 2 days), are projected over the horizon, and the resulting rate's
  logit is blended 65/35 with the drift view. The count view's implied
  shift is capped (±1.1/+1.3 logit) before blending: a launch-week
  catch-up burst — or a dilution plunge — in the counts cannot dominate
  the median on its own.
- **Saturation damping.** A positive blended shift is bent as the
  forecast rate approaches a ceiling share of tracked players (tier
  prior: platinum 12%, gold 35%, silver 65%, bronze 97%; secret trophies
  lower): earn rates saturate, they do not run through their ceiling.
  The prior is tightened per trophy by a ceiling fitted to its own
  history — a profile fit of dE = m·dP + π·(s·P − E)·dt over a grid of
  candidate ceilings, trusted only when it beats the prior's fit by a
  clear margin. Because the player path is inside the regression, a
  fading player-growth surge does not masquerade as a ceiling. The
  fitted ceiling also caps the shift, and the up tail compresses as the
  median nears the ceiling (the down tail does not — dilution room
  remains); negative shifts are never damped.
- **Spread.** Per-trophy logit volatility from consecutive read steps,
  detrended by the full-span slope so a steady trend is not read as
  noise, shrunk toward the cross-trophy median, scaled by √horizon, plus
  a trend-uncertainty term (0.40·|total shift|) and a small
  horizon-scaled base term; floored, and capped at 1.15 logit units —
  close to the house baseline's own spread at the final horizon
  (0.5·√4.52 ≈ 1.06). Volatility measured on bending launch-week series
  systematically overstates the noise in the smooth counts dynamics
  that actually move earn rates; the cap binds exactly there, while
  genuinely noisy histories rarely reach it.
- **Tails.** Mildly asymmetric in the direction of the trend. A rate pinned
  at the edge (≥ 99.9% or ≤ 0.1%) gets its inward tail widened with a
  spread floor of 0.55 — its clamped logit is nearly unidentified, and new
  tracked players can only dilute it one way. (A draft that also pulled a
  pinned median to 99.5% was tested and dropped: when the rate stays
  pinned it costs more than it saves, because the expensive upper
  quantiles are dragged down too.)
- Slopes and volatility are computed on count-smoothed logits
  ((earned + 0.5)/(players + 1) per read) so the 0.01/99.99 clamp does not
  fake a huge slope when a rate leaves 100%; the median starts from the raw
  last rate.

## Validation

- The practice cases' histories are constant (638 players, identical rates
  in every read — the page did not move in the recorded window), so a
  rolling-origin backtest on them (10–12 origins per case) can only check
  validity and short-horizon calibration: **ours 0.0078–0.0080 mean pinball
  loss vs the baseline's 0.0192–0.0201** on the same origins.
- The final case's regime (multi-day moving history, ~4.5-day horizon) is
  stress-tested on synthetic launch-week histories with known ground truth,
  in two families: logit-path scenarios (decelerating drift, growing
  denominator, per-trophy noise) and counts-based launch scenarios
  (players_tracked surges with decaying growth while each trophy's earners
  chase a tier-dependent ceiling share at a tier-dependent speed, so
  dilution emerges structurally). Mean pinball loss vs the persistence
  baseline on identical cases (40 seeds per scenario; negative is better),
  this version vs the first submitted version:

  | scenario | v1 | v2 | v3 |
  |---|---|---|---|
  | flat, calm | −0.084 | −0.094 | −0.093 |
  | flat, wild | −0.002 | −0.002 | −0.002 |
  | rising, calm | −0.121 | −0.125 | −0.125 |
  | rising, wild | −0.021 | −0.020 | −0.019 |
  | rising fast | −0.107 | −0.110 | −0.110 |
  | dilution (logit) | **+0.005** | **−0.021** | −0.021 |
  | launch base | +0.090 | +0.044 | **−0.058** |
  | launch dilution+ | −0.056 | −0.052 | −0.139 |
  | launch slow burn | +0.217 | +0.106 | **−0.017** |
  | launch surge+wild | +0.013 | +0.001 | −0.096 |
  | sparse launch | — | +0.052 | −0.046 |
  | sparse slow burn | — | +0.109 | −0.022 |
  | sparse rising | — | −0.060 | −0.060 |
  | sparse dilution | — | −0.054 | −0.054 |
  | **library total** | **−0.068** | **−0.227** | **−0.860** |

  (v1/v2 totals over the extended 14-scenario library; sparse scenarios
  thin the history to ~10 irregularly spaced reads, the final case's
  expected shape.) The v1→v2 changes are the count-view endpoint
  correction and shift cap, tier-prior saturation damping, detrended
  volatility, and the spread following the total shift. The v2→v3
  changes are the per-trophy fitted ceilings with headroom compression,
  and the spread cap brought from 1.7 down to 1.15. Built, measured,
  and rejected along the way: a structural cohort view (helped launch
  scenarios slightly, cost more on logit scenarios), an adaptive
  trend-decay time from slope ratios (too noisy), a two-segment
  detrended volatility (read genuine noise as trend), and per-level
  spread multipliers — the two scenario families want opposite
  calibrations (logit histories wider tails, launch histories
  narrower), and every fitted set failed cross-validation on the other
  family, so no global calibration constants are carried. With v3 the
  model beats the persistence baseline in every scenario in the
  library, including the near-ceiling ones where standing still is
  hardest to beat.
- Output contract verified on both practice cases: 55 rows, five finite
  non-decreasing quantiles in 0..100, 2,694 bytes (< 65,536 cap), runs
  under `python3 -P -s` in milliseconds. The competition is graded on the
  server only (program/v1), so there is no local grade to report.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=cewd94qm&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
