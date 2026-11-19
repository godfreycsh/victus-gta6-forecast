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
  forecast rate approaches a tier-plausible ceiling share of tracked
  players (platinum 12%, gold 35%, silver 65%, bronze 97%; secret
  trophies lower): earn rates saturate, they do not run through their
  ceiling.
- **Spread.** Per-trophy logit volatility from consecutive read steps,
  detrended by the full-span slope so a steady trend is not read as
  noise, shrunk toward the cross-trophy median, scaled by √horizon, plus
  a trend-uncertainty term (0.40·|total shift|) and a small
  horizon-scaled base term; floored and capped at 1.7 logit units.
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

  | scenario | v1 | v2 |
  |---|---|---|
  | flat, calm | −0.084 | −0.094 |
  | flat, wild | −0.002 | −0.002 |
  | rising, calm | −0.121 | −0.125 |
  | rising, wild | −0.021 | −0.020 |
  | rising fast | −0.107 | −0.110 |
  | dilution (logit) | **+0.005** | **−0.021** |
  | launch base | +0.090 | +0.044 |
  | launch dilution+ | −0.056 | −0.052 |
  | launch slow burn | +0.217 | +0.106 |
  | launch surge+wild | +0.013 | +0.001 |
  | **library total** | **−0.068** | **−0.273** |

  The v1→v2 changes are the count-view endpoint correction and shift cap,
  the saturation damping, detrended volatility, and the spread following
  the total shift. Two mechanisms were built, measured, and rejected: a
  structural cohort view (dE = m·dP + π·(P − E)·dt per trophy) that helped
  the launch scenarios slightly but cost more on the logit scenarios, and
  an adaptive trend-decay time from recent-vs-full slope ratios, which
  proved too noisy. The two launch scenarios still lost to persistence
  are the ones whose rates have essentially reached their ceilings by
  the end of the history — there, standing still is genuinely hard to
  beat, and v2's remaining gap is the price of moving at all.
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
