"""gta6-trophy-rarity-forecast-01 entry "Victus Forecast" (Python, stdlib only).

Model, in one paragraph: the house baseline is persistence on the logit scale
(median = last read's rate, spread = 0.5 * sqrt(horizon in days)). This entry
keeps that skeleton but estimates, from the case's own history, (1) a
per-trophy logit drift, shrunk hard toward the cross-trophy median drift and
extrapolated with an exponentially decaying trend (launch-week trends
decelerate, so a trend measured over the history is not projected linearly
over the whole horizon); (2) a count view of the same move -- earners and
players-tracked grow at different rates, and the rate is their ratio -- whose
growth rates are endpoint-corrected like the drift (a window's average
growth overstates the growth at its end when growth itself is decaying, the
usual launch-week shape: the player pool's growth front-loads, so the raw
average understates how fast dilution is fading) and whose implied median
shift is capped before it is blended with the drift view, so a launch-week
catch-up burst in the counts cannot extrapolate into an explosion; and (3)
a per-trophy logit volatility, measured on slope-detrended steps so a
steady trend is not mistaken for noise, shrunk toward the cross-trophy
median, which together with a trend-uncertainty term sets the spread. A
positive blended shift is also damped as the forecast rate approaches a
tier-plausible ceiling share (platinum trophies saturate far below bronze
ones): earn rates bend toward their ceiling instead of running through it.
Slope estimates are shrunk by an empirical-Bayes weight (a trophy's slope
counts only as far as it rises above its own estimation noise, which grows
with volatility), so calm histories keep their drift and wild ones fall
back toward the global drift with a wider spread. The series used for
slopes and volatility is count-smoothed with each read's players_tracked
((earned + 0.5) / (players + 1)) so the 0.01/99.99 clamp's logit of 9.2
does not fake a huge slope when a rate leaves 100%; the median itself
starts from the raw last rate, with one guard: a rate at the very edge
(>= 99.9% or <= 0.1%) has an almost unidentified logit, so the tail
pointing back toward the middle is widened (with a floor on that side's
spread) while the median stays at the drift forecast -- cheap insurance if
the rate stays pinned, real cover if new tracked players dilute it. (A
draft that also pulled the median to 99.5% was dropped: the synthetic
stress test showed it costs more than it saves when the rate stays pinned,
because the upper quantiles, whose pinball weights are the expensive ones,
are dragged down with it.) Quantiles are also mildly asymmetric in the
direction of the estimated drift: when a trophy is trending up, its upper
tail is wider than its lower tail.

Two further mechanisms were built and rejected on the stress library
before this version shipped: a structural cohort view (fitting
dE = m*dP + pi*(P - E)*dt per trophy to separate new-player dilution from
existing-player progression) helped the counts-based launch scenarios a
little but cost more on logit-trend scenarios than it earned, and an
adaptive trend-decay time estimated from recent-vs-full slope ratios
proved too noisy to trust. The shipped model is the combination that won
the full scenario library head-to-head against the previous version.

Everything is deterministic, O(reads x trophies), standard library only, and
keeps the starter's I/O contract exactly: read every *.in in
$MAKEFUN_INPUT_DIR, write <id>.out = {"trophies": [{"id": ..., "q": [q05,
q25, q50, q75, q95]}, ...]} with five finite, non-decreasing quantiles in
0..100 per trophy, rounded to 4 decimals half-up like the starter.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (starter's line).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (chosen by reasoning about the final horizon, and ---
# --- stress-tested on synthetic launch-week histories; see repo README) ---
DRIFT_TAU_DAYS = 2.5      # trend decay time for the logit-drift view
COUNT_TAU_DAYS = 2.0      # growth decay time for the count view
RECENT_WINDOW_DAYS = 1.5  # window for the "recent" slope; full span also used
DRIFT_SHRINK = 0.55       # weight on a trophy's own slope vs the global median
COUNT_SHRINK = 0.5        # weight on a trophy's own earner growth vs global
BLEND_DRIFT = 0.65        # median blend: drift view vs count view
VIEW_SHIFT_DOWN = -1.1    # cap on the count view's implied median shift
VIEW_SHIFT_UP = 1.3
SHIFT_CAP_DOWN = -1.4     # caps on the total median shift, in logit units
SHIFT_CAP_UP = 2.2
SLOPE_CAP_DOWN = -1.2     # caps on a shrunk daily logit slope
SLOPE_CAP_UP = 1.8
SIGMA_CAP = 1.7
VOL_FLOOR = 0.08
TREND_UNC = 0.40          # spread grows with |shift|: trend may not persist
BASE_UNC = 0.08
# Tier-plausible ceiling shares (fraction of tracked players) for the
# saturation damping of positive shifts; secret trophies sit lower.
TIER_CAP = {"platinum": 0.12, "gold": 0.35, "silver": 0.65, "bronze": 0.97}
SECRET_CAP_MULT = 0.8


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, exactly as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO timestamp -> epoch seconds (float), or None."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def smoothed_logit(rate: float, players) -> float:
    """Logit of the count-smoothed rate: (earned + 0.5) / (players + 1).

    With no usable player count this is the plain clamped logit. The
    smoothing matters only at the extremes (a 100% rate on 638 players
    becomes a logit near 7.2 instead of the clamp's 9.2; a 4-earner trophy
    is pulled slightly toward the middle), and vanishes as counts grow.
    """
    if players and players > 0:
        earned = rate / 100.0 * players
        pct = 100.0 * (earned + 0.5) / (players + 1.0)
        return logit(pct)
    return logit(rate)


def ols_slope(ts, ys):
    """Least-squares slope of ys against ts (both plain lists); 0 if flat."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    my = sum(ys) / n
    num = 0.0
    den = 0.0
    for t, y in zip(ts, ys):
        dt = t - mt
        num += dt * (y - my)
        den += dt * dt
    if den <= 0:
        return 0.0
    return num / den


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def cap(x, lo, hi):
    return min(hi, max(lo, x))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    # --- read times, in days relative to the first read -------------------
    raw_times = []
    for read in history:
        t = parse_time(read.get("at")) or parse_time(read.get("slot"))
        raw_times.append(t)
    if all(t is not None for t in raw_times):
        t0 = raw_times[0]
        times = [(t - t0) / 86400.0 for t in raw_times]
    else:  # no usable clock: assume the house's 30-minute cadence
        times = [0.5 * i / 24.0 for i in range(len(history))]
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    recent_from = times[-1] - min(span, RECENT_WINDOW_DAYS)
    recent_width = min(span, RECENT_WINDOW_DAYS)

    # --- per-trophy series -------------------------------------------------
    # For each trophy: times, smoothed logits, log earner counts, last values.
    per = []
    for i in range(len(trophies)):
        ts, lgs, les = [], [], []
        last_rate, last_players, last_logit = None, None, None
        last_raw_logit = None
        for read, t in zip(history, times):
            rate = read["rates"][i] if i < len(read.get("rates") or []) else None
            if rate is None:
                continue
            players = read.get("players_tracked")
            lg = smoothed_logit(rate, players)
            ts.append(t)
            lgs.append(lg)
            if players and players > 0:
                les.append(math.log(rate / 100.0 * players + 0.5))
            else:
                les.append(None)
            last_rate, last_players, last_logit = rate, players, lg
            last_raw_logit = logit(rate)
        per.append({
            "ts": ts, "lgs": lgs, "les": les,
            "last_rate": last_rate, "last_players": last_players,
            "last_logit": last_logit, "last_raw_logit": last_raw_logit,
        })

    # Global fallbacks for trophies with no usable series.
    known_logits = [p["last_raw_logit"] for p in per
                    if p["last_raw_logit"] is not None]
    global_last_logit = median(known_logits) if known_logits else logit(1.0)

    # --- slopes: recent window and full span, per trophy -------------------
    def window_slope(ts, ys, since):
        wt = [t for t in ts if t >= since]
        wy = [y for t, y in zip(ts, ys) if t >= since]
        if len(wt) >= 2 and wt[-1] - wt[0] > 1e-9:
            return ols_slope(wt, wy)
        return ols_slope(ts, ys)

    def endpoint(slope_w, width, tau):
        # A window's OLS slope averages a decaying trend over the window, so
        # it overstates the trend at the window's end (the point the forecast
        # extrapolates from) by (tau/W) * (exp(W/tau) - 1) for window W.
        # Divide that factor back out so the result estimates the
        # instantaneous slope at the last read.
        if width <= 1e-9:
            return slope_w
        factor = (tau / width) * (math.exp(width / tau) - 1.0)
        return slope_w / factor

    for p in per:
        if len(p["ts"]) >= 2 and span > 1e-9:
            p["slope_recent"] = window_slope(p["ts"], p["lgs"], recent_from)
            p["slope_full"] = ols_slope(p["ts"], p["lgs"])
        else:
            p["slope_recent"] = 0.0
            p["slope_full"] = 0.0
        p["slope"] = 0.6 * endpoint(p["slope_recent"], recent_width, DRIFT_TAU_DAYS) \
            + 0.4 * endpoint(p["slope_full"], span, DRIFT_TAU_DAYS)
        # earner-count growth on the recent window (None where no counts),
        # endpoint-corrected on the count view's own decay time
        pairs = [(t, e) for t, e in zip(p["ts"], p["les"]) if e is not None]
        if len(pairs) >= 2 and span > 1e-9:
            wt = [t for t, _ in pairs if t >= recent_from]
            we = [e for t, e in pairs if t >= recent_from]
            if len(wt) >= 2 and wt[-1] - wt[0] > 1e-9:
                p["g_earn"] = endpoint(ols_slope(wt, we),
                                       recent_width, COUNT_TAU_DAYS)
            else:
                p["g_earn"] = ols_slope([t for t, _ in pairs], [e for _, e in pairs])
        else:
            p["g_earn"] = None
        # random-walk volatility per day from consecutive logit steps,
        # detrended by the full-span slope so trend is not read as noise
        if len(p["ts"]) >= 2:
            num = 0.0
            den = 0.0
            for (t1, y1), (t2, y2) in zip(zip(p["ts"], p["lgs"]),
                                          zip(p["ts"][1:], p["lgs"][1:])):
                dt = t2 - t1
                if dt > 1e-9:
                    step = (y2 - y1) - p["slope_full"] * dt
                    num += step ** 2
                    den += dt
            p["vol"] = math.sqrt(num / den) if den > 0 else 0.0
        else:
            p["vol"] = None
        # Estimation noise of the slope: Var(slope) ~= vol^2 / Sxx over the
        # recent window (the window the combined slope leans on most).
        wt = [t for t in p["ts"] if t >= recent_from]
        if len(wt) >= 2 and p["vol"] is not None:
            mt = sum(wt) / len(wt)
            sxx = sum((t - mt) ** 2 for t in wt)
            p["slope_var"] = (p["vol"] ** 2) / sxx if sxx > 1e-9 else 1e9
        else:
            p["slope_var"] = 1e9

    global_slope = median([p["slope"] for p in per])
    # Cross-trophy slope signal: observed slope variance minus typical
    # estimation noise. The empirical-Bayes weight per trophy follows.
    if per:
        ms = sum(p["slope"] for p in per) / len(per)
        slope_obs_var = sum((p["slope"] - ms) ** 2 for p in per) / len(per)
    else:
        slope_obs_var = 0.0
    slope_noise = median([p["slope_var"] for p in per]) if per else 1e9
    slope_signal_var = max(slope_obs_var - slope_noise, 0.01)
    for p in per:
        w = slope_signal_var / (slope_signal_var + p["slope_var"])
        p["slope_weight"] = cap(w, 0.15, 0.9)
    g_earn_known = [p["g_earn"] for p in per if p["g_earn"] is not None]
    global_g_earn = median(g_earn_known) if g_earn_known else 0.0
    vols_known = [p["vol"] for p in per if p["vol"] is not None]
    global_vol = median(vols_known) if vols_known else 0.5

    # --- players-tracked growth (recent window), shared by all trophies ----
    pts = [(t, math.log(read["players_tracked"]))
           for read, t in zip(history, times)
           if read.get("players_tracked") and read["players_tracked"] > 0]
    if len(pts) >= 2 and span > 1e-9:
        wt = [t for t, _ in pts if t >= recent_from]
        wp = [v for t, v in pts if t >= recent_from]
        if len(wt) >= 2 and wt[-1] - wt[0] > 1e-9:
            g_players = endpoint(ols_slope(wt, wp),
                                 recent_width, COUNT_TAU_DAYS)
        else:
            g_players = ols_slope([t for t, _ in pts], [v for _, v in pts])
    else:
        g_players = 0.0
    g_players = cap(g_players, 0.0, 1.2)

    # Effective horizons under exponentially decaying trends/growth.
    h_drift = DRIFT_TAU_DAYS * (1.0 - math.exp(-horizon_days / DRIFT_TAU_DAYS)) \
        if horizon_days > 0 else 0.0
    h_count = COUNT_TAU_DAYS * (1.0 - math.exp(-horizon_days / COUNT_TAU_DAYS)) \
        if horizon_days > 0 else 0.0

    rows = []
    for trophy, p in zip(trophies, per):
        base_logit = p["last_raw_logit"] if p["last_raw_logit"] is not None \
            else global_last_logit

        # Drift view: empirical-Bayes-shrunk, capped slope over the decayed
        # horizon. The fixed DRIFT_SHRINK is the weight's ceiling in calm
        # data; noisy slopes are shrunk further toward the global drift.
        w_slope = min(p.get("slope_weight", DRIFT_SHRINK), DRIFT_SHRINK + 0.35)
        slope = cap(w_slope * p["slope"]
                    + (1.0 - w_slope) * global_slope,
                    SLOPE_CAP_DOWN, SLOPE_CAP_UP)
        shift = cap(slope * h_drift, SHIFT_CAP_DOWN, SHIFT_CAP_UP)
        logit_drift = base_logit + shift

        # Count view: grow earners and players separately, take the ratio.
        # Its implied shift from the last rate is capped before blending,
        # so a launch-week catch-up burst in the counts cannot dominate
        # the median (and a dilution plunge cannot either).
        logit_med = logit_drift
        if p["g_earn"] is not None and p["last_players"] \
                and p["last_rate"] is not None:
            w_e = cap(p.get("slope_weight", COUNT_SHRINK), 0.2, 0.85)
            g_e = cap(w_e * p["g_earn"]
                      + (1.0 - w_e) * global_g_earn, -0.3, 2.5)
            earned_last = p["last_rate"] / 100.0 * p["last_players"]
            earned_f = earned_last * math.exp(g_e * h_count)
            players_f = p["last_players"] * math.exp(g_players * h_count)
            if players_f > 0:
                rate_b = clamp_rate(100.0 * earned_f / players_f)
                shift_b = cap(logit(rate_b) - base_logit,
                              VIEW_SHIFT_DOWN, VIEW_SHIFT_UP)
                logit_med = BLEND_DRIFT * logit_drift \
                    + (1.0 - BLEND_DRIFT) * (base_logit + shift_b)

        # Saturation damping: bend a positive total shift as the forecast
        # rate approaches the tier's plausible ceiling share.
        total_shift = logit_med - base_logit
        if total_shift > 0:
            tier_cap = TIER_CAP.get(trophy.get("tier"), 0.9)
            if trophy.get("secret"):
                tier_cap *= SECRET_CAP_MULT
            r_pre = inv_logit(logit_med) / 100.0
            if r_pre > 0.55 * tier_cap:
                damp = cap(1.15 - (r_pre / tier_cap) ** 2, 0.15, 1.0)
                total_shift *= damp
                logit_med = base_logit + total_shift

        # Extreme-rate guard: at >= 99.9% (or <= 0.1%) the clamped logit is
        # nearly unidentified; mark it so the tails below can widen inward.
        extreme = 0.0  # +1 pinned at the top, -1 pinned at the bottom
        if p["last_rate"] is not None and horizon_days > 0.25:
            if p["last_rate"] >= 99.9:
                extreme = 1.0
            elif p["last_rate"] <= 0.1:
                extreme = -1.0

        # Spread: own volatility (shrunk to the global median) scaled by
        # sqrt(horizon), plus trend uncertainty and a small base term.
        vol = p["vol"] if p["vol"] is not None else global_vol
        vol_shrunk = max(0.5 * vol + 0.5 * global_vol, VOL_FLOOR) \
            if span > 0.05 else global_vol
        # The base term and the floor scale with sqrt(horizon): a 90-minute
        # practice horizon honestly needs a far narrower spread than 4.5 days.
        base_unc = BASE_UNC * math.sqrt(min(horizon_days, 1.0))
        sigma_floor = 0.02 + 0.10 * math.sqrt(min(horizon_days, 1.0))
        sigma = math.sqrt((vol_shrunk * math.sqrt(horizon_days)) ** 2
                          + (TREND_UNC * abs(total_shift)) ** 2
                          + base_unc ** 2)
        sigma = cap(sigma, sigma_floor, SIGMA_CAP)

        # Mild asymmetry in the direction of the trend.
        if total_shift > 0.25:
            up_mult, down_mult = 1.10, 0.93
        elif total_shift < -0.25:
            up_mult, down_mult = 0.93, 1.10
        else:
            up_mult = down_mult = 1.0
        # A pinned rate can only really move one way: widen the tail that
        # points back toward the middle (with its own spread floor, since a
        # pinned series shows no volatility of its own), compress the tail
        # at the edge slightly.
        sigma_up, sigma_down = sigma * up_mult, sigma * down_mult
        if extreme > 0:
            sigma_down = max(sigma_down * 1.5, 0.55)
            sigma_up *= 0.85
        elif extreme < 0:
            sigma_up = max(sigma_up * 1.5, 0.55)
            sigma_down *= 0.85

        qs = []
        for z in Z:
            side = sigma_up if z > 0 else (sigma_down if z < 0 else sigma)
            qs.append(round4(inv_logit(logit_med + z * side)))
        # Rounding cannot break monotonicity (inv_logit is increasing and
        # round4 is monotone), but guard cheaply anyway.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
