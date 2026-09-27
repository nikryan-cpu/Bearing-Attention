"""Evaluation protocol shared by all detectors.

A detector turns z-scored features into one raw anomaly score per snapshot. The
score is expressed in units of its spread on held-out healthy data, clipped and
smoothed; an alarm needs several exceedances in a row. Thresholds are set on
held-out healthy data for a target false-alarm rate, so every method is compared
at the same rate. There are no wear labels, only the end of each test, so the
main result is the lead time: how long before the end of the test the alarm that
never clears again was raised.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bearing_attention.features import baseline_zscore


@dataclass
class AlarmRule:
    smooth: int  # trailing median over this many snapshots
    raise_after: int  # consecutive exceedances that raise the alarm
    clear_after: int  # consecutive normal snapshots that clear it
    clip: float  # cap on the normalized score


def alarm_rule(cfg):
    return AlarmRule(**cfg["alarm"])


def healthy_end(n, cfg):
    return int(cfg["normalization"]["healthy_fraction"] * n)


def normalize(raw, reference):
    """Express raw scores as robust z-scores of their distribution on held-out healthy data."""
    reference = reference[np.isfinite(reference)]
    center = np.median(reference)
    scale = 1.4826 * np.median(np.abs(reference - center))
    return (raw - center) / max(scale, 1e-12)


def trailing_median(x, window):
    """Median of the current and the `window - 1` previous values (NaNs ignored)."""
    padded = np.concatenate([np.full(window - 1, np.nan), x])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    out = np.full(len(x), np.nan)
    has_value = np.isfinite(windows).any(axis=1)
    out[has_value] = np.nanmedian(windows[has_value], axis=1)
    return out


def process(raw, reference, rule):
    # Near the end of life the score grows by orders of magnitude and jumps around.
    # Beyond "far outside normal" those values carry no extra information, so they are
    # capped. The median and the threshold comparison are unaffected by the cap as long
    # as the threshold stays below it, which run_protocol checks.
    return trailing_median(np.clip(normalize(raw, reference), -rule.clip, rule.clip), rule.smooth)


def alarm_states(score, threshold, rule):
    """Per-snapshot alarm state: raised after `raise_after` exceedances in a row,
    cleared after `clear_after` normal snapshots in a row. NaN counts as normal."""
    above = np.nan_to_num(score, nan=-np.inf) > threshold
    states = np.zeros(len(score), dtype=bool)
    on, run_above, run_below = False, 0, 0
    for t, a in enumerate(above):
        run_above, run_below = (run_above + 1, 0) if a else (0, run_below + 1)
        if not on and run_above >= rule.raise_after:
            on = True
        elif on and run_below >= rule.clear_after:
            on = False
        states[t] = on
    return states


def alarm_onsets(states):
    return np.flatnonzero(states & ~np.concatenate([[False], states[:-1]]))


def alarm_fraction(series, threshold, rule):
    """Share of the scored (non-NaN) snapshots that are in alarm."""
    total = sum(np.isfinite(s).sum() for s in series)
    return sum(alarm_states(s, threshold, rule).sum() for s in series) / total


def calibrate_threshold(series, false_alarm_rate, rule):
    """Lowest threshold whose share of alarm snapshots on `series` is <= false_alarm_rate.

    The share can only shrink as the threshold grows, so a binary search over the
    observed score values is enough.
    """
    values = np.unique(np.concatenate([s[np.isfinite(s)] for s in series]))
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if alarm_fraction(series, values[mid], rule) <= false_alarm_rate:
            hi = mid
        else:
            lo = mid + 1
    return float(values[lo])


def bearing_metrics(score, threshold, rule, n_baseline, healthy_stop, interval_s):
    """Alarm statistics of one test bearing whose last snapshot is the failure."""
    states = alarm_states(score, threshold, rule)
    onsets = alarm_onsets(states)
    n = len(score)
    final = onsets[-1] if states[-1] else None
    false_alarms = int(((onsets >= n_baseline) & (onsets < healthy_stop)).sum())
    # an alarm that was already on during the healthy segment is not a detection
    detected = final is not None and final >= healthy_stop
    lead = (n - 1 - final) if detected else np.nan
    # The first alarm after the healthy segment may clear again (vibration of a damaged
    # bearing can calm down for a while), so it is reported next to the sustained one.
    later = onsets[onsets >= healthy_stop]
    first_lead = (n - 1 - later[0]) if len(later) else np.nan
    return {
        "detected": detected,
        "lead_snapshots": lead,
        "lead_min": lead * interval_s / 60,
        "first_alarm_lead_min": first_lead * interval_s / 60,
        "false_alarms_healthy": false_alarms,
        "healthy_snapshots": max(healthy_stop - n_baseline, 0),
        "alarm_on_since_healthy": final is not None and final < healthy_stop,
        "cleared_alarms_after_healthy": int((onsets >= healthy_stop).sum()) - int(detected),
    }


def conditional_auc(scores, wear_fraction, n_baseline, cfg):
    """ROC AUC and average precision under a stated convention, not ground truth:
    the last `wear_fraction` of each life is "worn", the healthy segment after the
    baseline is "healthy", and the middle of the life is left out."""
    y, s = [], []
    for score in scores:
        n = len(score)
        worn = np.arange(n) >= n - max(int(wear_fraction * n), 1)
        healthy = (np.arange(n) >= n_baseline) & (np.arange(n) < healthy_end(n, cfg))
        keep = (worn | healthy) & np.isfinite(score)
        y.append(worn[keep])
        s.append(score[keep])
    y, s = np.concatenate(y), np.concatenate(s)
    return roc_auc_score(y, s), average_precision_score(y, s)


def fold_plan(groups, train_only):
    """For each test group: [(inner training bearings, held-out calibration group), ...]
    and the bearings the final detector is fitted on."""
    plan = []
    for g, test in enumerate(groups):
        pool = [grp for i, grp in enumerate(groups) if i != g]
        inner = [([b for j, grp in enumerate(pool) if j != h for b in grp] + list(train_only), held_out)
                 for h, held_out in enumerate(pool)]
        plan.append((test, inner, [b for grp in pool for b in grp] + list(train_only)))
    return plan


def run_protocol(make_detector, records, feature_names, cfg, log=None):
    """Nested grouped cross-validation of one detector.

    Every group of bearings is tested once, with a detector fitted on the healthy
    segments of the other groups (plus the train-only bearings). Its scores are
    normalized and its thresholds calibrated on healthy data of the other groups,
    each scored by an inner detector that was fitted without that group. No bearing
    is ever used for fitting or calibration while it is being tested.
    """
    ev, rule = cfg["evaluation"], alarm_rule(cfg)
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    z = {b: baseline_zscore(r["features"], n_baseline) for b, r in records.items()}

    def fit(bearings):
        healthy = [z[b][: healthy_end(len(z[b]), cfg)] for b in bearings]
        return make_detector().fit(healthy, feature_names)

    results = {}
    for g, (test, inner, final_train) in enumerate(fold_plan(ev["groups"], ev["train_only"])):
        reference_raw = []
        for inner_train, held_out in inner:
            detector = fit(inner_train)
            for b in held_out:
                stop = healthy_end(len(z[b]), cfg)
                if stop > n_baseline:
                    reference_raw.append(detector.score(z[b], stop=stop))
        reference = np.concatenate([raw[n_baseline:] for raw in reference_raw])
        calibration = [process(raw, reference, rule)[n_baseline:] for raw in reference_raw]
        thresholds = {far: calibrate_threshold(calibration, far, rule) for far in ev["false_alarm_rates"]}
        if log and max(thresholds.values()) >= rule.clip:
            log(f"group {g}: a threshold reached the clip value {rule.clip}, no alarm is possible at that rate")

        detector = fit(final_train)
        for b in test:
            raw = detector.score(z[b])
            results[b] = {"raw": raw, "score": process(raw, reference, rule),
                          "thresholds": thresholds, "group": g, "detector": detector}
        if log:
            log(f"group {g}: tested {', '.join(test)}; thresholds "
                + ", ".join(f"{far:.1%}: {t:.2f}" for far, t in thresholds.items()))
    return results
