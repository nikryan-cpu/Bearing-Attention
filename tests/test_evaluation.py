import numpy as np
import pytest

from bearing_attention import evaluation as ev
from bearing_attention.config import load_config

RULE = ev.AlarmRule(smooth=1, raise_after=3, clear_after=2, clip=100)


def test_alarm_needs_consecutive_exceedances_and_clears():
    score = np.array([5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 0, 5])
    states = ev.alarm_states(score, threshold=1, rule=RULE)
    assert states.astype(int).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert ev.alarm_onsets(states).tolist() == [5]


def test_nan_counts_as_normal():
    states = ev.alarm_states(np.array([5, np.nan, 5, 5, 5]), threshold=1, rule=RULE)
    assert states.tolist() == [False, False, False, False, True]


def test_trailing_median_uses_only_the_past():
    x = np.array([0.0, 0, 0, 10, 0, 0])
    y = x.copy()
    y[5:] = 100
    assert ev.trailing_median(x, 3)[:5].tolist() == ev.trailing_median(y, 3)[:5].tolist()
    # a single spike does not survive a median of three
    assert ev.trailing_median(x, 3).max() == 0


def test_threshold_is_the_lowest_meeting_the_rate():
    rng = np.random.default_rng(0)
    series = [rng.normal(size=500) for _ in range(4)]
    thr = ev.calibrate_threshold(series, 0.01, RULE)
    assert ev.alarm_fraction(series, thr, RULE) <= 0.01
    values = np.unique(np.concatenate(series))
    below = values[np.searchsorted(values, thr) - 1]
    assert ev.alarm_fraction(series, below, RULE) > 0.01


def test_bearing_metrics():
    score = np.zeros(100)
    score[12:15] = 5  # false alarm inside the healthy segment [10, 20)
    score[40:44] = 5  # an alarm that clears again
    score[90:] = 5  # final alarm, raised at index 92
    m = ev.bearing_metrics(score, 1, RULE, n_baseline=10, healthy_stop=20, interval_s=10)
    assert m["detected"] and m["lead_snapshots"] == 7 and m["lead_min"] == pytest.approx(70 / 60)
    assert m["first_alarm_lead_min"] == pytest.approx((99 - 42) * 10 / 60)
    assert m["false_alarms_healthy"] == 1
    assert m["cleared_alarms_after_healthy"] == 1
    assert not m["alarm_on_since_healthy"]


def test_alarm_that_never_clears_after_a_false_alarm_is_not_a_detection():
    score = np.zeros(100)
    score[12:] = 5
    m = ev.bearing_metrics(score, 1, RULE, n_baseline=10, healthy_stop=20, interval_s=10)
    assert not m["detected"] and np.isnan(m["lead_snapshots"])
    assert m["alarm_on_since_healthy"] and m["false_alarms_healthy"] == 1


def test_no_bearing_is_fitted_or_calibrated_while_tested():
    cfg = load_config()["evaluation"]
    plan = ev.fold_plan(cfg["groups"], cfg["train_only"])
    tested = [b for test, _, _ in plan for b in test]
    assert sorted(tested) == sorted(b for grp in cfg["groups"] for b in grp)
    for test, inner, final_train in plan:
        assert not set(test) & set(final_train)
        for inner_train, held_out in inner:
            assert not set(test) & (set(inner_train) | set(held_out))
            assert not set(held_out) & set(inner_train)
    assert all(b in final for _, _, final in plan for b in cfg["train_only"])


def test_conditional_auc_of_a_perfect_score():
    cfg = load_config()
    n = 1000
    score = (np.arange(n) >= 900).astype(float)
    auc, ap = ev.conditional_auc([score], 0.1, cfg["normalization"]["baseline_snapshots"], cfg)
    assert auc == 1 and ap == 1


def test_alarm_share_ignores_snapshots_without_a_score():
    series = [np.array([np.nan] * 6 + [5, 5, 5, 0])]
    # raised at the third exceedance, still on at the next normal snapshot
    assert ev.alarm_fraction(series, 1, RULE) == pytest.approx(2 / 4)
