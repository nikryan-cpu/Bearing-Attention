import numpy as np
import pytest

from bearing_attention import chronos_detector
from bearing_attention.chronos_detector import ChronosDetector


class LastValue:
    """Stand-in for a Chronos-Bolt pipeline: every quantile is the last observed value."""

    def __init__(self):
        self.calls = 0

    def predict(self, inputs, prediction_length):
        self.calls += 1
        return inputs[:, -1:, None].repeat(1, 9, prediction_length)


@pytest.fixture(autouse=True)
def empty_cache():
    chronos_detector._FORECASTS.clear()


def walk(n, channels=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, channels)).cumsum(axis=0)


def test_forecast_layout_matches_the_transformer():
    z = walk(50)
    forecasts = ChronosDetector("c", 16, 4, "fake", pipeline=LastValue()).forecast(z)
    assert forecasts.shape == (50 - 16, 4, 2)
    # row s forecasts from the window z[s : s + 16], so its last value is z[s + 15]
    np.testing.assert_allclose(forecasts[3, :, 1], np.full(4, z[18, 1]), rtol=1e-6)


def test_each_record_is_forecast_once():
    pipeline = LastValue()
    detector = ChronosDetector("c", 16, 4, "fake", batch_size=1000, pipeline=pipeline)
    z = walk(60)
    detector.fit([z[:40], walk(10)], ["a_rms", "a_peak"])  # the second segment is too short and skipped
    calls = pipeline.calls
    detector.forecast(z[:40])
    assert pipeline.calls == calls


def test_score_is_causal_and_flags_a_jump():
    detector = ChronosDetector("c", 16, 4, "fake", pipeline=LastValue()).fit([walk(200, seed=1)], ["a_x", "b_x"])
    record = walk(120, seed=2)
    record[100:] += 50
    score = detector.score(record)
    np.testing.assert_allclose(detector.score(record, stop=90), score[:90])
    assert np.isnan(score[:16]).all()
    assert np.nanmax(score[100:104]) > np.nanmax(score[20:95]) + np.log(10)

