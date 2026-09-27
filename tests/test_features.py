import numpy as np
import pytest

from bearing_attention import features, femto
from bearing_attention.config import load_config

CFG = load_config()["features"]
NAMES = features.feature_names(CFG["bands_hz"])


def sine(freq, amplitude, fs, seconds, n=1):
    t = np.arange(int(fs * seconds)) / fs
    return np.tile(amplitude * np.sin(2 * np.pi * freq * t), (n, 1))


def impacts(rate_hz, fs=25600, seconds=0.1, resonance_hz=4000, seed=0):
    """Decaying 4 kHz rings struck `rate_hz` times per second, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    x = 0.1 * rng.normal(size=t.size)
    for t0 in np.arange(0, seconds, 1 / rate_hz):
        after = t >= t0
        dt = t[after] - t0
        x[after] += np.exp(-dt / 3e-4) * np.sin(2 * np.pi * resonance_hz * dt)
    return x[None, :]


def test_defect_frequencies_femto_condition_1():
    f = features.defect_frequencies(1800 / 60, **femto.GEOMETRY)
    assert f["bpfo"] == pytest.approx(168.34, abs=0.01)
    assert f["bpfi"] == pytest.approx(221.66, abs=0.01)
    assert f["bsf"] == pytest.approx(107.66, abs=0.01)
    assert f["ftf"] == pytest.approx(12.95, abs=0.01)


def test_time_features_of_a_sine():
    row = features.snapshot_features(sine(150, 2.0, 25600, 0.1), 25600, CFG)[0]
    values = dict(zip(NAMES, row))
    assert values["rms"] == pytest.approx(2 / np.sqrt(2), rel=1e-3)
    assert values["crest"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert values["kurtosis"] == pytest.approx(1.5, rel=1e-2)


@pytest.mark.parametrize("fs, seconds", [(25600, 0.1), (20000, 1.024)])
def test_band_power_is_in_physical_units(fs, seconds):
    row = features.snapshot_features(sine(1234, 2.0, fs, seconds), fs, CFG)[0]
    bands = {n: v for n, v in zip(NAMES, row) if n.startswith("band_")}
    assert bands["band_1000_2000"] == pytest.approx(2.0, rel=0.01)
    assert sum(bands.values()) == pytest.approx(2.0, rel=0.01)


def test_envelope_features_react_to_periodic_impacts():
    rng = np.random.default_rng(1)
    noise = 0.1 * rng.normal(size=(1, 2560))
    hit = dict(zip(NAMES, features.snapshot_features(impacts(150), 25600, CFG)[0]))
    calm = dict(zip(NAMES, features.snapshot_features(noise, 25600, CFG)[0]))
    assert hit["env_peakiness"] > 3 * calm["env_peakiness"]
    assert hit["env_modulation"] > calm["env_modulation"]

    ratios = features.defect_ratios(impacts(150), 25600, CFG, {"bpfo": 150, "bpfi": 215, "bsf": 110})[0]
    bpfo, bpfi, bsf = ratios
    assert bpfo > 5
    assert bpfo > 3 * max(bpfi, bsf)


def test_bearing_features_names_and_shape():
    acc = np.random.default_rng(2).normal(size=(4, 2560, 2))
    matrix, names = features.bearing_features(acc, 25600, CFG, femto.AXES)
    assert matrix.shape == (4, 2 * len(NAMES))
    assert names[0] == "horiz_rms" and names[len(NAMES)] == "vert_rms"
    assert np.all(matrix > 0)


def test_baseline_zscore():
    rng = np.random.default_rng(3)
    base = np.exp(rng.normal(0, 0.1, size=(1000, 1)))
    z = features.baseline_zscore(np.vstack([base, 2 * base[:1]]), n_baseline=1000)
    assert abs(np.median(z[:1000])) < 0.05
    assert np.median(np.abs(z[:1000])) == pytest.approx(0.6745, abs=0.05)
    assert z[-1, 0] == pytest.approx((np.log(2 * base[0, 0]) - np.median(np.log(base))) / 0.1, rel=0.1)


def test_constant_baseline_uses_the_minimum_scale():
    x = np.array([[1.0]] * 10 + [[2.0]])
    z = features.baseline_zscore(x, n_baseline=10)
    assert z[-1, 0] == pytest.approx(np.log(2) / features.MIN_LOG_SCALE)


def test_records_round_trip(tmp_path):
    records = {"Bearing1_1": {"features": np.ones((3, 2)), "end_is_failure": True}}
    path = tmp_path / "f.npz"
    features.save_records(path, records, feature_names=["a", "b"])
    loaded, names = features.load_records(path)
    np.testing.assert_array_equal(loaded["Bearing1_1"]["features"], np.ones((3, 2)))
    assert loaded["Bearing1_1"]["end_is_failure"] is True
    assert names == {"feature_names": ["a", "b"]}
