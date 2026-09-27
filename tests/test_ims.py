import numpy as np
import pytest

from bearing_attention import ims
from bearing_attention.config import load_config


def write_test(folder, names, channels=4, stopped_last=0, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    for i, name in enumerate(names):
        scale = 0.001 if i >= len(names) - stopped_last else 0.1
        data = scale * rng.normal(size=(ims.SNAPSHOT_LENGTH, channels))
        np.savetxt(folder / name, data, fmt="%.4f", delimiter="\t")


NAMES = ["2004.02.12.10.32.39", "2004.02.12.10.42.39", "2004.02.12.10.52.39", "2004.02.12.11.02.39"]


def test_four_channels_are_four_bearings(tmp_path):
    write_test(tmp_path / "2nd_test", NAMES)
    bearings = ims.load_bearings(tmp_path / "2nd_test")
    assert [b.name for b in bearings] == ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]
    assert bearings[0].acc.shape == (4, ims.SNAPSHOT_LENGTH, 1)
    np.testing.assert_allclose(bearings[0].t_recorded, [0, 600, 1200, 1800])
    assert bearings[0].end_is_failure and not bearings[1].end_is_failure


def test_eight_channels_give_two_axes_per_bearing(tmp_path):
    write_test(tmp_path / "1st_test", NAMES, channels=8)
    bearings = ims.load_bearings(tmp_path / "1st_test")
    assert bearings[2].acc.shape[2] == 2 and bearings[2].axes == ("ch1", "ch2")
    assert [b.end_is_failure for b in bearings] == [False, False, True, True]


def test_snapshots_after_the_rig_stopped_are_dropped(tmp_path):
    write_test(tmp_path / "2nd_test", NAMES, stopped_last=2)
    assert len(ims.load_bearings(tmp_path / "2nd_test")[0]) == 2


def test_gap_in_recording_is_reported(tmp_path, caplog):
    write_test(tmp_path / "3rd_test", NAMES[:3] + ["2004.02.13.08.00.00"])
    ims.load_test(tmp_path / "3rd_test")
    assert "differ from 10 min" in caplog.text


def test_cache_round_trip(tmp_path):
    write_test(tmp_path / "2nd_test", NAMES[:2])
    first = ims.load_test(tmp_path / "2nd_test", tmp_path / "cache")
    second = ims.load_test(tmp_path / "2nd_test", tmp_path / "cache")
    np.testing.assert_array_equal(first[0], second[0])


REAL = load_config()["paths"]["raw"] / "ims" / "2nd_test"


@pytest.mark.skipif(not REAL.exists(), reason="IMS test 2 not downloaded")
def test_real_second_test():
    acc, t = ims.load_test(REAL, load_config()["paths"]["processed"] / "ims")
    assert acc.shape == (984, ims.SNAPSHOT_LENGTH, 4)
    assert np.all(np.diff(t) == ims.SNAPSHOT_INTERVAL_S)
    assert ims.running_snapshots(acc) == 982
