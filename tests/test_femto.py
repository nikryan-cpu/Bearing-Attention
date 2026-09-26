import numpy as np
import pytest

from bearing_attention import femto
from bearing_attention.config import load_config


def write_snapshot(path, start, seed, sep=",", exponent_digits=2):
    """Write a fake acc_*.csv: 2560 rows of hour, minute, second, microsecond, horiz, vert."""
    rng = np.random.default_rng(seed)
    hour, rest = divmod(start, 3600)
    minute, second = divmod(rest, 60)
    lines = []
    for i in range(femto.SNAPSHOT_LENGTH):
        micro = f"{i * 39.0625:.4e}"
        if exponent_digits == 3:
            mantissa, exp = micro.split("e")
            micro = f"{mantissa}e{exp[0]}{int(exp[1:]):03d}"
        h, v = rng.normal(size=2)
        lines.append(sep.join([str(int(hour)), str(int(minute)), str(int(second)), micro, f"{h:.3f}", f"{v:.3f}"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture
def raw_dir(tmp_path):
    root = tmp_path / "femto"
    # the third snapshot comes 20 s late and after midnight
    for i, start in enumerate([86380, 86390, 10]):
        write_snapshot(root / "Learning_set/Bearing1_1" / f"acc_{i + 1:05d}.csv", start, seed=i)
    (root / "Learning_set/Bearing1_1/temp_00001.csv").write_text("9,0,0,0,40.1\n")
    for i, start in enumerate([100, 110]):
        write_snapshot(root / "Test_set/Bearing2_3" / f"acc_{i + 1:05d}.csv", start, seed=10 + i,
                       sep=";", exponent_digits=3)
    return root


def test_finds_bearings_and_subsets(raw_dir):
    assert set(femto.find_bearings(raw_dir)) == {("Learning_set", "Bearing1_1"), ("Test_set", "Bearing2_3")}


def test_load_bearing_shapes_and_midnight_wrap(raw_dir):
    b = femto.load_bearing(raw_dir / "Learning_set/Bearing1_1")
    assert b.acc.shape == (3, 2560, 2)
    assert b.acc.dtype == np.float32
    np.testing.assert_allclose(b.t_recorded, [0, 10, 30])
    np.testing.assert_allclose(b.t, [0, 10, 20])
    assert b.off_grid_steps == 1
    assert b.run_to_failure and b.end_is_failure
    assert (b.condition, b.rpm, b.load_n) == (1, 1800, 4000)


def test_semicolons_and_long_exponents(raw_dir):
    b = femto.load_bearing(raw_dir / "Test_set/Bearing2_3")
    np.testing.assert_allclose(b.t_recorded, [0, 10])
    assert not b.run_to_failure and not b.end_is_failure
    assert b.condition == 2
    assert np.isfinite(b.acc).all()


def test_values_match_the_csv(raw_dir):
    start, samples = femto.read_snapshot(raw_dir / "Learning_set/Bearing1_1/acc_00002.csv")
    assert start == 86390
    expected = np.round(np.random.default_rng(1).normal(size=(2560, 2)), 3)
    np.testing.assert_allclose(samples, expected, atol=1e-6)


def test_missing_snapshot_is_an_error(raw_dir):
    (raw_dir / "Learning_set/Bearing1_1/acc_00002.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        femto.load_bearing(raw_dir / "Learning_set/Bearing1_1")


def test_short_snapshot_is_an_error(raw_dir):
    path = raw_dir / "Learning_set/Bearing1_1/acc_00001.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:100]))
    with pytest.raises(ValueError, match="2560x6"):
        femto.load_bearing(raw_dir / "Learning_set/Bearing1_1")


def test_cache_is_used_on_second_load(raw_dir, tmp_path):
    cache = tmp_path / "cache"
    first = femto.load_bearing(raw_dir / "Learning_set/Bearing1_1", cache)
    for f in (raw_dir / "Learning_set/Bearing1_1").glob("acc_*.csv"):
        f.unlink()
    second = femto.load_bearing(raw_dir / "Learning_set/Bearing1_1", cache)
    np.testing.assert_array_equal(first.acc, second.acc)
    np.testing.assert_array_equal(first.t_recorded, second.t_recorded)


def test_load_all_and_summary(raw_dir):
    assert [b.name for b in femto.load_all(raw_dir)] == ["Bearing1_1"]
    table = femto.summary(femto.load_all(raw_dir, subsets=None))
    assert list(table["snapshots"]) == [3, 2]
    assert list(table["off_grid_time_stamps"]) == [1, 0]


def test_ambiguous_end_of_life():
    b = femto.Bearing("Bearing1_4", "Full_Test_Set", np.zeros((2, 2560, 2), np.float32), np.array([0.0, 10.0]))
    assert b.run_to_failure and not b.end_is_failure


REAL = load_config()["paths"]
REAL_FEMTO = REAL["raw"] / "femto"

# Actual RULs of the test bearings in seconds, table 3 of "IEEE PHM 2012 Prognostic
# challenge: Outline, Experiments, Scoring of results, Winners"
OFFICIAL_RUL_S = {
    "Bearing1_3": 5730, "Bearing1_4": 339, "Bearing1_5": 1610, "Bearing1_6": 1460,
    "Bearing1_7": 7570, "Bearing2_3": 7530, "Bearing2_4": 1390, "Bearing2_5": 3090,
    "Bearing2_6": 1290, "Bearing2_7": 580, "Bearing3_3": 820,
}


@pytest.mark.skipif(not REAL_FEMTO.exists(), reason="FEMTO data not downloaded")
def test_real_archive_layout():
    found = femto.find_bearings(REAL_FEMTO)
    learning = {name for subset, name in found if subset == "Learning_set"}
    assert learning == {"Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"}
    assert {name for subset, name in found if subset == "Full_Test_Set"} == set(OFFICIAL_RUL_S)
    assert {name for subset, name in found if subset == "Test_set"} == set(OFFICIAL_RUL_S)


@pytest.mark.skipif(not REAL_FEMTO.exists(), reason="FEMTO data not downloaded")
@pytest.mark.parametrize("name", sorted(OFFICIAL_RUL_S))
def test_full_records_extend_the_truncated_ones(name):
    cache = REAL["processed"] / "femto"
    full = femto.load_bearing(REAL_FEMTO / "Full_Test_Set" / name, cache)
    truncated = femto.load_bearing(REAL_FEMTO / "Test_set" / name, cache)
    np.testing.assert_array_equal(full.acc[: len(truncated)], truncated.acc)
    extra_s = (len(full) - len(truncated)) * femto.SNAPSHOT_INTERVAL_S
    if full.end_is_failure:
        assert extra_s == OFFICIAL_RUL_S[name]
    else:
        assert extra_s != OFFICIAL_RUL_S[name]
