"""FEMTO/PRONOSTIA bearing records (IEEE PHM 2012 challenge data)."""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_attention.features import defect_frequencies

log = logging.getLogger(__name__)

SNAPSHOT_LENGTH = 2560
SNAPSHOT_INTERVAL_S = 10.0

# Shaft speed (rpm) and radial load (N) per operating condition, Nectoux et al. 2012
CONDITIONS = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}

# Appendix A.1 of the challenge outline document: 13 rolling elements of 3.5 mm on a
# 25.6 mm mean diameter. No contact angle is given, so 0 deg is used; at the 10 Hz
# resolution of a 0.1 s snapshot even 15 deg would shift BPFO by less than 1 Hz.
GEOMETRY = {"n_elements": 13, "element_d": 3.5, "pitch_d": 25.6}
AXES = ("horiz", "vert")

# Test_set holds the competition records, cut off some time before failure.
# Full_Test_Set (Validation_Set.zip in the NASA archive) has the same records up to the end.
RUN_TO_FAILURE_SUBSETS = {"Learning_set", "Full_Test_Set"}
TRUNCATED_SUBSETS = {"Test_set"}

# For ten of the eleven test bearings the full record ends exactly the organizers'
# actual RUL after the truncation point. Bearing1_4 runs on for 2890 s instead of
# the published 339 s, so it is unclear where its life ends.
AMBIGUOUS_END_OF_LIFE = {"Bearing1_4"}

_ACC_FILE = re.compile(r"acc_(\d+)\.csv$")


@dataclass
class Bearing:
    name: str
    subset: str
    acc: np.ndarray  # (n_snapshots, 2560, 2): horizontal, vertical
    t_recorded: np.ndarray  # seconds since the first snapshot, from the time stamps in the files

    @property
    def t(self):
        # Snapshots are taken every 10 s and the files are numbered without gaps.
        # A few time stamps are corrupted (Bearing1_1, files 2121-2122, jump to
        # another time of day and back), so the file order is the reliable clock.
        return np.arange(len(self.acc)) * SNAPSHOT_INTERVAL_S

    @property
    def off_grid_steps(self):
        return int((np.abs(np.diff(self.t_recorded) - SNAPSHOT_INTERVAL_S) > 1.0).sum())

    @property
    def run_to_failure(self):
        return self.subset in RUN_TO_FAILURE_SUBSETS

    @property
    def end_is_failure(self):
        """True if the last snapshot can be taken as the moment of failure."""
        return self.run_to_failure and self.name not in AMBIGUOUS_END_OF_LIFE

    @property
    def condition(self):
        return int(self.name.removeprefix("Bearing").split("_")[0])

    @property
    def rpm(self):
        return CONDITIONS[self.condition][0]

    @property
    def load_n(self):
        return CONDITIONS[self.condition][1]

    @property
    def defect_hz(self):
        return defect_frequencies(self.rpm / 60, **GEOMETRY)

    def __len__(self):
        return len(self.acc)


def find_bearings(raw_dir):
    """Return {(subset, name): folder} for every bearing folder below `raw_dir`."""
    found = {}
    for folder in sorted(Path(raw_dir).rglob("Bearing*_*")):
        if not folder.is_dir():
            continue
        subset = folder.parent.name
        if subset not in RUN_TO_FAILURE_SUBSETS | TRUNCATED_SUBSETS:
            raise ValueError(f"unknown FEMTO subset folder: {folder.parent}")
        found[(subset, folder.name)] = folder
    return found


def read_snapshot(path):
    """Read one acc_*.csv file: (seconds of day at its first sample, samples of shape (2560, 2))."""
    with open(path, encoding="ascii") as f:
        lines = f.read().splitlines()
    sep = ";" if ";" in lines[0] else ","
    table = np.loadtxt(lines, delimiter=sep, ndmin=2)
    if table.shape != (SNAPSHOT_LENGTH, 6):
        raise ValueError(f"{path}: expected {SNAPSHOT_LENGTH}x6 values, got {table.shape}")
    hour, minute, second, microsecond = table[0, :4]
    return hour * 3600 + minute * 60 + second + microsecond * 1e-6, table[:, 4:6].astype(np.float32)


def elapsed_seconds(seconds_of_day):
    steps = np.diff(np.asarray(seconds_of_day, dtype=float))
    # the files only store the time of day, so a test running past midnight wraps around
    steps[steps < 0] += 86400
    return np.concatenate([[0.0], np.cumsum(steps)])


def load_bearing(folder, cache_dir=None):
    """Load all snapshots of one bearing, using an .npz cache in `cache_dir` if given."""
    folder = Path(folder)
    subset, name = folder.parent.name, folder.name
    cache = Path(cache_dir) / f"{subset}_{name}.npz" if cache_dir else None
    if cache and cache.exists():
        with np.load(cache) as z:
            return Bearing(name, subset, z["acc"], z["t_recorded"])

    files = sorted(folder.glob("acc_*.csv"), key=lambda p: int(_ACC_FILE.search(p.name).group(1)))
    indices = [int(_ACC_FILE.search(p.name).group(1)) for p in files]
    if indices != list(range(1, len(files) + 1)):
        missing = sorted(set(range(1, max(indices) + 1)) - set(indices))
        raise ValueError(f"{folder}: snapshot files missing, e.g. {missing[:5]}")

    starts, snapshots = zip(*(read_snapshot(p) for p in files))
    bearing = Bearing(name, subset, np.stack(snapshots), elapsed_seconds(starts))
    if bearing.off_grid_steps:
        log.warning("%s/%s: %d time stamps are off the %.0f s grid, using the file order",
                    subset, name, bearing.off_grid_steps, SNAPSHOT_INTERVAL_S)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, acc=bearing.acc, t_recorded=bearing.t_recorded)
    return bearing


def load_all(raw_dir, cache_dir=None, subsets=("Learning_set", "Full_Test_Set")):
    """Load every bearing of the given subsets (all subsets if `subsets` is None)."""
    bearings = []
    for (subset, name), folder in find_bearings(raw_dir).items():
        if subsets is not None and subset not in subsets:
            continue
        log.info("loading %s/%s", subset, name)
        bearings.append(load_bearing(folder, cache_dir))
    return bearings


def summary(bearings):
    return pd.DataFrame([{
        "bearing": b.name,
        "subset": b.subset,
        "condition": b.condition,
        "rpm": b.rpm,
        "load_n": b.load_n,
        "snapshots": len(b),
        "duration_h": round(b.t[-1] / 3600, 2),
        "off_grid_time_stamps": b.off_grid_steps,
        "end_is_failure": b.end_is_failure,
    } for b in bearings])
