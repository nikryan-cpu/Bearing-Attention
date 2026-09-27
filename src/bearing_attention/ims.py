"""NASA IMS bearing data (University of Cincinnati): one run-to-failure test of four
bearings on a common shaft."""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SNAPSHOT_LENGTH = 20480
SNAPSHOT_INTERVAL_S = 600.0
SHAFT_RPM = 2000
N_BEARINGS = 4

# From the readme: which bearing failed at the end of each test. Test 1 has two
# accelerometers per bearing (8 channels), tests 2 and 3 have one (4 channels).
FAILED = {"1st_test": {"Bearing3", "Bearing4"}, "2nd_test": {"Bearing1"}, "3rd_test": {"Bearing3"}}


@dataclass
class Bearing:
    name: str
    test: str
    acc: np.ndarray  # (n_snapshots, 20480, axes)
    t_recorded: np.ndarray  # seconds since the first snapshot, from the file names

    @property
    def t(self):
        return np.arange(len(self.acc)) * SNAPSHOT_INTERVAL_S

    @property
    def end_is_failure(self):
        return self.name in FAILED.get(self.test, set())

    @property
    def axes(self):
        return tuple(f"ch{i + 1}" for i in range(self.acc.shape[2]))

    def __len__(self):
        return len(self.acc)


def snapshot_time(path):
    return datetime.strptime(Path(path).name, "%Y.%m.%d.%H.%M.%S")


def read_snapshot(path):
    table = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if table.shape[0] != SNAPSHOT_LENGTH:
        raise ValueError(f"{path}: expected {SNAPSHOT_LENGTH} rows, got {table.shape[0]}")
    return table


def load_test(folder, cache_dir=None):
    """Load every snapshot of one test folder: (acc (n, 20480, channels), seconds since start)."""
    folder = Path(folder)
    cache = Path(cache_dir) / f"ims_{folder.name}.npz" if cache_dir else None
    if cache and cache.exists():
        with np.load(cache) as z:
            return z["acc"], z["t_recorded"]

    files = sorted((p for p in folder.iterdir() if p.is_file()), key=snapshot_time)
    times = np.array([snapshot_time(p).timestamp() for p in files])
    acc = np.stack([read_snapshot(p) for p in files])
    t_recorded = times - times[0]
    steps = np.diff(t_recorded)
    # The readme warns that longer gaps mean the test resumed on the next working
    # day; test 1 also starts with 5-minute steps. Windows are counted in
    # snapshots, so gaps are reported rather than filled in.
    irregular = np.abs(steps - SNAPSHOT_INTERVAL_S) > 60
    if irregular.any():
        log.warning("%s: %d intervals differ from 10 min (largest %.0f min)",
                    folder.name, irregular.sum(), steps.max() / 60)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, acc=acc, t_recorded=t_recorded)
    return acc, t_recorded


def running_snapshots(acc):
    """Number of leading snapshots recorded while the rig was running.

    The last files of a test can be recorded after the motor stopped: every channel
    near zero (test 2 ends with two such files). They are not part of the life.
    """
    rms = np.sqrt(np.stack([np.mean(np.square(s, dtype=np.float64), axis=0) for s in acc]))
    stopped = (rms < 0.1 * np.median(rms, axis=0)).all(axis=1)
    n = len(acc)
    while n and stopped[n - 1]:
        n -= 1
    return n


def load_bearings(folder, cache_dir=None):
    """The four bearings of a test, each with its own accelerometer channels."""
    acc, t_recorded = load_test(folder, cache_dir)
    n = running_snapshots(acc)
    if n < len(acc):
        log.info("%s: dropping %d snapshots recorded after the rig stopped", Path(folder).name, len(acc) - n)
        acc, t_recorded = acc[:n], t_recorded[:n]
    per_bearing = acc.shape[2] // N_BEARINGS
    return [Bearing(f"Bearing{i + 1}", Path(folder).name, acc[:, :, i * per_bearing:(i + 1) * per_bearing], t_recorded)
            for i in range(N_BEARINGS)]
