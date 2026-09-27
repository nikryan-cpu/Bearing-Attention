"""Compute per-snapshot features for every FEMTO bearing recorded up to the end of its test."""
import argparse
import logging
from pathlib import Path

import numpy as np

from bearing_attention import features, femto
from bearing_attention.config import load_config

log = logging.getLogger("build_features")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    paths, feature_cfg = cfg["paths"], cfg["features"]
    fs = cfg["datasets"]["femto"]["sampling_rate_hz"]

    records = {}
    for b in femto.load_all(paths["raw"] / "femto", paths["processed"] / "femto"):
        matrix, names = features.bearing_features(b.acc, fs, feature_cfg, femto.AXES)
        defects = np.concatenate(
            [features.defect_ratios(b.acc[:, :, i], fs, feature_cfg, b.defect_hz) for i in range(len(femto.AXES))],
            axis=1,
        )
        records[b.name] = {"features": matrix, "defects": defects, "end_is_failure": b.end_is_failure}
        log.info("%s: %d snapshots x %d features", b.name, *matrix.shape)

    defect_names = [f"{axis}_env_{d}" for axis in femto.AXES for d in features.DEFECTS]
    out = paths["processed"] / "features" / "femto.npz"
    features.save_records(out, records, feature_names=names, defect_names=defect_names)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
