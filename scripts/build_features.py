"""Compute per-snapshot features for every FEMTO bearing recorded up to the end of its
test, or for the four bearings of the configured IMS test."""
import argparse
import logging
from pathlib import Path

import numpy as np

from bearing_attention import features, femto, ims
from bearing_attention.config import load_config

log = logging.getLogger("build_features")


def build_femto(cfg):
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
    return out


def build_ims(cfg):
    # no defect-frequency features: the IMS readme gives no bearing geometry
    paths, feature_cfg = cfg["paths"], cfg["features"]
    spec = cfg["datasets"]["ims"]
    records = {}
    for b in ims.load_bearings(paths["raw"] / "ims" / spec["test"], paths["processed"] / "ims"):
        matrix, names = features.bearing_features(b.acc, spec["sampling_rate_hz"], feature_cfg, b.axes)
        records[b.name] = {"features": matrix, "end_is_failure": b.end_is_failure}
        log.info("%s: %d snapshots x %d features", b.name, *matrix.shape)
    out = paths["processed"] / "features" / f"ims_{spec['test']}.npz"
    features.save_records(out, records, feature_names=names)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="femto", choices=["femto", "ims"])
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    out = build_femto(cfg) if args.dataset == "femto" else build_ims(cfg)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
