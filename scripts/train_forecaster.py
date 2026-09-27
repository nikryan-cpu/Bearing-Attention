"""Train every forecaster variant on the healthy segments of all FEMTO bearings.

These models are the ones applied to IMS without retraining. The FEMTO results
come from scripts/evaluate.py, whose models never saw the bearing they score.
"""
import argparse
import logging
import time
from pathlib import Path

from bearing_attention import evaluation, features, forecaster
from bearing_attention.config import load_config

log = logging.getLogger("train_forecaster")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", help="variant names from config.yaml (default: all)")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    records, names = features.load_records(cfg["paths"]["processed"] / "features" / "femto.npz")
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    healthy = []
    for record in records.values():
        z = features.baseline_zscore(record["features"], n_baseline)
        healthy.append(z[: evaluation.healthy_end(len(z), cfg)])

    factories = forecaster.variants(cfg)
    for name in args.variants or cfg["transformer"]["variants"]:
        detector = factories[f"transformer_{name}"]()
        start = time.time()
        detector.fit(healthy, names["feature_names"])
        out = cfg["paths"]["models"] / f"forecaster_{name}.pt"
        detector.save(out)
        n_params = sum(p.numel() for p in detector.model.parameters())
        log.info("%s: %d parameters, %.0f s, loss per epoch %s -> %s", name, n_params, time.time() - start,
                 " ".join(f"{v:.3f}" for v in detector.loss_history), out)


if __name__ == "__main__":
    main()
