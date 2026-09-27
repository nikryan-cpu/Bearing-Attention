"""Export everything the demo shows into app/data, one compressed .npz per bearing.

Only derived data is written (features, scores, forecasts, attention, coarse spectra):
the datasets may be used freely but not redistributed, so no raw recording ends up
in the app. FEMTO forecasts come from the cross-validation model that never saw the
bearing; IMS uses the model trained on FEMTO, as in the transfer evaluation.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

from bearing_attention import evaluation, features, femto, ims
from bearing_attention.config import REPO_ROOT, load_config
from bearing_attention.forecaster import ForecastDetector

log = logging.getLogger("export_demo")


def binned(freqs, values, edges, reduce):
    """Reduce the columns of `values` into frequency bins given by `edges`."""
    index = np.digitize(freqs, edges) - 1
    out = np.full((len(values), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        cols = index == b
        if cols.any():
            out[:, b] = reduce(values[:, cols], axis=1)
    return out


def spectra(x, fs, cfg):
    """Coarse log power spectrum and envelope spectrum of every snapshot of one axis."""
    demo, feature_cfg = cfg["demo"], cfg["features"]
    psd_edges = np.linspace(0, demo["spectrum_max_hz"], demo["spectrum_bins"] + 1)
    env_edges = np.arange(0, demo["envelope_max_hz"] + demo["envelope_bin_hz"], demo["envelope_bin_hz"])
    psd_rows, env_rows = [], []
    for start in range(0, len(x), 256):
        chunk = x[start:start + 256].astype(np.float64)
        chunk = chunk - chunk.mean(axis=1, keepdims=True)
        freqs, psd = periodogram(chunk, fs=fs, window="hann", axis=1)
        psd_rows.append(np.log10(binned(freqs, psd, psd_edges, np.mean) + 1e-12))
        env_freqs, env = features.envelope_spectrum(features.envelope(chunk, fs, feature_cfg["envelope_band_hz"]), fs)
        # max keeps a narrow defect line visible when IMS's 1 Hz bins are merged into 10 Hz ones
        env_rows.append(binned(env_freqs, env, env_edges, np.max))
    centers = lambda e: (e[:-1] + e[1:]) / 2
    return np.concatenate(psd_rows), centers(psd_edges), np.concatenate(env_rows), centers(env_edges)


def attention_per_patch(attention):
    """(windows, channels, layers, heads, queries, keys) -> attention each key patch receives."""
    return attention.mean(axis=(1, 2, 3, 4))


def export(out, meta, acc_axis, fs, z, feature_names, scores, detector, extra, cfg):
    demo = cfg["demo"]
    rule = evaluation.alarm_rule(cfg)
    first_axis = feature_names[0].split("_", 1)[0]
    channels = [feature_names.index(f"{first_axis}_{c}") for c in demo["forecast_channels"]]
    forecasts, attention = detector.forecast(z, return_attention=True)
    psd, psd_freqs, env, env_freqs = spectra(acc_axis, fs, cfg)

    arrays = {
        "z": z[:, channels].astype(np.float32),
        "rms_g": np.sqrt(np.mean(np.square(acc_axis, dtype=np.float64), axis=1)).astype(np.float32),
        "forecast": forecasts[:, :, channels].astype(np.float16),
        "attention": attention_per_patch(attention).astype(np.float32),
        "psd": psd.astype(np.float16), "psd_freqs": psd_freqs.astype(np.float32),
        "env": env.astype(np.float16), "env_freqs": env_freqs.astype(np.float32),
        **extra,
    }
    for method, rec in scores.items():
        arrays[f"score_{method}"] = rec["score"].astype(np.float32)
        arrays[f"thresholds_{method}"] = np.asarray(rec["thresholds"], dtype=np.float32)
        arrays[f"alarm_{method}"] = np.stack([evaluation.alarm_states(rec["score"], t, rule) for t in rec["thresholds"]])
    meta = {**meta, "methods": list(scores), "false_alarm_rates": cfg["evaluation"]["false_alarm_rates"],
            "forecast_channels": demo["forecast_channels"], "window": detector.window, "horizon": detector.horizon,
            "patch_spans": detector.model.patch_spans(), "n": len(z)}
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    log.info("%s: %.1f MB", out.name, out.stat().st_size / 2**20)


def load_scores(cfg, dataset, methods):
    out = {}
    for method in methods:
        records, _ = features.load_records(cfg["paths"]["processed"] / "scores" / f"{dataset}_{method}.npz")
        out[method] = records
    return out


def export_femto(cfg, out_dir):
    paths, demo = cfg["paths"], cfg["demo"]
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    records, names = features.load_records(paths["processed"] / "features" / "femto.npz")
    scores = load_scores(cfg, "femto", demo["methods"])
    found = femto.find_bearings(paths["raw"] / "femto")
    groups = cfg["evaluation"]["groups"]
    for name in demo["femto_bearings"]:
        subset = next(s for s, b in found if b == name and s in femto.RUN_TO_FAILURE_SUBSETS)
        bearing = femto.load_bearing(found[(subset, name)], paths["processed"] / "femto")
        group = next(i for i, g in enumerate(groups) if name in g)
        detector = ForecastDetector.load(paths["models"] / "cv" / f"transformer_{demo['forecaster']}_group{group}.pt")
        defects = records[name]["defects"]  # horizontal then vertical, BPFO / BPFI / BSF
        meta = {"dataset": "FEMTO", "bearing": name, "subset": subset, "condition": bearing.condition,
                "rpm": bearing.rpm, "load_n": bearing.load_n, "interval_s": femto.SNAPSHOT_INTERVAL_S,
                "fs": cfg["datasets"]["femto"]["sampling_rate_hz"], "baseline": n_baseline,
                "healthy_end": evaluation.healthy_end(len(bearing), cfg), "end_is_failure": bearing.end_is_failure,
                "defect_hz": bearing.defect_hz, "n_elements": femto.GEOMETRY["n_elements"],
                "model_note": f"cross-validation model of group {group}, which never saw this bearing"}
        z = features.baseline_zscore(records[name]["features"], n_baseline)
        extra = {"defects": np.maximum(defects[:, :3], defects[:, 3:]).astype(np.float32)}
        export(out_dir / f"femto_{name}.npz", meta, bearing.acc[:, :, 0], meta["fs"], z, names["feature_names"],
               {m: scores[m][name] for m in demo["methods"]}, detector, extra, cfg)


def export_ims(cfg, out_dir):
    paths, demo, spec = cfg["paths"], cfg["demo"], cfg["datasets"]["ims"]
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    records, names = features.load_records(paths["processed"] / "features" / f"ims_{spec['test']}.npz")
    scores = load_scores(cfg, "ims", demo["methods"])
    detector = ForecastDetector.load(paths["models"] / f"forecaster_{demo['forecaster']}.pt")
    detector = detector.for_features(names["feature_names"])
    for bearing in ims.load_bearings(paths["raw"] / "ims" / spec["test"], paths["processed"] / "ims"):
        meta = {"dataset": "IMS", "bearing": bearing.name, "test": spec["test"], "rpm": ims.SHAFT_RPM,
                "interval_s": ims.SNAPSHOT_INTERVAL_S, "fs": spec["sampling_rate_hz"], "baseline": n_baseline,
                "healthy_end": int(spec["healthy_fraction"] * len(bearing)), "end_is_failure": bearing.end_is_failure,
                "defect_hz": None, "n_elements": None,
                "model_note": "model trained on FEMTO, applied without retraining"}
        z = features.baseline_zscore(records[bearing.name]["features"], n_baseline)
        export(out_dir / f"ims_{bearing.name}.npz", meta, bearing.acc[:, :, 0], meta["fs"], z,
               names["feature_names"], {m: scores[m][bearing.name] for m in demo["methods"]}, detector, {}, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    out_dir = REPO_ROOT / "app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_femto(cfg, out_dir)
    export_ims(cfg, out_dir)
    total = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    log.info("app/data: %.1f MB in total", total / 2**20)


if __name__ == "__main__":
    main()
