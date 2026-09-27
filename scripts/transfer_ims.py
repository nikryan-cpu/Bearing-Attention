"""Apply the detectors built on FEMTO to IMS test 2 without retraining.

Each IMS bearing is normalized against its own first snapshots, and thresholds come
from the same false-alarm procedure as on FEMTO, calibrated on the healthy segments
of IMS bearings other than the one being scored. Bearing 1 is the one that failed.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_attention import evaluation, features
from bearing_attention.baselines import ECODDetector, FeatureThreshold
from bearing_attention.chronos_detector import ChronosDetector
from bearing_attention.config import load_config
from bearing_attention.forecaster import ForecastDetector, LevelOrSurprise
from bearing_attention.plotting import AQUA, BLUE, INK_SECONDARY, METHOD_LABELS, MUTED, ORANGE, SHADE, apply_style, plt

log = logging.getLogger("transfer_ims")
ALARM = "#f6c9c9"


def femto_single_axis_healthy(cfg):
    """Healthy FEMTO segments, with the two accelerometer axes as separate
    one-sensor records, so that detectors fit the one-sensor IMS layout."""
    records, _ = features.load_records(cfg["paths"]["processed"] / "features" / "femto.npz")
    per_axis = len(features.feature_names(cfg["features"]["bands_hz"]))
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    healthy = []
    for record in records.values():
        z = features.baseline_zscore(record["features"], n_baseline)
        z = z[: evaluation.healthy_end(len(z), cfg)]
        healthy += [z[:, a:a + per_axis] for a in range(0, z.shape[1], per_axis)]
    return healthy


def detectors_from_femto(cfg, feature_names):
    healthy = femto_single_axis_healthy(cfg)
    rms = FeatureThreshold("rms").fit(healthy, feature_names)
    detectors = {"rms": rms,
                 "kurtosis": FeatureThreshold("kurtosis").fit(healthy, feature_names),
                 "ecod": ECODDetector().fit(healthy, feature_names)}
    for name in cfg["transformer"]["variants"]:
        saved = cfg["paths"]["models"] / f"forecaster_{name}.pt"
        detectors[f"transformer_{name}"] = ForecastDetector.load(saved).for_features(feature_names)
    detectors["rms_or_transformer"] = LevelOrSurprise(
        "rms_or_transformer", rms, detectors["transformer_w128_p16"]).scale_parts(healthy)
    ch = cfg["chronos"]
    detectors["chronos_bolt_small"] = ChronosDetector(
        "chronos_bolt_small", ch["window"], ch["horizon"], ch["model"], ch["batch_size"]).fit(healthy, feature_names)
    return detectors


def evaluate(name, detector, z, failed, cfg):
    spec, rule = cfg["datasets"]["ims"], evaluation.alarm_rule(cfg)
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    rates = cfg["evaluation"]["false_alarm_rates"]
    stop = {b: int(spec["healthy_fraction"] * len(z[b])) for b in z}
    raw = {b: detector.score(z[b]) for b in z}
    others = [b for b in z if b not in failed]

    rows, scores = [], {}
    for bearing in z:
        # thresholds never use the bearing being scored, and never a failed bearing
        calibration = [b for b in others if b != bearing]
        reference = np.concatenate([raw[b][n_baseline:stop[b]] for b in calibration])
        series = [evaluation.process(raw[b][:stop[b]], reference, rule)[n_baseline:] for b in calibration]
        thresholds = [evaluation.calibrate_threshold(series, far, rule) for far in rates]
        score = evaluation.process(raw[bearing], reference, rule)
        scores[bearing] = {"raw": raw[bearing], "score": score, "thresholds": thresholds}
        for far, threshold in zip(rates, thresholds):
            m = evaluation.bearing_metrics(score, threshold, rule, n_baseline, stop[bearing],
                                           spec["snapshot_interval_s"])
            rows.append({"method": name, "false_alarm_rate": far, "bearing": bearing,
                         "failed": bearing in failed, "threshold": threshold,
                         "calibrated_on": " ".join(calibration), **m,
                         "lead_h": m["lead_min"] / 60, "first_alarm_lead_h": m["first_alarm_lead_min"] / 60,
                         "alarm_on_at_end": bool(evaluation.alarm_states(score, threshold, rule)[-1])})
    features.save_records(cfg["paths"]["processed"] / "scores" / f"ims_{name}.npz", scores,
                          false_alarm_rates=[str(f) for f in rates])
    return rows


def summarize(metrics):
    def one(g):
        failed, others = g[g.failed], g[~g.failed]
        return pd.Series({
            "bearing1_detected": bool(failed["detected"].all()),
            "bearing1_lead_h": failed["lead_h"].iloc[0],
            "bearing1_first_alarm_lead_h": failed["first_alarm_lead_h"].iloc[0],
            "false_alarms_healthy_all_bearings": int(g["false_alarms_healthy"].sum()),
            "bearings_2_4_alarm_on_at_end": int(others["alarm_on_at_end"].sum()),
        })
    return metrics.groupby(["method", "false_alarm_rate"], sort=False).apply(one, include_groups=False).reset_index()


def plot(cfg, z, feature_names, methods, rate, out):
    spec, rule = cfg["datasets"]["ims"], evaluation.alarm_rule(cfg)
    days_per_snapshot = spec["snapshot_interval_s"] / 86400
    rates = [float(r) for r in cfg["evaluation"]["false_alarm_rates"]]
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(10, 2.1 * (len(methods) + 1)), sharex=True)
    rms = feature_names.index("ch1_rms")
    for bearing, color in zip(sorted(z), (BLUE, ORANGE, AQUA, MUTED)):
        days = np.arange(len(z[bearing])) * days_per_snapshot
        axes[0].plot(days, np.clip(z[bearing][:, rms], -10, 80), color=color, lw=0.8, label=bearing)
    axes[0].set_ylabel("RMS z-score")
    axes[0].legend(loc="upper left", ncol=4)
    axes[0].set_title("IMS test 2: RMS of all four bearings (bearing 1 failed)", loc="left", fontsize=10)

    for ax, method in zip(axes[1:], methods):
        scores, _ = features.load_records(cfg["paths"]["processed"] / "scores" / f"ims_{method}.npz")
        rec = scores["Bearing1"]
        threshold = rec["thresholds"][rates.index(rate)]
        days = np.arange(len(rec["score"])) * days_per_snapshot
        ax.axvspan(0, int(spec["healthy_fraction"] * len(days)) * days_per_snapshot, color=SHADE, lw=0)
        states = evaluation.alarm_states(rec["score"], threshold, rule)
        ax.fill_between(days, 0, 1, where=states, transform=ax.get_xaxis_transform(), color=ALARM, lw=0)
        ax.plot(days, rec["score"], color=BLUE, lw=0.8)
        ax.axhline(threshold, color=INK_SECONDARY, lw=0.9, ls="--")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(-12, 110)
        ax.set_ylabel("score")
        ax.set_title(f"bearing 1, {METHOD_LABELS[method]} (trained on FEMTO), threshold at {rate:.0%} false alarms",
                     loc="left", fontsize=10)
    axes[-1].set_xlabel("days since start")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # model downloads from Hugging Face
    apply_style()
    cfg = load_config(args.config)
    test = cfg["datasets"]["ims"]["test"]
    records, names = features.load_records(cfg["paths"]["processed"] / "features" / f"ims_{test}.npz")
    feature_names = names["feature_names"]
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    z = {b: features.baseline_zscore(r["features"], n_baseline) for b, r in sorted(records.items())}
    failed = {b for b, r in records.items() if r["end_is_failure"]}

    rows = []
    for name, detector in detectors_from_femto(cfg, feature_names).items():
        log.info("scoring %s", name)
        rows += evaluate(name, detector, z, failed, cfg)

    out_dir = cfg["paths"]["results"] / "ims"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot(cfg, z, feature_names, ["rms", "ecod", "transformer_w128_p16"], 0.01,
         cfg["paths"]["results"] / "figures" / "ims_transfer.png")

    summary["false_alarm_rate"] = summary["false_alarm_rate"].map("{:.1%}".format)
    with pd.option_context("display.width", 180, "display.float_format", "{:.1f}".format):
        print(summary.to_string(index=False))
    print("lead times in hours before the end of the test; one snapshot = 10 min")


if __name__ == "__main__":
    main()
