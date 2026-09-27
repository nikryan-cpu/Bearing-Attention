"""Run the evaluation protocol on FEMTO for the chosen detectors and write the metrics."""
import argparse
import logging
from pathlib import Path

import pandas as pd

from bearing_attention import baselines, evaluation, features, forecaster
from bearing_attention.config import load_config

log = logging.getLogger("evaluate")


def evaluate(name, make_detector, records, feature_names, cfg):
    results = evaluation.run_protocol(make_detector, records, feature_names, cfg, log=log.info)
    rule = evaluation.alarm_rule(cfg)
    n_baseline = cfg["normalization"]["baseline_snapshots"]
    interval_s = cfg["datasets"]["femto"]["snapshot_interval_s"]
    rates = cfg["evaluation"]["false_alarm_rates"]

    rows = []
    for bearing, res in sorted(results.items()):
        n = len(res["score"])
        for far in rates:
            metrics = evaluation.bearing_metrics(res["score"], res["thresholds"][far], rule, n_baseline,
                                                 evaluation.healthy_end(n, cfg), interval_s)
            rows.append({"method": name, "false_alarm_rate": far, "bearing": bearing, "group": res["group"],
                         "threshold": res["thresholds"][far], "snapshots": n, **metrics})

    auc, ap = evaluation.conditional_auc([r["score"] for r in results.values()],
                                         cfg["evaluation"]["wear_fraction"], n_baseline, cfg)
    scores = {b: {"raw": r["raw"], "score": r["score"], "thresholds": [r["thresholds"][f] for f in rates],
                  "group": r["group"]} for b, r in results.items()}
    features.save_records(cfg["paths"]["processed"] / "scores" / f"femto_{name}.npz", scores,
                          false_alarm_rates=[str(f) for f in rates])
    # the forecasters of each test group are kept for the demo (forecasts, attention)
    for r in results.values():
        if hasattr(r["detector"], "save"):
            r["detector"].save(cfg["paths"]["models"] / "cv" / f"{name}_group{r['group']}.pt")
    metrics = pd.DataFrame(rows)
    metrics["conditional_roc_auc"] = auc
    metrics["conditional_average_precision"] = ap
    return metrics


def summarize(metrics):
    return (metrics.groupby(["method", "false_alarm_rate"], sort=False)
            .apply(lambda g: pd.Series({
                "detected": f"{int(g['detected'].sum())}/{len(g)}",
                "median_lead_min": g["lead_min"].median(),
                "mean_lead_min": g["lead_min"].mean(),
                "median_first_alarm_lead_min": g["first_alarm_lead_min"].median(),
                "cleared_alarms": int(g["cleared_alarms_after_healthy"].sum()),
                "false_alarms_healthy": int(g["false_alarms_healthy"].sum()),
                "alarm_on_since_healthy": int(g["alarm_on_since_healthy"].sum()),
                "cond_roc_auc": g["conditional_roc_auc"].iloc[0],
                "cond_avg_precision": g["conditional_average_precision"].iloc[0],
            }), include_groups=False)
            .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", help="detectors to run (default: all)")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    detectors = {**baselines.DETECTORS, **forecaster.variants(cfg)}
    unknown = set(args.methods) - set(detectors)
    if unknown:
        parser.error(f"unknown methods {sorted(unknown)}; choose from {sorted(detectors)}")
    records, name_lists = features.load_records(cfg["paths"]["processed"] / "features" / "femto.npz")
    out_dir = cfg["paths"]["results"] / "femto"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.methods or list(detectors):
        log.info("evaluating %s", name)
        evaluate(name, detectors[name], records, name_lists["feature_names"], cfg).to_csv(
            out_dir / f"metrics_{name}.csv", index=False)

    # the summary covers every method evaluated so far, not only this run
    metrics = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("metrics_*.csv"))])
    summary = summarize(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summary["false_alarm_rate"] = summary["false_alarm_rate"].map("{:.1%}".format)
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    print("lead: from the alarm that stays on until failure (medians/means over detected bearings); "
          "first alarm: the first one after the healthy segment, even if it clears again.\n"
          "ROC/PR use the convention 'last 10 % of life = worn, healthy segment = healthy'.")


if __name__ == "__main__":
    main()
