"""Anomaly scores over time for a few FEMTO bearings, with thresholds and alarms."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bearing_attention import evaluation, features
from bearing_attention.config import load_config
from bearing_attention.plotting import BLUE, INK_SECONDARY, METHOD_LABELS, SHADE, apply_style, plt

ALARM = "#f6c9c9"
TICKS = [-10, 0, 5, 10, 20, 50, 100]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["rms", "kurtosis", "ecod"])
    parser.add_argument("--bearings", nargs="+", default=["Bearing1_1", "Bearing2_2", "Bearing2_5", "Bearing3_3"])
    parser.add_argument("--rate", type=float, default=0.01, help="false-alarm rate whose threshold is drawn")
    parser.add_argument("--out", default="femto_scores.png")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    apply_style()
    cfg = load_config(args.config)
    paths, rule = cfg["paths"], evaluation.alarm_rule(cfg)
    hours_per_snapshot = cfg["datasets"]["femto"]["snapshot_interval_s"] / 3600

    fig, axes = plt.subplots(len(args.bearings), len(args.methods), figsize=(4 * len(args.methods), 2.4 * len(args.bearings)),
                             squeeze=False)
    for col, method in enumerate(args.methods):
        scores, names = features.load_records(paths["processed"] / "scores" / f"femto_{method}.npz")
        rate_index = [float(r) for r in names["false_alarm_rates"]].index(args.rate)
        for row, bearing in enumerate(args.bearings):
            ax = axes[row, col]
            rec = scores[bearing]
            score, threshold = rec["score"], rec["thresholds"][rate_index]
            hours = np.arange(len(score)) * hours_per_snapshot
            ax.axvspan(0, evaluation.healthy_end(len(score), cfg) * hours_per_snapshot, color=SHADE, lw=0)
            states = evaluation.alarm_states(score, threshold, rule)
            ax.fill_between(hours, 0, 1, where=states, transform=ax.get_xaxis_transform(), color=ALARM, lw=0)
            ax.plot(hours, score, color=BLUE, lw=0.8)
            ax.axhline(threshold, color=INK_SECONDARY, lw=0.9, ls="--")
            ax.set_yscale("symlog", linthresh=10)
            ax.set_yticks(TICKS, [str(t) for t in TICKS])
            ax.set_ylim(-12, 110)
            if row == 0:
                ax.set_title(METHOD_LABELS.get(method, method), loc="left", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{bearing}\nscore")
            if row == len(args.bearings) - 1:
                ax.set_xlabel("hours since start")

    handles = [Line2D([], [], color=BLUE, label="smoothed score (healthy-spread units)"),
               Line2D([], [], color=INK_SECONDARY, ls="--", label=f"threshold at {args.rate:.1%} false alarms"),
               Patch(color=ALARM, label="alarm on"),
               Patch(color=SHADE, label="healthy segment")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    out = paths["results"] / "figures" / args.out
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
