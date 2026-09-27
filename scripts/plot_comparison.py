"""Compare all evaluated detectors on FEMTO: summary figure, per-bearing lead times,
and a head-to-head table of the transformer against each baseline."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_attention.config import load_config
from bearing_attention.plotting import AQUA, BLUE, INK_SECONDARY, METHOD_LABELS, ORANGE, apply_style, plt


def per_method(metrics, rate):
    rows = metrics[metrics.false_alarm_rate == rate]
    return rows.groupby("method").agg(detected=("detected", "sum"), lead=("lead_min", "median"),
                                      cleared=("cleared_alarms_after_healthy", "sum"))


def summary_figure(metrics, rates, out):
    methods = [m for m in METHOD_LABELS if m in set(metrics.method)]
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    panels = [("detected", "bearings detected (of 16)"), ("lead", "median lead time (min)"),
              ("cleared", "alarms that cleared again")]
    for (column, title), ax in zip(panels, axes):
        for rate, color, offset in zip(rates, (BLUE, ORANGE), (0.12, -0.12)):
            values = per_method(metrics, rate).reindex(methods)[column]
            ax.scatter(values, y + offset, color=color, s=30, zorder=3, label=f"{rate:.0%} false alarms")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [METHOD_LABELS[m] for m in methods])
    axes[0].set_xlim(-0.5, 16.5)
    axes[0].set_xticks([0, 4, 8, 12, 16])
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2)
    fig.suptitle("FEMTO, 16 bearings, all methods at the same false-alarm rates", x=0.01, y=0.99,
                 ha="left", fontsize=11)
    fig.savefig(out)
    plt.close(fig)


def lead_per_bearing(metrics, rate, methods, out):
    rows = metrics[(metrics.false_alarm_rate == rate) & metrics.method.isin(methods)]
    table = rows.pivot(index="bearing", columns="method", values="lead_min")
    bearings = sorted(table.index)
    x = np.arange(len(bearings))
    missed_y = 0.1
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for method, color, offset in zip(methods, (BLUE, ORANGE, AQUA), (-0.2, 0, 0.2)):
        lead = table.reindex(bearings)[method].to_numpy()
        found = np.isfinite(lead)
        ax.scatter(x[found] + offset, lead[found], color=color, s=28, zorder=3, label=METHOD_LABELS[method])
        ax.scatter(x[~found] + offset, np.full((~found).sum(), missed_y), color=color, marker="x", s=28, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(missed_y * 0.7, 600)
    ticks = [missed_y, 1, 10, 100]
    ax.set_yticks(ticks, ["missed", "1", "10", "100"])
    ax.axhline(0.3, color=INK_SECONDARY, lw=0.6)
    ax.set_xticks(x, [b.removeprefix("Bearing") for b in bearings])
    ax.set_xlabel("bearing")
    ax.set_ylabel("lead time before failure (min)")
    ax.set_title(f"Lead time of the alarm that stays on until failure, {rate:.0%} false alarms", loc="left",
                 fontsize=10, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(out)
    plt.close(fig)


def head_to_head(metrics, rates, challenger, others):
    rows = []
    for rate in rates:
        sub = metrics[metrics.false_alarm_rate == rate].pivot(index="bearing", columns="method", values="lead_min")
        for other in others:
            a, b = sub[challenger], sub[other]
            # a missed detection counts as later than any detection
            a_, b_ = a.fillna(-1), b.fillna(-1)
            rows.append({"false_alarm_rate": rate, "against": other,
                         "transformer_earlier": int((a_ > b_ + 0.01).sum()),
                         "same_or_both_missed": int((abs(a_ - b_) <= 0.01).sum()),
                         "transformer_later": int((a_ < b_ - 0.01).sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    apply_style()
    cfg = load_config(args.config)
    results = cfg["paths"]["results"]
    metrics = pd.concat([pd.read_csv(p) for p in sorted((results / "femto").glob("metrics_*.csv"))])
    rates = [0.01, 0.05]
    figures = results / "figures"
    summary_figure(metrics, rates, figures / "femto_comparison.png")
    lead_per_bearing(metrics, 0.05, ["rms", "ecod", "transformer_w128_p16"], figures / "femto_lead_per_bearing.png")

    table = head_to_head(metrics, rates, "transformer_w128_p16", ["rms", "kurtosis", "ecod"])
    table.to_csv(results / "femto" / "head_to_head.csv", index=False)
    print(table.to_string(index=False))
    print(f"figures written to {figures}")


if __name__ == "__main__":
    main()
