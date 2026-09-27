"""Forecast vs reality, and where the model looked, for one FEMTO bearing.

Uses the cross-validation model of the group in which the bearing was tested,
so the model never saw this bearing.
"""
import argparse
from pathlib import Path

import numpy as np

from bearing_attention import features
from bearing_attention.config import load_config
from bearing_attention.forecaster import ForecastDetector
from bearing_attention.plotting import BLUE, INK_SECONDARY, MUTED, ORANGE, apply_style, plt


def attention_per_patch(attention):
    """Average attention each past patch receives: (channels, layers, heads, q, k) -> (k,)."""
    return attention.mean(axis=(0, 1, 2, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bearing", default="Bearing1_1")
    parser.add_argument("--variant", default="w128_p16")
    parser.add_argument("--channels", nargs="+", default=["horiz_rms", "horiz_band_500_1000", "vert_kurtosis"])
    parser.add_argument("--at", nargs=2, type=float, default=[0.4, 0.99],
                        help="moments to show, as fractions of the life")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    apply_style()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    group = next(i for i, g in enumerate(cfg["evaluation"]["groups"]) if args.bearing in g)
    detector = ForecastDetector.load(paths["models"] / "cv" / f"transformer_{args.variant}_group{group}.pt")
    records, names = features.load_records(paths["processed"] / "features" / "femto.npz")
    z = features.baseline_zscore(records[args.bearing]["features"], cfg["normalization"]["baseline_snapshots"])
    columns = [names["feature_names"].index(c) for c in args.channels]
    forecasts, attention = detector.forecast(z, return_attention=True)
    minutes = cfg["datasets"]["femto"]["snapshot_interval_s"] / 60
    w, h = detector.window, detector.horizon

    fig, axes = plt.subplots(2, len(columns) + 1, figsize=(3.2 * (len(columns) + 1), 5.6),
                             gridspec_kw={"width_ratios": [1] * len(columns) + [0.8]})
    for row, fraction in enumerate(args.at):
        # row s of the forecasts starts right after the window z[s : s + w]
        s = min(int(fraction * len(z)) - w, len(forecasts) - 1 - h)
        now = s + w
        past_t = (np.arange(s, now) - now) * minutes
        future_t = (np.arange(now, now + h) - now) * minutes
        for col, c in enumerate(columns):
            ax = axes[row, col]
            ax.plot(past_t, z[s:now, c], color=MUTED, lw=0.8, label="past window (input)")
            ax.plot(future_t, z[now:now + h, c], color=BLUE, lw=1.2, label="what happened")
            ax.plot(future_t, forecasts[s, :, c], color=ORANGE, lw=1.2, label="forecast")
            ax.axvline(0, color=INK_SECONDARY, lw=0.6, ls=":")
            if row == 0:
                ax.set_title(args.channels[col], loc="left")
            if col == 0:
                ax.set_ylabel(f"{fraction:.0%} of life\nz-score")
            if row == 1:
                ax.set_xlabel("minutes from now")

        ax = axes[row, -1]
        spans = detector.model.patch_spans()
        centers = [((a + b) / 2 - w) * minutes for a, b in spans]
        ax.bar(centers, attention_per_patch(attention[s]), width=detector.variant["stride"] * minutes * 0.8,
               color=BLUE)
        ax.set_title("attention received per patch", loc="left")
        if row == 1:
            ax.set_xlabel("minutes from now (patch centre)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.45, 1.04))
    fig.suptitle(f"{args.bearing}, forecaster {args.variant} (model of the group that did not see it)",
                 x=0.01, y=1.09, ha="left", fontsize=11)
    fig.tight_layout()
    out = paths["results"] / "figures" / f"forecast_{args.bearing}_{args.variant}.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
