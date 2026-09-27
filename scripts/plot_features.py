"""Sanity-check figures for the FEMTO features and the healthy-segment choice."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch

from bearing_attention import features, femto
from bearing_attention.config import load_config
from bearing_attention.plotting import (BLUE, DIVERGING, INK_SECONDARY, MUTED, ORANGE, SHADE,
                                        apply_style, plt)


def rms_over_life(records, names, n_baseline, healthy_fraction, out):
    idx = [names.index("horiz_rms"), names.index("vert_rms")]
    fig, axes = plt.subplots(6, 3, figsize=(10, 12), sharex=True)
    for ax, (bearing, rec) in zip(axes.ravel(), sorted(records.items())):
        rms = rec["features"][:, idx]
        life = np.arange(len(rms)) / len(rms)
        ax.axvspan(0, healthy_fraction, color=SHADE, lw=0)
        ax.axvline(n_baseline / len(rms), color=MUTED, lw=0.8, ls=":")
        ax.semilogy(life, rms[:, 0], color=BLUE, lw=0.7)
        ax.semilogy(life, rms[:, 1], color=ORANGE, lw=0.7)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
        note = "" if rec["end_is_failure"] else ", end ambiguous"
        ax.set_title(f"{bearing} ({len(rms)} snapshots{note})", loc="left")
    axes.ravel()[-1].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("fraction of life")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMS (g)")
    handles = [Line2D([], [], color=BLUE, label="horizontal"),
               Line2D([], [], color=ORANGE, label="vertical"),
               Patch(color=SHADE, label=f"healthy segment (first {healthy_fraction:.0%})"),
               Line2D([], [], color=MUTED, ls=":", label=f"end of the {n_baseline}-snapshot baseline")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("RMS over the life of every FEMTO bearing", y=1.045, x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def healthy_vs_end(records, names, n_baseline, healthy_fraction, out):
    idx = [names.index("horiz_rms"), names.index("vert_rms")]
    rows = []
    for bearing, rec in sorted(records.items()):
        z = features.baseline_zscore(rec["features"], n_baseline)[:, idx]
        n = len(z)
        healthy = z[: max(int(healthy_fraction * n), 1)]
        rows.append((bearing, np.percentile(healthy, 1), np.percentile(healthy, 99), np.median(z[-10:].max(axis=1))))

    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(rows))
    lo, hi, end = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    ax.vlines(x, lo, hi, color=BLUE, lw=3, label="healthy segment, 1st to 99th percentile of both axes")
    ax.scatter(x, end, color=ORANGE, s=28, zorder=3, label="last 10 snapshots, median of the stronger axis")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_yscale("symlog", linthresh=5)
    ticks = [-20, -10, -5, 0, 5, 10, 20, 50]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_ylim(-25, 80)
    ax.set_xticks(x, [r[0].removeprefix("Bearing") for r in rows])
    ax.set_xlabel("bearing")
    ax.set_ylabel("normalized RMS (robust z)")
    ax.set_title("Normalized RMS: healthy segment vs end of life", loc="left",
                 fontsize=10, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def feature_heatmap(bearing, rec, names, n_baseline, interval_s, out):
    z = np.clip(features.baseline_zscore(rec["features"], n_baseline), -10, 10)
    hours = len(z) * interval_s / 3600
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(z.T, aspect="auto", cmap=DIVERGING, vmin=-10, vmax=10, interpolation="nearest",
                      extent=(0, hours, len(names) - 0.5, -0.5))
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("hours since start")
    ax.grid(False)
    ax.set_title(f"{bearing}: every feature, z-scored against the first {n_baseline} snapshots",
                 loc="left", fontsize=10)
    fig.colorbar(image, ax=ax, label="z (clipped to +-10)", pad=0.01)
    fig.savefig(out)
    plt.close(fig)


def envelope_check(subset, bearing, cfg, paths, out):
    b = femto.load_bearing(paths["raw"] / "femto" / subset / bearing, paths["processed"] / "femto")
    feature_cfg = cfg["features"]
    fs = cfg["datasets"]["femto"]["sampling_rate_hz"]
    axis = femto.AXES.index("vert")
    early = b.acc[: int(cfg["normalization"]["healthy_fraction"] * len(b)), :, axis]
    late = b.acc[-20:, :, axis]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for x, color, label in ((early, BLUE, "healthy segment (mean)"), (late, ORANGE, "last 20 snapshots (mean)")):
        env = features.envelope(x - x.mean(axis=1, keepdims=True), fs, feature_cfg["envelope_band_hz"])
        freqs, spec = features.envelope_spectrum(env, fs)
        keep = freqs <= 700
        ax.plot(freqs[keep], spec[:, keep].mean(axis=0), color=color, marker="o", ms=2.5, label=label)
    bpfo, bpfi = b.defect_hz["bpfo"], b.defect_hz["bpfi"]
    for h in (1, 2, 3):
        ax.axvline(h * bpfo, color=INK_SECONDARY, lw=0.8, ls="--")
        ax.text(h * bpfo + 4, ax.get_ylim()[1] * 0.95, f"{h}x BPFO", color=INK_SECONDARY, va="top", fontsize=8)
    ax.axvline(bpfi, color=MUTED, lw=0.8, ls=":")
    ax.text(bpfi + 4, ax.get_ylim()[1] * 0.8, "BPFI", color=MUTED, va="top", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("envelope amplitude (g)")
    ax.set_title(f"{b.name}, vertical axis: envelope spectrum ({feature_cfg['envelope_band_hz'][0] // 1000}-"
                 f"{feature_cfg['envelope_band_hz'][1] // 1000} kHz band), "
                 f"BPFO = {bpfo:.0f} Hz at {b.rpm} rpm", loc="left", fontsize=10)
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    apply_style()
    cfg = load_config(args.config)
    paths, norm = cfg["paths"], cfg["normalization"]
    records, name_lists = features.load_records(paths["processed"] / "features" / "femto.npz")
    names = name_lists["feature_names"]
    out_dir = paths["results"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rms_over_life(records, names, norm["baseline_snapshots"], norm["healthy_fraction"],
                  out_dir / "femto_rms_over_life.png")
    healthy_vs_end(records, names, norm["baseline_snapshots"], norm["healthy_fraction"],
                   out_dir / "femto_healthy_vs_end.png")
    feature_heatmap("Bearing1_1", records["Bearing1_1"], names, norm["baseline_snapshots"],
                    cfg["datasets"]["femto"]["snapshot_interval_s"], out_dir / "femto_features_Bearing1_1.png")
    envelope_check("Learning_set", "Bearing2_2", cfg, paths, out_dir / "femto_envelope_Bearing2_2.png")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
