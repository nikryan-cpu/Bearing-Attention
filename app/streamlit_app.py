"""Bearing Attention demo.

Everything shown here was computed beforehand by scripts/export_demo.py and lives in
app/data, so the app needs only numpy, matplotlib and streamlit. The raw recordings
are not part of it (the datasets may not be redistributed); if they have been
downloaded locally, the app also shows the real waveform and plays the real sound.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import streamlit as st  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
# DEMO_NO_RAW=1 shows locally what the deployed app shows, without the raw recordings
LOCAL_RAW = None if os.environ.get("DEMO_NO_RAW") else HERE.parent / "data" / "processed"

BLUE, ORANGE, INK, INK2, MUTED, GRID, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
GOOD, WARNING, CRITICAL, ALARM_SHADE = "#0ca30c", "#fab219", "#d03b3b", "#f6c9c9"
METHODS = {"rms": "RMS threshold", "ecod": "ECOD", "transformer_w128_p16": "Transformer (128/16)"}
CHANNELS = {"rms": "RMS", "kurtosis": "kurtosis", "band_500_1000": "energy 0.5-1 kHz",
            "band_8000_10000": "energy 8-10 kHz", "env_modulation": "envelope modulation",
            "env_peakiness": "envelope peakiness"}
DEFECT_LABELS = {"bpfo": "BPFO", "bpfi": "BPFI", "bsf": "BSF"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left", "legend.frameon": False,
})


@st.cache_data
def load(path):
    with np.load(path) as z:
        record = {k: z[k] for k in z.files}
    record["meta"] = json.loads(str(record["meta"]))
    return record


def catalog():
    out = {}
    for path in sorted(DATA.glob("*.npz")):
        dataset, bearing = path.stem.split("_", 1)
        out.setdefault(dataset.upper(), []).append((bearing, path))
    return out


@st.cache_resource
def raw_recording(dataset, bearing, subset):
    """The real vibration of one bearing (first axis), if the dataset was downloaded locally."""
    if LOCAL_RAW is None:
        return None
    if dataset == "FEMTO":
        path = LOCAL_RAW / "femto" / f"{subset}_{bearing}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["acc"][:, :, 0]
    else:
        path = LOCAL_RAW / "ims" / "ims_2nd_test.npz"
        if path.exists():
            with np.load(path) as z:
                return z["acc"][:, :, int(bearing.removeprefix("Bearing")) - 1]
    return None


def time_label(snapshots, interval_s):
    seconds = snapshots * interval_s
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.2f} days"


def time_axis(n, interval_s):
    """x values and axis label for a whole life."""
    if n * interval_s > 2 * 86400:
        return np.arange(n) * interval_s / 86400, "days since start"
    return np.arange(n) * interval_s / 3600, "hours since start"


def alarm_onsets(states):
    return np.flatnonzero(states & ~np.concatenate([[False], states[:-1]]))


def lead_times(states, healthy_end, interval_s):
    """Lead of the alarm that stays on until the end, and of the first alarm after the healthy segment."""
    onsets = alarm_onsets(states)
    n = len(states)
    sustained = onsets[-1] if states[-1] and onsets[-1] >= healthy_end else None
    later = onsets[onsets >= healthy_end]
    return (None if sustained is None else time_label(n - 1 - sustained, interval_s),
            None if not len(later) else time_label(n - 1 - later[0], interval_s),
            int(((onsets >= 0) & (onsets < healthy_end)).sum()))


def status(score, threshold, in_alarm):
    if in_alarm:
        return CRITICAL, "Alarm"
    if not np.isfinite(score):
        return MUTED, "No score yet (the method is warming up)"
    if score > 0.6 * threshold:
        return WARNING, "Elevated"
    return GOOD, "Normal"


def resynthesize(psd_row, psd_freqs, env_row, env_freqs, fs=20000, seconds=2.0, seed=0):
    """Sound with the snapshot's power spectrum and the rhythm of its envelope spectrum.

    The phases are random, so this is not the recorded signal, only something that
    sounds like it: a hiss shaped by the spectrum, modulated at the frequencies where
    the envelope spectrum has lines (the repetition rate of impacts).
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    amplitude = np.sqrt(np.interp(freqs, psd_freqs, 10.0 ** psd_row.astype(np.float64), right=0))
    carrier = np.fft.irfft(amplitude * np.exp(2j * np.pi * rng.random(len(freqs))), n)
    t = np.arange(n) / fs
    lines = np.clip(env_row.astype(np.float64) - np.nanmedian(env_row), 0, None)
    lines[(env_freqs < 20) | ~np.isfinite(lines)] = 0
    modulation = sum(a * np.cos(2 * np.pi * f * t + 2 * np.pi * rng.random())
                     for a, f in zip(lines, env_freqs) if a > 0)
    if np.any(modulation):
        carrier = carrier * (1 + 0.8 * modulation / np.abs(modulation).max())
    return fade(carrier)


def fade(x, ms=20, fs=20000):
    ramp = min(len(x) // 2, int(fs * ms / 1000))
    x = x / (np.abs(x).max() + 1e-12) * 0.8
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return x


def live_bearing(rec, method, rate_index, t):
    meta = rec["meta"]
    channels = meta["forecast_channels"]
    rms = channels.index("rms")
    clean = lambda a, digits: [round(float(v), digits) if np.isfinite(v) else None for v in np.ravel(a)]
    defect_hz = meta["defect_hz"]
    shaft_hz = meta["rpm"] / 60
    spans = meta["patch_spans"]
    data = {
        "n": meta["n"], "interval_s": meta["interval_s"], "rpm": meta["rpm"], "start": t,
        "n_elements": meta["n_elements"],
        # IMS publishes no geometry, so its drawing uses generic ratios and is only a schematic
        "cage_ratio": defect_hz["ftf"] / shaft_hz if defect_hz else 0.4,
        "spin_ratio": defect_hz["bsf"] / shaft_hz if defect_hz else 3.0,
        "score": clean(rec[f"score_{method}"], 3),
        "threshold": float(rec[f"thresholds_{method}"][rate_index]),
        "alarm": [int(v) for v in rec[f"alarm_{method}"][rate_index]],
        "rmsz": clean(rec["z"][:, rms], 3),
        "forecast": [clean(row, 3) for row in rec["forecast"][:, :, rms]],
        "attention": [clean(row, 4) for row in rec["attention"]],
        "patch_spans": spans, "stride": spans[1][0] - spans[0][0],
        "window": meta["window"], "horizon": meta["horizon"], "healthy_end": meta["healthy_end"],
        "method_label": METHODS[method], "rate_label": f"{meta['false_alarm_rates'][rate_index]:.1%}",
        "defects": [clean(row, 2) for row in rec["defects"]] if "defects" in rec else None,
        # healthy envelope ratios sit around 1.5-2; well above that a defect line stands out
        "defect_highlight": 3.0,
        "defect_note": ("At the 10 Hz resolution of a 0.1 s snapshot, BPFI and twice BSF overlap for condition 1."
                        if defect_hz else
                        "Defect frequencies are not shown: the IMS documentation gives no bearing geometry. "
                        "The drawing is a schematic."),
    }
    html = (HERE / "bearing_view.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data))
    st.caption("The player starts at the moment chosen above and then runs on its own. Rotation is slowed down "
               "60 times; colour and shaking follow the selected method's score and the RMS level.")
    st.iframe(html, height=480)


def signal_tab(rec, t, raw):
    meta = rec["meta"]
    x, xlabel = time_axis(meta["n"], meta["interval_s"])
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        freqs_khz = rec["psd_freqs"] / 1000
        ax.imshow(rec["psd"].T.astype(np.float32), aspect="auto", origin="lower", cmap="viridis",
                  extent=(x[0], x[-1], freqs_khz[0], freqs_khz[-1]), interpolation="nearest")
        ax.axvline(x[t], color="white", lw=1.2)
        ax.grid(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frequency (kHz)")
        ax.set_title("Spectrogram over the whole life (log power)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        healthy = np.median(rec["psd"][: meta["healthy_end"]].astype(np.float32), axis=0)
        ax.plot(rec["psd_freqs"] / 1000, healthy, color=MUTED, lw=1, label="healthy segment (median)")
        ax.plot(rec["psd_freqs"] / 1000, rec["psd"][t].astype(np.float32), color=BLUE, lw=1.2, label="this snapshot")
        ax.set_xlabel("frequency (kHz)")
        ax.set_ylabel("log10 power")
        ax.set_title("Power spectrum")
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.0))
        ax.plot(rec["env_freqs"], rec["env"][t].astype(np.float32), color=BLUE, lw=1.2)
        if meta["defect_hz"]:
            top = np.nanmax(rec["env"][t].astype(np.float32))
            for name, label in DEFECT_LABELS.items():
                f0 = meta["defect_hz"][name]
                for h in (1, 2, 3):
                    if h * f0 <= rec["env_freqs"][-1]:
                        ax.axvline(h * f0, color=INK2 if name == "bpfo" else MUTED, lw=0.7, ls="--")
                ax.text(f0 + 3, top * (0.95 if name == "bpfo" else 0.8 if name == "bpfi" else 0.65), label,
                        color=INK2, fontsize=8)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
        ax.set_title("Envelope spectrum (2-9 kHz band): repetition rate of impacts")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        if raw is not None:
            fig, ax = plt.subplots(figsize=(6, 3.0))
            seconds = np.arange(raw.shape[1]) / meta["fs"]
            ax.plot(seconds, raw[t], color=BLUE, lw=0.5)
            ax.set_xlabel("seconds")
            ax.set_ylabel("acceleration")
            ax.set_title("Raw snapshot (local data)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown("**Raw waveform**")
            st.caption("The raw recordings are not shipped with this app because the datasets may not be "
                       "redistributed. Run the app locally after `python scripts/download_data.py` to see them.")

    st.markdown("**Listen**")
    if raw is not None:
        # a FEMTO snapshot lasts 0.1 s, so twenty consecutive ones (3.3 min of life) are chained
        count = 20 if meta["dataset"] == "FEMTO" else 2
        clip = raw[t:t + count].reshape(-1).astype(np.float64)
        st.audio(fade(clip - clip.mean(), fs=meta["fs"]), sample_rate=meta["fs"])
        st.caption("Real recording at its original sampling rate; the 0-10 kHz band is audible as it is.")
    else:
        st.audio(resynthesize(rec["psd"][t], rec["psd_freqs"], rec["env"][t], rec["env_freqs"], seed=t),
                 sample_rate=20000)
        st.caption("Resynthesized from this snapshot's spectrum and envelope spectrum with random phases: "
                   "it sounds like the vibration (hiss level and colour, impact rhythm) but it is not the "
                   "recording itself.")


def forecast_tab(rec, t):
    meta = rec["meta"]
    channels = meta["forecast_channels"]
    channel = st.selectbox("Feature", channels, format_func=CHANNELS.get, key="forecast_channel")
    c = channels.index(channel)
    window, horizon = meta["window"], meta["horizon"]
    s = t - window
    st.caption(f"The transformer sees the last {window} snapshots ({time_label(window, meta['interval_s'])}) "
               f"and forecasts the next {horizon}. Model: {meta['model_note']}.")
    if s < 0 or s >= len(rec["forecast"]):
        st.info(f"Move the slider: a forecast needs {window} snapshots of history and {horizon} after it.")
        return
    minutes = meta["interval_s"] / 60
    unit, per = ("h", 60) if meta["interval_s"] >= 600 else ("min", 1)
    rel = lambda idx: (np.asarray(idx) - t) * minutes / per
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        z = rec["z"][:, c]
        ax.plot(rel(np.arange(s, t)), z[s:t], color=MUTED, lw=0.9, label="past window (input)")
        ax.plot(rel(np.arange(t, t + horizon)), z[t:t + horizon], color=BLUE, lw=1.5, label="what happened")
        ax.plot(rel(np.arange(t, t + horizon)), rec["forecast"][s, :, c].astype(np.float32), color=ORANGE, lw=1.5,
                label="forecast")
        ax.axvline(0, color=INK2, lw=0.7, ls=":")
        ax.set_xlabel(f"{unit} from now")
        ax.set_ylabel("z-score vs the start of life")
        ax.set_title(f"{CHANNELS[channel]}: forecast vs reality", pad=22)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        spans = meta["patch_spans"]
        centers = [rel(s + (a + b) / 2) for a, b in spans]
        width = (spans[1][0] - spans[0][0]) * minutes / per * 0.8
        ax.bar(centers, rec["attention"][s], width=width, color=BLUE)
        ax.axhline(1 / len(spans), color=INK2, lw=0.8, ls="--")
        ax.text(centers[0], 1 / len(spans) * 1.04, "equal attention", color=INK2, fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(f"{unit} from now (patch centre)")
        ax.set_title("Which parts of the past the model looked at", pad=22)
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Attention: how much weight the patches of the window receive from the others, averaged over "
               "layers, heads and features. Flat bars mean the model uses the whole recent trend; a tall bar "
               "near 'now' means it relies on what just changed.")


def methods_tab(rec, t, rate_index):
    meta = rec["meta"]
    x, xlabel = time_axis(meta["n"], meta["interval_s"])
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(10, 2.0 * len(METHODS)), sharex=True)
    rows = []
    for ax, (method, label) in zip(axes, METHODS.items()):
        score = rec[f"score_{method}"]
        threshold = rec[f"thresholds_{method}"][rate_index]
        states = rec[f"alarm_{method}"][rate_index]
        ax.axvspan(x[0], x[meta["healthy_end"]], color=SHADE, lw=0)
        ax.fill_between(x, 0, 1, where=states, transform=ax.get_xaxis_transform(), color=ALARM_SHADE, lw=0)
        ax.plot(x, score, color=BLUE, lw=0.8)
        ax.axhline(threshold, color=INK2, lw=0.9, ls="--")
        ax.axvline(x[t], color=INK, lw=1)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([-10, 0, 10, 100], ["-10", "0", "10", "100"])
        ax.set_ylim(-12, 110)
        ax.set_title(label)
        sustained, first, false_alarms = lead_times(states, meta["healthy_end"], meta["interval_s"])
        rows.append({"method": label, "threshold": round(float(threshold), 2),
                     "sustained alarm before the end": sustained or "none",
                     "first alarm after the healthy segment": first or "none",
                     "alarms during the healthy segment": false_alarms})
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(rows, hide_index=True, width="stretch")
    note = ("This bearing's record ends at failure." if meta["end_is_failure"] else
            "This bearing did not fail; late alarms here come from the failing bearing 1 on the same shaft "
            "or from its own wear." if meta["dataset"] == "IMS" else
            "The end of this record is ambiguous, see the README.")
    st.caption("Grey: healthy segment. Red: alarm on. Scores are in units of their spread on held-out healthy "
               "data and capped at 100. " + note)


ABOUT = """
**What this is.** Vibration features of a bearing are forecast by a small transformer trained only on healthy
data; when the forecast error becomes unusual, the bearing is flagged. It is compared with a plain RMS threshold
and with ECOD, all at the same false-alarm rate. Everything here is precomputed.

**Data.** FEMTO/PRONOSTIA (IEEE PHM 2012 challenge, accelerated life tests, one 0.1 s snapshot every 10 s) and
NASA IMS test 2 (four bearings on one shaft, one 1 s snapshot every 10 minutes; bearing 1 failed). IMS is only
used to check how a model trained on FEMTO transfers, without retraining.

**Main results.** On FEMTO the transformer is not better than the simple baselines at a strict false-alarm rate:
most FEMTO failures are abrupt, and short transients in the healthy data raise its threshold. It reacts to
changes in behaviour rather than to slow growth. On IMS the FEMTO model finds the failure of bearing 1 about two
days ahead, RMS and ECOD about three.

**Limitations.** Laboratory data with accelerated ageing, few bearings, very different snapshot rates, no labels
for the onset of wear (only the end of each test), and no published benchmark to compare with.

**Sources.** P. Nectoux et al., PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test, IEEE PHM
2012; "FEMTO Bearing Data Set" and J. Lee, H. Qiu, G. Yu, J. Lin, Rexnord Technical Services, "Bearing Data Set",
IMS, University of Cincinnati, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
"""


def main():
    st.set_page_config(page_title="Bearing Attention", layout="wide")
    st.title("Bearing Attention")
    st.caption("Early detection of bearing wear from vibration: a small forecasting transformer against simple "
               "baselines, on precomputed results.")

    records = catalog()
    dataset = st.sidebar.radio("Dataset", list(records), horizontal=True)
    bearing, path = st.sidebar.selectbox("Bearing", records[dataset], format_func=lambda item: item[0])
    rec = load(path)
    meta = rec["meta"]
    method = st.sidebar.radio("Method", list(METHODS), format_func=METHODS.get)
    rates = meta["false_alarm_rates"]
    rate = st.sidebar.select_slider("False-alarm rate", options=rates, value=0.01, format_func="{:.1%}".format)
    rate_index = rates.index(rate)
    st.sidebar.caption(f"{meta['dataset']} {bearing}: {meta['n']} snapshots, one every "
                       f"{time_label(1, meta['interval_s'])}, {meta['rpm']} rpm. "
                       + ("The record ends at failure." if meta["end_is_failure"] else "This bearing did not fail."))

    t = st.slider("Moment in the life (snapshot)", 0, meta["n"] - 1, value=int(0.9 * meta["n"]), key=f"t_{path.stem}")
    score = float(rec[f"score_{method}"][t])
    threshold = float(rec[f"thresholds_{method}"][rate_index])
    color, label = status(score, threshold, bool(rec[f"alarm_{method}"][rate_index][t]))
    score_text = f"{score:.1f}" if np.isfinite(score) else "-"
    st.markdown(
        f"<div style='display:flex;gap:24px;align-items:center;flex-wrap:wrap'>"
        f"<div style='display:flex;gap:8px;align-items:center'><div style='width:16px;height:16px;border-radius:3px;"
        f"background:{color}'></div><b>{label}</b></div>"
        f"<div>{METHODS[method]} score <b>{score_text}</b>, threshold <b>{threshold:.1f}</b></div>"
        f"<div>{time_label(t, meta['interval_s'])} since start, {time_label(meta['n'] - 1 - t, meta['interval_s'])} "
        f"to the end of the test</div></div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Live bearing", "Signal and sound", "Forecast and attention", "All methods", "About"])
    with tabs[0]:
        live_bearing(rec, method, rate_index, t)
    with tabs[1]:
        signal_tab(rec, t, raw_recording(meta["dataset"], bearing, meta.get("subset")))
    with tabs[2]:
        forecast_tab(rec, t)
    with tabs[3]:
        methods_tab(rec, t, rate_index)
    with tabs[4]:
        st.markdown(ABOUT)



if __name__ == "__main__":
    main()
