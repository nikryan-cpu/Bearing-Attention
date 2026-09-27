"""Per-snapshot vibration features, computed the same way for every dataset.

Everything is defined in physical units (Hz, seconds), so FEMTO (25.6 kHz,
0.1 s snapshots) and IMS (20 kHz, 1 s snapshots) produce the same features.
Defect-frequency features need the bearing geometry, which only FEMTO documents
officially, so they are kept apart from the features the detectors use.
"""
import numpy as np
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt
from scipy.stats import kurtosis

DEFECTS = ("bpfo", "bpfi", "bsf")

# Deviations smaller than 1 % (0.01 in log units) are treated as noise, which also
# keeps nearly constant baseline features from producing huge z-scores.
MIN_LOG_SCALE = 0.01


def defect_frequencies(shaft_hz, n_elements, element_d, pitch_d, contact_angle_deg=0.0):
    """Kinematic defect frequencies in Hz: outer race, inner race, ball spin, cage."""
    ratio = element_d / pitch_d * np.cos(np.radians(contact_angle_deg))
    return {
        "bpfo": n_elements / 2 * shaft_hz * (1 - ratio),
        "bpfi": n_elements / 2 * shaft_hz * (1 + ratio),
        "bsf": pitch_d / (2 * element_d) * shaft_hz * (1 - ratio ** 2),
        "ftf": shaft_hz / 2 * (1 - ratio),
    }


def feature_names(bands_hz):
    names = ["rms", "peak", "kurtosis", "crest"]
    names += [f"band_{lo}_{hi}" for lo, hi in zip(bands_hz[:-1], bands_hz[1:])]
    return names + ["env_modulation", "env_peakiness"]


def envelope(x, fs, band_hz):
    sos = butter(4, band_hz, btype="bandpass", fs=fs, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1))


def envelope_spectrum(env, fs):
    """One-sided amplitude spectrum of each envelope row (mean removed, Hann window)."""
    window = np.hanning(env.shape[-1])
    centered = env - env.mean(axis=-1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered * window, axis=-1)) * 2 / window.sum()
    return np.fft.rfftfreq(env.shape[-1], 1 / fs), spectrum


def _background(freqs, spectrum, max_hz):
    return np.median(spectrum[:, (freqs > 0) & (freqs <= max_hz)], axis=1)


def snapshot_features(x, fs, cfg):
    """Detector features of every snapshot of one sensor axis.

    x: (n_snapshots, n_samples) -> (n_snapshots, len(feature_names(cfg["bands_hz"]))).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    peak = np.abs(x).max(axis=1)
    columns = [rms, peak, kurtosis(x, axis=1, fisher=False), peak / rms]

    freqs, psd = periodogram(x, fs=fs, window="hann", axis=1)
    df = freqs[1] - freqs[0]
    bands = cfg["bands_hz"]
    for lo, hi in zip(bands[:-1], bands[1:]):
        columns.append(psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df)

    env = envelope(x, fs, cfg["envelope_band_hz"])
    columns.append(env.std(axis=1) / env.mean(axis=1))
    env_freqs, env_spec = envelope_spectrum(env, fs)
    # Periodic impacts show up as a line in the envelope spectrum, whatever their frequency.
    low = (env_freqs >= cfg["envelope_min_hz"]) & (env_freqs <= cfg["envelope_max_hz"])
    columns.append(env_spec[:, low].max(axis=1) / _background(env_freqs, env_spec, cfg["envelope_max_hz"]))
    return np.stack(columns, axis=1)


def defect_ratios(x, fs, cfg, defect_hz):
    """Envelope amplitude at BPFO, BPFI and BSF (first harmonics) over the median level.

    x: (n_snapshots, n_samples) -> (n_snapshots, 3) in the order of DEFECTS.
    """
    x = np.asarray(x, dtype=np.float64)
    freqs, spec = envelope_spectrum(envelope(x - x.mean(axis=1, keepdims=True), fs,
                                             cfg["envelope_band_hz"]), fs)
    background = _background(freqs, spec, cfg["envelope_max_hz"])
    df = freqs[1] - freqs[0]
    columns = []
    for name in DEFECTS:
        f0 = defect_hz[name]
        peaks = []
        for h in range(1, cfg["defect_harmonics"] + 1):
            # 2 % covers slip and speed uncertainty; at least 1.5 bins for coarse spectra
            tol = max(1.5 * df, 0.02 * h * f0)
            peaks.append(spec[:, np.abs(freqs - h * f0) <= tol].max(axis=1))
        columns.append(np.mean(peaks, axis=0) / background)
    return np.stack(columns, axis=1)


def bearing_features(acc, fs, cfg, axis_names, chunk=256):
    """Detector features of a whole record. acc: (n_snapshots, n_samples, n_axes).

    Snapshots are processed `chunk` at a time to bound memory on long IMS snapshots.
    """
    per_axis = feature_names(cfg["bands_hz"])
    blocks = [np.concatenate([snapshot_features(acc[i:i + chunk, :, a], fs, cfg) for i in range(0, len(acc), chunk)])
              for a in range(len(axis_names))]
    names = [f"{axis}_{name}" for axis in axis_names for name in per_axis]
    return np.concatenate(blocks, axis=1), names


def save_records(path, records, **name_lists):
    """Save {bearing: {field: array}} plus lists of column names to one .npz file."""
    arrays = {key: np.asarray(names) for key, names in name_lists.items()}
    for bearing, fields in records.items():
        for field, value in fields.items():
            arrays[f"{bearing}__{field}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_records(path):
    """Inverse of save_records: (records, {list name: list of column names})."""
    records, name_lists = {}, {}
    with np.load(path) as z:
        for key in z.files:
            if "__" in key:
                bearing, field = key.split("__", 1)
                value = z[key]
                records.setdefault(bearing, {})[field] = value.item() if value.ndim == 0 else value
            else:
                name_lists[key] = [str(n) for n in z[key]]
    return records, name_lists


def baseline_zscore(features, n_baseline):
    """Robust z-score of log features against the first `n_baseline` snapshots.

    All features are positive and grow multiplicatively with damage, so they are
    compared in log space; median and MAD keep a short start-up transient in the
    baseline from inflating the scale.
    """
    logf = np.log(np.maximum(features, 1e-12))
    base = logf[:n_baseline]
    center = np.median(base, axis=0)
    scale = 1.4826 * np.median(np.abs(base - center), axis=0)
    return (logf - center) / np.maximum(scale, MIN_LOG_SCALE)
