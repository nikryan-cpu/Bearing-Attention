import importlib.util
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
HAS_DATA = any((APP.parent / "data").glob("*.npz"))


def load_app_module():
    spec = importlib.util.spec_from_file_location("demo_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resynthesized_sound_carries_the_impact_rhythm():
    app = load_app_module()
    psd_freqs = np.linspace(39, 9961, 128)
    env_freqs = np.arange(5, 700, 10.0)
    env = np.full(len(env_freqs), 0.01)
    env[env_freqs == 155] = 0.5  # a strong line at 155 Hz
    sound = app.resynthesize(np.full(128, -4.0), psd_freqs, env, env_freqs, fs=20000, seconds=1.0)
    assert sound.shape == (20000,) and np.isfinite(sound).all()
    assert np.abs(sound).max() <= 0.8 + 1e-9

    envelope_spectrum = np.abs(np.fft.rfft(np.abs(sound) - np.abs(sound).mean()))
    freqs = np.fft.rfftfreq(len(sound), 1 / 20000)
    band = (freqs > 20) & (freqs < 700)
    assert abs(freqs[band][np.argmax(envelope_spectrum[band])] - 155) < 3


@pytest.mark.skipif(not HAS_DATA, reason="run scripts/export_demo.py first")
@pytest.mark.parametrize("no_raw", [False, True])
def test_app_runs_on_every_dataset_and_method(no_raw, monkeypatch):
    from streamlit.testing.v1 import AppTest

    if no_raw:
        monkeypatch.setenv("DEMO_NO_RAW", "1")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    for dataset in ("IMS", "FEMTO"):
        at.sidebar.radio[0].set_value(dataset).run()
        for method in ("ecod", "transformer_w128_p16"):
            at.sidebar.radio[1].set_value(method).run()
            assert not at.exception, (dataset, method)
