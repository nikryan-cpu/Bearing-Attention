import numpy as np
import pytest
import torch

from bearing_attention.forecaster import ForecastDetector, windows
from bearing_attention.model import PatchForecaster

SMALL = {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.0,
         "epochs": 40, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "seed": 0}
VARIANT = {"window": 32, "patch": 8, "stride": 4, "horizon": 4}


def small_model():
    torch.manual_seed(0)
    return PatchForecaster(32, 4, 8, 4, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0).eval()


def test_shapes_and_attention():
    model = small_model()
    forecast, attention = model(torch.randn(5, 32, 3), return_attention=True)
    assert forecast.shape == (5, 4, 3)
    assert attention.shape == (5, 3, 1, 2, model.n_patches, model.n_patches)
    torch.testing.assert_close(attention.sum(-1), torch.ones(attention.shape[:-1]))
    assert model.patch_spans()[-1] == (24, 32)


def test_patches_must_tile_the_window():
    with pytest.raises(ValueError):
        PatchForecaster(30, 4, 8, 4)


def test_channels_are_independent():
    model = small_model()
    x = torch.randn(2, 32, 3)
    changed = x.clone()
    changed[:, :, 1] += torch.randn(2, 32)
    with torch.no_grad():
        a, b = model(x), model(changed)
    torch.testing.assert_close(a[:, :, [0, 2]], b[:, :, [0, 2]])
    assert not torch.allclose(a[:, :, 1], b[:, :, 1])


def test_level_shift_moves_the_forecast_by_the_same_amount():
    model = small_model()
    x = torch.randn(2, 32, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(x + 7.0), model(x) + 7.0, atol=1e-5, rtol=0)


def sines(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    return np.sin(2 * np.pi * t / np.array([10.0, 17.0]) + rng.uniform(0, 6, 2)) + 0.05 * rng.normal(size=(n, 2))


def test_learns_to_forecast_and_flags_a_change():
    detector = ForecastDetector("t", VARIANT, SMALL).fit([sines(400, s) for s in range(3)], ["a", "b"])
    assert detector.loss_history[-1] < 0.2 * detector.loss_history[0]

    record = sines(200, 7)
    record[150:] += 3 * np.random.default_rng(1).normal(size=(50, 2))
    score = detector.score(record)
    assert np.isnan(score[:32]).all()
    # scores are log errors: the change makes the error at least ten times larger
    assert np.nanmedian(score[160:]) - np.nanmedian(score[40:140]) > np.log(10)


def test_score_is_causal_and_survives_save_load(tmp_path):
    detector = ForecastDetector("t", VARIANT, {**SMALL, "epochs": 2}).fit([sines(200, 0)], ["a", "b"])
    record = sines(120, 3)
    full = detector.score(record)
    np.testing.assert_allclose(detector.score(record, stop=80), full[:80])
    detector.save(tmp_path / "m.pt")
    np.testing.assert_allclose(ForecastDetector.load(tmp_path / "m.pt").score(record), full)


def test_windows():
    w = windows(np.arange(10).reshape(5, 2), 3)
    assert w.shape == (3, 3, 2)
    assert w[1, :, 0].tolist() == [2, 4, 6]
