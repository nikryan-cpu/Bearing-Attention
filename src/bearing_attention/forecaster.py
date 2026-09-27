"""Anomaly detector built on the patch forecaster: trained to forecast healthy
feature series, it scores each snapshot by how badly it was forecast."""
import copy
import logging

import numpy as np
import torch

from bearing_attention.model import PatchForecaster

log = logging.getLogger(__name__)


def windows(series, length):
    """All windows of `length` consecutive rows: (n - length + 1, length, channels)."""
    return np.lib.stride_tricks.sliding_window_view(series, length, axis=0).transpose(0, 2, 1)


class ForecastDetector:
    def __init__(self, name, variant, model_cfg):
        self.name = name
        self.variant = dict(variant)
        self.model_cfg = dict(model_cfg)
        self.window, self.horizon = variant["window"], variant["horizon"]
        self.loss_history = []

    def _build(self):
        m = self.model_cfg
        return PatchForecaster(self.window, self.horizon, self.variant["patch"], self.variant["stride"],
                               m["d_model"], m["n_heads"], m["n_layers"], m["d_ff"], m["dropout"])

    def fit(self, healthy, feature_names):
        m = self.model_cfg
        torch.manual_seed(m["seed"])
        rng = np.random.default_rng(m["seed"])
        span = self.window + self.horizon
        chunks = [windows(h.astype(np.float32), span) for h in healthy if len(h) >= span]
        if not chunks:
            raise ValueError(f"no healthy segment is longer than {span} snapshots")
        data = torch.from_numpy(np.concatenate(chunks))
        self.feature_names = list(feature_names)

        self.model = self._build()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
        self.model.train()
        for epoch in range(m["epochs"]):
            order = rng.permutation(len(data))
            total = 0.0
            for start in range(0, len(order), m["batch_size"]):
                batch = data[order[start:start + m["batch_size"]]]
                loss = torch.mean((self.model(batch[:, :self.window]) - batch[:, self.window:]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history.append(total / len(data))
        log.debug("%s: %d windows, final loss %.4f", self.name, len(data), self.loss_history[-1])

        # Channels differ in how predictable they are (kurtosis is much noisier than
        # RMS), so each channel's error is measured against its typical healthy error.
        errors = self._squared_errors(data[:, :self.window], data[:, self.window:])
        self.channel_scale = np.median(errors.reshape(-1, errors.shape[-1]), axis=0) + 1e-6
        return self

    @torch.no_grad()
    def _squared_errors(self, past, future):
        self.model.eval()
        out = [(self.model(past[i:i + 1024]) - future[i:i + 1024]) ** 2 for i in range(0, len(past), 1024)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def forecast(self, z, return_attention=False):
        """Forecasts made from every full window of z: row s holds the forecast of
        snapshots window + s ... window + s + horizon - 1."""
        self.model.eval()
        past = torch.from_numpy(windows(z.astype(np.float32), self.window)[:-1].copy())
        outputs = [self.model(past[i:i + 512], return_attention=return_attention) for i in range(0, len(past), 512)]
        if not return_attention:
            return torch.cat(outputs).numpy()
        return torch.cat([f for f, _ in outputs]).numpy(), torch.cat([a for _, a in outputs]).numpy()

    def score(self, z, stop=None):
        """Log of the mean normalized squared error of every snapshot over all forecasts
        that predicted it. Uses only earlier snapshots, so it can run online; NaN until
        the first full window.

        The errors span several orders of magnitude (a transient can be a thousand
        times the typical error), so the log is taken; alarms only depend on the order
        of the scores, but on this scale the evaluation's clip never reaches a threshold.
        """
        z = z[:stop]
        n = len(z)
        scores = np.full(n, np.nan)
        if n <= self.window:
            return scores
        forecasts = self.forecast(z)
        total, count = np.zeros(n), np.zeros(n)
        for h in range(self.horizon):
            t = np.arange(self.window + h, n)
            error = ((forecasts[: len(t), h] - z[t]) ** 2 / self.channel_scale).mean(axis=1)
            np.add.at(total, t, error)
            np.add.at(count, t, 1)
        seen = count > 0
        scores[seen] = np.log(total[seen] / count[seen])
        return scores

    def for_features(self, feature_names):
        """The same model for another sensor layout (e.g. one accelerometer instead of
        two). Nothing is refitted: each channel keeps the healthy error scale learned
        for its feature, averaged over the training axes."""
        by_feature = {}
        for name, scale in zip(self.feature_names, self.channel_scale):
            by_feature.setdefault(name.split("_", 1)[1], []).append(scale)
        adapted = copy.copy(self)
        adapted.feature_names = list(feature_names)
        adapted.channel_scale = np.array([np.mean(by_feature[n.split("_", 1)[1]]) for n in feature_names])
        return adapted

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "variant": self.variant, "model_cfg": self.model_cfg,
                    "state": self.model.state_dict(), "channel_scale": self.channel_scale,
                    "feature_names": self.feature_names, "loss_history": self.loss_history}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, weights_only=False)
        detector = cls(saved["name"], saved["variant"], saved["model_cfg"])
        detector.model = detector._build()
        detector.model.load_state_dict(saved["state"])
        detector.channel_scale = saved["channel_scale"]
        detector.feature_names = saved["feature_names"]
        detector.loss_history = saved["loss_history"]
        return detector


class LevelOrSurprise:
    """Added after the first results, not part of the original plan: the larger of the
    RMS level score and the forecast-error score, each on its own healthy scale.
    RMS sees slow growth that the forecaster finds predictable; the forecaster sees
    changes in behaviour that barely move RMS."""

    def __init__(self, name, level, surprise):
        self.name = name
        self.parts = [level, surprise]

    def fit(self, healthy, feature_names):
        for part in self.parts:
            part.fit(healthy, feature_names)
        return self.scale_parts(healthy)

    def scale_parts(self, healthy):
        """Put each already fitted part on the scale of its scores on `healthy`."""
        self.scales = []
        for part in self.parts:
            scores = np.concatenate([part.score(h) for h in healthy])
            scores = scores[np.isfinite(scores)]
            center = np.median(scores)
            self.scales.append((center, 1.4826 * np.median(np.abs(scores - center)) + 1e-12))
        return self

    def score(self, z, stop=None):
        parts = [(part.score(z, stop) - center) / scale
                 for part, (center, scale) in zip(self.parts, self.scales)]
        # before the forecaster's first full window only the RMS part exists
        return np.fmax(*parts)


def variants(cfg):
    """{detector name: factory} for every configured window/patch variant."""
    t = cfg["transformer"]

    def factory(name):
        return lambda: ForecastDetector(f"transformer_{name}", t["variants"][name], t["model"])

    return {f"transformer_{name}": factory(name) for name in t["variants"]}
