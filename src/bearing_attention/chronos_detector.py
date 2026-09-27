"""Zero-shot comparison: Chronos-Bolt forecasting the same feature series.

Chronos-Bolt (Amazon, chronos-forecasting package, Apache-2.0) is pretrained on a
large collection of public time series and is used here without any training on
bearings. Only the typical healthy error of each feature is measured, exactly as
for the transformer, and the score is built the same way.
"""
import hashlib

import numpy as np
import torch

from bearing_attention.forecaster import ErrorScore, windows

_PIPELINES = {}
# Forecasts do not depend on the fold, so each record is forecast once and reused
# by every fit and score of the cross-validation.
_FORECASTS = {}
MEDIAN = 4  # index of the 0.5 quantile among Chronos-Bolt's 0.1 ... 0.9 outputs


def load_pipeline(model_id):
    if model_id not in _PIPELINES:
        from chronos import BaseChronosPipeline

        _PIPELINES[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32)
    return _PIPELINES[model_id]


class ChronosDetector(ErrorScore):
    def __init__(self, name, window, horizon, model_id, batch_size=1024, pipeline=None):
        self.name = name
        self.window, self.horizon = window, horizon
        self.model_id = model_id
        self.batch_size = batch_size
        self.pipeline = pipeline

    def fit(self, healthy, feature_names):
        self.feature_names = list(feature_names)
        self.channel_scale = self.healthy_channel_scale(healthy)
        return self

    def forecast(self, z, return_attention=False):
        """Median forecasts from every full window of z, in the layout of ForecastDetector.forecast."""
        z = np.ascontiguousarray(z, dtype=np.float32)
        key = (self.model_id, self.window, self.horizon, hashlib.sha1(z.tobytes()).hexdigest(), z.shape)
        if key not in _FORECASTS:
            pipeline = self.pipeline or load_pipeline(self.model_id)
            past = windows(z, self.window)[:-1]  # (n_windows, window, channels)
            n_windows, _, channels = past.shape
            series = torch.from_numpy(past.transpose(0, 2, 1).reshape(-1, self.window).copy())
            out = [pipeline.predict(series[i:i + self.batch_size], prediction_length=self.horizon)[:, MEDIAN]
                   for i in range(0, len(series), self.batch_size)]
            median = torch.cat(out).numpy() if out else np.empty((0, self.horizon), np.float32)
            _FORECASTS[key] = median.reshape(n_windows, channels, self.horizon).transpose(0, 2, 1)
        return _FORECASTS[key]
