"""Baseline detectors. Each one is fitted on healthy z-scored features and returns
one raw anomaly score per snapshot (higher means more anomalous)."""
import numpy as np
from pyod.models.ecod import ECOD


class FeatureThreshold:
    """The largest z-score of one feature (e.g. RMS) over the sensor axes.

    Only an increase counts: vibration getting weaker is not a sign of wear.
    """

    def __init__(self, feature):
        self.feature = feature
        self.name = feature

    def fit(self, healthy, feature_names):
        self.columns = [i for i, n in enumerate(feature_names) if n.endswith(f"_{self.feature}")]
        return self

    def score(self, z, stop=None):
        return z[:stop, self.columns].max(axis=1)


class ECODDetector:
    """ECOD (Li et al. 2022, PyOD) fitted on the pooled healthy snapshots."""

    name = "ecod"

    def fit(self, healthy, feature_names):
        self.model = ECOD().fit(np.concatenate(healthy))
        return self

    def score(self, z, stop=None):
        # PyOD's decision_function rebuilds the empirical distributions from the
        # training data plus the batch being scored, so scoring a whole record at
        # once would let later snapshots (including the failure) shape earlier
        # scores. One snapshot at a time keeps the score causal.
        return np.array([self.model.decision_function(row[None])[0] for row in z[:stop]])


DETECTORS = {
    "rms": lambda: FeatureThreshold("rms"),
    "kurtosis": lambda: FeatureThreshold("kurtosis"),
    "ecod": ECODDetector,
}
