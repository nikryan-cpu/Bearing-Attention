import numpy as np

from bearing_attention.baselines import ECODDetector, FeatureThreshold

NAMES = ["horiz_rms", "horiz_kurtosis", "vert_rms", "vert_kurtosis"]


def test_feature_threshold_takes_the_larger_axis():
    z = np.array([[1.0, 9, 2, 9], [-3, 0, -1, 0]])
    detector = FeatureThreshold("rms").fit([], NAMES)
    assert detector.score(z).tolist() == [2, -1]
    assert detector.score(z, stop=1).tolist() == [2]


def test_ecod_scores_do_not_depend_on_later_snapshots():
    rng = np.random.default_rng(0)
    detector = ECODDetector().fit([rng.normal(size=(300, 4))], NAMES)
    record = rng.normal(size=(20, 4))
    record[15:] += 8  # a failure at the end
    full = detector.score(record)
    np.testing.assert_allclose(detector.score(record, stop=15), full[:15])
    assert full[15:].min() > full[:15].max()
