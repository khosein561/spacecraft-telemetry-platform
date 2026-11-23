"""Rolling z-score baseline."""
import numpy as np
import pytest

from telemetry.anomaly.baseline import RollingZScore, ZScoreDetector, flags_to_sequences, zscores


def test_streaming_matches_vectorised():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 600)
    x[300] += 12
    z_vec = zscores(x, window=100, min_periods=30)
    det = RollingZScore(window=100, min_periods=30)
    z_str = np.array([det.update(v) for v in x])
    assert np.allclose(z_vec, z_str, atol=1e-6)


def test_spike_is_flagged_and_noise_is_not():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 1000)
    x[700] += 15
    res = ZScoreDetector(window=200, threshold=5.0).detect(x)
    assert res.sequences == [(700, 700)]


def test_window_excludes_the_current_point():
    """If the point were in its own baseline a huge spike would hide itself. It must score very high."""
    x = np.zeros(300)
    x[250] = 100.0
    assert zscores(x, window=100, min_periods=50)[250] > 1e6


def test_warmup_removes_cold_start():
    rng = np.random.default_rng(2)
    train, test = rng.normal(0, 1, 500), rng.normal(0, 1, 300)
    test[5] += 20                                                  # spike right at the start of the test data
    cold = ZScoreDetector(window=200, min_periods=50, threshold=6.0).detect(test)
    warm = ZScoreDetector(window=200, min_periods=50, threshold=6.0).detect(test, warmup=train)
    assert cold.sequences == [] and warm.sequences == [(5, 5)]     # only the warm start can judge point 5


def test_flags_merge_across_small_gaps_only():
    flags = np.zeros(200, dtype=bool)
    flags[[10, 12, 15, 100, 101]] = True
    assert flags_to_sequences(flags, merge_gap=5) == [(10, 15), (100, 101)]
    assert flags_to_sequences(flags, merge_gap=0) == [(10, 10), (12, 12), (15, 15), (100, 101)]
    assert flags_to_sequences(np.zeros(10, dtype=bool)) == []


def test_constant_signal_does_not_divide_by_zero():
    z = zscores(np.ones(300))
    assert np.isfinite(z).all() and z.max() < 1e-3


def test_bad_parameters():
    with pytest.raises(ValueError):
        RollingZScore(window=10, min_periods=20)


def test_difference_mode_catches_a_spike_the_level_mode_misses():
    t = np.arange(2000)
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * t / 300) + rng.normal(0, 0.01, 2000)     # strong slow swing, tiny noise
    x[1500] += 0.5                                                  # modest spike: well under one std of the swing
    level = ZScoreDetector(window=200, threshold=4.0).detect(x)
    diff = ZScoreDetector(window=200, threshold=4.0, diff=True).detect(x)
    assert all(not (s <= 1500 <= e) for s, e in level.sequences)    # swallowed by the swing
    assert any(s <= 1500 <= e for s, e in diff.sequences)
