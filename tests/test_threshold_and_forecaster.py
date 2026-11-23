"""Dynamic thresholding and the LSTM forecaster (small, fast sanity tests - not benchmarks)."""
import numpy as np
import pytest

from telemetry.anomaly.evaluate import sequence_metrics
from telemetry.anomaly.forecast_model import LSTMDetector, make_windows, prediction_errors, fit_forecaster
from telemetry.anomaly.threshold import dynamic_threshold, segmented_threshold, smooth_errors


def noisy_errors(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.05, 0.01, n))


def test_threshold_finds_an_obvious_error_burst_and_nothing_else():
    e = noisy_errors()
    e[1500:1540] += 0.6
    eps, seqs = dynamic_threshold(smooth_errors(e))
    assert len(seqs) == 1 and seqs[0][0] <= 1520 <= seqs[0][1] and eps > 0.06


def test_quiet_errors_produce_no_alerts():
    for seed in range(5):
        _, seqs = dynamic_threshold(smooth_errors(noisy_errors(seed=seed)))
        assert seqs == [], f"false alarms on pure noise (seed {seed}): {seqs}"


def test_threshold_adapts_when_the_noise_level_changes():
    quiet = noisy_errors(2000, 1)
    loud = noisy_errors(2000, 2) * 4                              # 4x noisier second half, still no anomaly
    thr, seqs = segmented_threshold(smooth_errors(np.concatenate([quiet, loud])), segment=2000)
    assert thr[100] < thr[2500] and seqs == []


def test_empty_and_constant_inputs():
    assert dynamic_threshold([]) == (0.0, [])
    _, seqs = dynamic_threshold(np.full(500, 0.1))
    assert seqs == []


def test_segments_merge_across_boundaries():
    e = noisy_errors(4000)
    e[1980:2030] += 0.8                                           # burst straddles the segment boundary at 2000
    _, seqs = segmented_threshold(smooth_errors(e), segment=2000)
    assert len(seqs) == 1


def test_make_windows_shapes_and_alignment():
    X, y = make_windows(np.arange(10.0), 3)
    assert X.shape == (7, 3, 1) and y.shape == (7,)
    assert list(X[0, :, 0]) == [0, 1, 2] and y[0] == 3 and list(X[-1, :, 0]) == [6, 7, 8] and y[-1] == 9
    with pytest.raises(ValueError):
        make_windows(np.arange(3.0), 3)


def sine(n, period=80.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * np.arange(n) / period) + rng.normal(0, noise, n)


def test_forecaster_learns_a_sine_wave():
    train = sine(2500)
    model, report = fit_forecaster(train, length=32, hidden=16, epochs=6, seed=0, device="cpu")
    err = prediction_errors(model, sine(600, seed=1), 32)
    assert np.isnan(err[:32]).all() and not np.isnan(err[32:]).any()
    assert err[32:].mean() < 0.12                                       # a sine with 0.02 noise: error should be small
    assert report.val_loss < 0.05


def test_lstm_detector_finds_an_injected_frequency_change_and_is_deterministic():
    train = sine(2500, seed=0)
    test = sine(2500, seed=1)
    test[1200:1400] = np.sin(2 * np.pi * np.arange(200) / 25.0)    # oscillation 3x faster: contextual anomaly
    truth = [(1200, 1399)]
    runs = []
    for _ in range(2):
        det = LSTMDetector(length=32, hidden=16, epochs=6, seed=0, smoothing_span=30, device="cpu")
        det.fit(train)
        runs.append(det.detect(test))
    assert runs[0].sequences == runs[1].sequences                  # same seed -> same result
    m = sequence_metrics(runs[0].sequences, truth)
    assert m["detected"] == 1 and m["false_alarms"] == 0


def test_detect_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        LSTMDetector().detect(np.zeros(10))
