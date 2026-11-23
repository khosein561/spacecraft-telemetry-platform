"""
Rolling z-score baseline detector.

WHY THIS FILE EXISTS: before reaching for a neural network you need a simple, explainable baseline
to beat. If an LSTM can't do better than "how many standard deviations is this from the recent
past?", it isn't earning its complexity.

score_t = |x_t - mean(previous W points)| / std(previous W points)

The window EXCLUDES the current point. (If the point were included in its own baseline, a big spike
would inflate the very mean/std used to judge it and hide itself.)

Two forms with identical results: `zscores` (vectorised, for evaluation) and `RollingZScore`
(one sample at a time, O(1), for a live stream). A test checks they agree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


def zscores(values, window: int = 200, min_periods: int = 50) -> np.ndarray:
    """Vectorised |z| of each point against the `window` points BEFORE it. 0 until `min_periods` exist."""
    s = pd.Series(np.asarray(values, dtype=float))
    prev = s.shift(1)
    mean = prev.rolling(window, min_periods=min_periods).mean()
    std = prev.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mean).abs() / (std + EPS)
    return z.fillna(0.0).to_numpy()


class RollingZScore:
    """Streaming version of `zscores` using running sums."""

    def __init__(self, window: int = 200, min_periods: int = 50):
        if window < 2 or not 1 <= min_periods <= window:
            raise ValueError("need window >= 2 and 1 <= min_periods <= window")
        self.window, self.min_periods = window, min_periods
        self._buf: deque = deque()
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, x: float) -> float:
        n = len(self._buf)
        if n >= self.min_periods:
            mean = self._sum / n
            var = max(self._sumsq / n - mean * mean, 0.0)
            score = abs(x - mean) / (var**0.5 + EPS)
        else:
            score = 0.0
        self._buf.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self._buf) > self.window:
            old = self._buf.popleft()
            self._sum -= old
            self._sumsq -= old * old
        return score


def flags_to_sequences(flags, merge_gap: int = 30) -> list:
    """Turn a boolean per-point array into (start, end) inclusive sequences.

    Flagged points separated by at most `merge_gap` unflagged points are merged into one sequence:
    an operator wants ONE incident, not forty alerts for one event.
    """
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    if idx.size == 0:
        return []
    seqs, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > merge_gap:
            seqs.append((start, prev))
            start = i
        prev = i
    seqs.append((start, prev))
    return seqs


@dataclass
class DetectionResult:
    sequences: list
    scores: np.ndarray  # per-point anomaly score (same length as the input)
    threshold: object = None


class ZScoreDetector:
    """Rolling z-score on the values (default) or on their first differences (`diff=True`).

    On a strongly periodic channel the rolling STANDARD DEVIATION of the values is large (the signal
    swings), so a small spike is only a fraction of a standard deviation and is never flagged. Scoring
    the first difference removes slow swings and makes sudden jumps stand out. Both are cheap, so the
    benchmark lets tuning choose between them: it keeps the baseline honest.
    """

    name = "zscore"

    def __init__(self, window: int = 400, threshold: float = 4.0, merge_gap: int = 30, min_periods: int = 50, diff: bool = False):
        self.window, self.threshold, self.merge_gap, self.min_periods, self.diff = window, threshold, merge_gap, min_periods, diff

    def _signal(self, x: np.ndarray) -> np.ndarray:
        return np.diff(x, prepend=x[:1]) if self.diff else x

    def detect(self, values, warmup=None) -> DetectionResult:
        """Score `values`. `warmup` (e.g. the tail of the training data) fills the window so the first
        test points are judged against real history instead of an empty buffer."""
        values = np.asarray(values, dtype=float)
        if warmup is not None and len(warmup):
            w = np.asarray(warmup, dtype=float)[-self.window :]
            full = np.concatenate([w, values])
            z = zscores(self._signal(full), self.window, self.min_periods)[len(w) :]
        else:
            z = zscores(self._signal(values), self.window, self.min_periods)
        return DetectionResult(flags_to_sequences(z > self.threshold, self.merge_gap), z, self.threshold)
