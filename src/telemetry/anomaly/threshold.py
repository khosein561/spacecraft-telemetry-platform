"""
Dynamic error thresholding (simplified from Hundman et al., "Detecting Spacecraft Anomalies Using
LSTMs and Nonparametric Dynamic Thresholding", KDD 2018).

WHY THIS FILE EXISTS: a forecaster produces a prediction ERROR at every step. Choosing a fixed
cut-off ("flag when error > 0.2") fails when channels have different noise levels or when noise
changes over time. The dynamic approach picks the threshold from the errors themselves:

  1. smooth the errors (EWMA) so one noisy step doesn't dominate;
  2. for each candidate threshold  eps = mean + z * std  (z from a range), measure how much the
     mean and std of the errors would DROP if the points above eps were removed, and penalise
     thresholds that flag many points / many separate sequences:

         score(z) = ( d_mean/mean + d_std/std ) / ( n_flagged + n_sequences^2 )

  3. keep the z with the best score, then PRUNE: a flagged sequence survives only if its peak error
     is meaningfully larger (by at least `min_pct_decrease`) than the next-lower peak.

No labels are used anywhere: the threshold is chosen from the errors alone. That matters: tuning
a threshold on the labelled test data would overstate performance.

This is a faithful-in-spirit, simplified implementation written for this project (it is not the
authors' code); differences are listed in docs/evaluation_report.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import flags_to_sequences


def smooth_errors(errors, span: int = 30) -> np.ndarray:
    """Exponentially weighted moving average of the raw prediction errors."""
    return pd.Series(np.asarray(errors, dtype=float)).ewm(span=span, adjust=False).mean().to_numpy()


def _best_z(e: np.ndarray, z_range) -> float:
    mu, sigma = e.mean(), e.std()
    if sigma == 0:
        return float(z_range[-1])
    best_z, best_score = float(z_range[-1]), -np.inf
    for z in z_range:
        eps = mu + z * sigma
        above = e > eps
        n_flagged = int(above.sum())
        if n_flagged == 0:
            continue
        below = e[~above]
        d_mu = mu - below.mean()
        d_sigma = sigma - below.std()
        n_seq = len(flags_to_sequences(above, merge_gap=0))
        score = (d_mu / mu + d_sigma / sigma) / (n_flagged + n_seq**2) if mu > 0 else -np.inf
        if score > best_score:
            best_z, best_score = float(z), score
    return best_z


def _prune(e: np.ndarray, seqs: list, eps: float, min_pct_decrease: float) -> list:
    """Drop sequences whose peak is not clearly above the rest (Hundman et al. pruning step)."""
    if not seqs:
        return []
    below = e[e <= eps]
    floor = float(below.max()) if below.size else 0.0
    peaks = sorted(((float(e[s : t + 1].max()), (s, t)) for s, t in seqs), reverse=True)
    values = [p for p, _ in peaks] + [floor]
    keep_until = -1
    for i in range(len(values) - 1):
        decrease = (values[i] - values[i + 1]) / values[i] if values[i] > 0 else 0.0
        if decrease >= min_pct_decrease:
            keep_until = i                       # the last big drop marks the boundary
    return sorted(seq for _, seq in peaks[: keep_until + 1])


def dynamic_threshold(errors, z_range=None, min_pct_decrease: float = 0.13, merge_gap: int = 30):
    """Return (threshold, sequences) for one segment of (already smoothed) errors."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return 0.0, []
    z_range = np.arange(2.5, 12.0, 0.5) if z_range is None else z_range
    z = _best_z(e, z_range)
    eps = float(e.mean() + z * e.std())
    seqs = flags_to_sequences(e > eps, merge_gap=merge_gap)
    return eps, _prune(e, seqs, eps, min_pct_decrease)


def segmented_threshold(errors, segment: int = 2000, **kwargs):
    """Apply `dynamic_threshold` to consecutive segments so the cut-off adapts as noise changes.

    Returns (threshold_per_point array, sequences). Sequences that touch across a segment boundary
    are merged. Segment shorter than 10% of `segment` at the tail is folded into the previous one.
    """
    e = np.asarray(errors, dtype=float)
    n = len(e)
    bounds = list(range(0, n, segment)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < segment // 10:
        bounds.pop(-2)
    thresholds = np.zeros(n)
    seqs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        eps, local = dynamic_threshold(e[a:b], **kwargs)
        thresholds[a:b] = eps
        seqs += [(s + a, t + a) for s, t in local]
    merged = []
    for s, t in sorted(seqs):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t))
        else:
            merged.append((s, t))
    return thresholds, merged
