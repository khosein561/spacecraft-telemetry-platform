"""
Synthetic SMAP/MSL-shaped benchmark generator.

WHY THIS FILE EXISTS: the real NASA SMAP/MSL arrays must be downloaded by you (see data/README.md);
they are not available in every environment, and a pipeline needs something to run on from the
first minute. This module writes files in EXACTLY the same layout as the real dataset
(`train/<id>.npy`, `test/<id>.npy`, values in column 0) so every other module works unchanged on either.

WHAT IS REAL AND WHAT IS NOT:
  * REAL: channel ids, spacecraft, test-set lengths, and the anomaly index ranges and classes come
    from NASA's published `labeled_anomalies.csv`.
  * SYNTHETIC: the signal values. They are generated, so detection results on this data say
    NOTHING about performance on real spacecraft telemetry. They exist to prove the harness
    (packets -> ingest -> detectors -> scoring) is wired correctly and measurable.

Signals are periodic / drifting / stepwise with noise, scaled to roughly [-1, 1] using the training
split (as the real data is). Anomalies are injected only into the test split:
  'point'      a handful of short spikes inside the labelled range
  'contextual' one of: level shift, amplitude collapse, flat-line, or doubled oscillation rate.
               These keep values in a normal range, so a naive threshold may miss them.
"""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("periodic", "two_tone", "stepwise", "drift")
CONTEXTUAL_STYLES = ("level_shift", "amplitude_collapse", "flatline", "fast_oscillation")


def _rng_for(chan_id: str, seed: int) -> np.random.Generator:
    digest = zlib.crc32(f"{seed}:{chan_id}".encode())
    return np.random.default_rng(digest)


def _base_signal(rng, kind: str, n: int, p1: float, p2: float, phase: float) -> np.ndarray:
    t = np.arange(n, dtype=float)
    if kind == "periodic":
        return np.sin(2 * np.pi * t / p1 + phase)
    if kind == "two_tone":
        return 0.7 * np.sin(2 * np.pi * t / p1 + phase) + 0.4 * np.sin(2 * np.pi * t / p2)
    if kind == "stepwise":
        levels = rng.uniform(-0.8, 0.8, size=n // 120 + 2)
        return np.repeat(levels, 120)[:n] + 0.15 * np.sin(2 * np.pi * t / p1)
    drift = np.cumsum(rng.normal(0, 0.01, size=n))
    return 0.5 * np.sin(2 * np.pi * t / p1 + phase) + drift - drift.mean()


def _inject(rng, y: np.ndarray, start: int, end: int, klass: str, p1: float, noise: float) -> str:
    end = min(end, len(y) - 1)
    if end < start:
        return "none"
    if klass == "point":
        n_spikes = max(2, (end - start) // 40)
        for _ in range(n_spikes):
            pos = int(rng.integers(start, end + 1))
            width = int(rng.integers(1, 4))
            y[pos : pos + width] += rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
        return "spikes"
    style = str(rng.choice(CONTEXTUAL_STYLES))
    seg = slice(start, end + 1)
    if style == "level_shift":
        y[seg] += rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)
    elif style == "amplitude_collapse":
        y[seg] = y[seg].mean() + 0.15 * (y[seg] - y[seg].mean())
    elif style == "flatline":
        y[seg] = y[start]
    else:  # fast_oscillation: replace the segment with the same shape at double frequency
        t = np.arange(end - start + 1)
        y[seg] = y[seg].mean() + 0.6 * np.sin(2 * np.pi * t / (p1 / 2.0))
    return style


def make_channel(chan_id: str, num_test: int, sequences, classes, seed: int = 0, extra_features: int = 0):
    """Return (train, test, truth) for one channel. `truth` lists what was injected where."""
    rng = _rng_for(chan_id, seed)
    kind = str(rng.choice(KINDS))
    p1 = float(rng.uniform(70, 450))
    p2 = float(rng.uniform(25, 90))
    phase = float(rng.uniform(0, 2 * np.pi))
    noise = float(rng.uniform(0.01, 0.05))
    n_train = int(num_test * rng.uniform(0.8, 1.3))

    full = _base_signal(rng, kind, n_train + num_test, p1, p2, phase)
    full = full + rng.normal(0, noise, size=full.shape)
    train, test = full[:n_train].copy(), full[n_train:].copy()

    lo, hi = float(train.min()), float(train.max())
    span = (hi - lo) or 1.0
    train = 2 * (train - lo) / span - 1
    test = 2 * (test - lo) / span - 1
    noise_scaled = noise * 2 / span

    truth = []
    for (start, end), klass in zip(sequences, classes):
        style = _inject(rng, test, start, end, klass, p1, noise_scaled)
        truth.append({"start": int(start), "end": int(end), "class": klass, "style": style})

    def widen(a):
        if extra_features <= 0:
            return a.reshape(-1, 1).astype(np.float32)
        cmds = np.zeros((len(a), extra_features), dtype=np.float32)
        return np.concatenate([a.reshape(-1, 1).astype(np.float32), cmds], axis=1)

    return widen(train), widen(test), {"kind": kind, "period": p1, "noise": noise, "injected": truth}


def generate_benchmark(out_dir: str | Path, labels: pd.DataFrame, channels=None, seed: int = 0, extra_features: int = 0) -> dict:
    """Write train/test .npy files for the chosen channels (default: all labelled) and return the truth map."""
    out = Path(out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    rows = labels if channels is None else labels[labels["chan_id"].isin(list(channels))]
    truth = {}
    for row in rows.itertuples():
        train, test, info = make_channel(row.chan_id, int(row.num_values), row.sequences, row.classes, seed, extra_features)
        np.save(out / "train" / f"{row.chan_id}.npy", train)
        np.save(out / "test" / f"{row.chan_id}.npy", test)
        truth[row.chan_id] = info
    return truth


def benchmark_fingerprint(out_dir: str | Path) -> str:
    """SHA-256 over all generated arrays (sorted by name): proves two runs produced identical data."""
    h = hashlib.sha256()
    for p in sorted(Path(out_dir).rglob("*.npy")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
