"""
Load SMAP/MSL arrays and labels.

WHY THIS FILE EXISTS: the project replays recorded spacecraft telemetry. This module hides the
dataset's file layout (from JPL's `telemanom` project, Hundman et al., KDD 2018):

    <data_dir>/train/<channel>.npy   shape (n_timesteps, n_features)
    <data_dir>/test/<channel>.npy    same n_features
    labeled_anomalies.csv            channel id, spacecraft, anomaly index ranges, class, length

Feature 0 of each array is the telemetry value; the other columns are one-hot encodings of
commands sent to the spacecraft, which this project does not use.

THE TIMESTAMP ASSUMPTION: the arrays contain no clock. `assign_timestamps` creates one from a
start time and a fixed cadence. The default (60 s) is an ASSUMPTION used for converting sample
counts into durations (alerts per day, latency in minutes). It is not a property of the data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings


def parse_anomaly_sequences(text: str) -> list:
    """Turn the CSV string '[[2149, 2349], [4536, 4844]]' into [(2149, 2349), (4536, 4844)].

    Uses ast.literal_eval (safe: it only accepts literals, never runs code) and validates shape.
    """
    value = ast.literal_eval(text)
    pairs = []
    for item in value:
        if len(item) != 2 or not all(isinstance(x, int) for x in item):
            raise ValueError(f"bad anomaly sequence {item!r}")
        start, end = item
        if start < 0 or end < start:
            raise ValueError(f"anomaly sequence must satisfy 0 <= start <= end, got {item!r}")
        pairs.append((start, end))
    return pairs


def load_labels(path: str | Path | None = None) -> pd.DataFrame:
    """Parse labeled_anomalies.csv into one row per channel with `sequences` as (start, end) pairs.

    Columns: chan_id, spacecraft, sequences (list of (start, end) inclusive test-set indices),
    classes (list of 'point'/'contextual'), num_values (length of the test array).
    """
    path = Path(path or get_settings()["labels_path"])
    raw = pd.read_csv(path)
    required = {"chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "chan_id": raw["chan_id"],
            "spacecraft": raw["spacecraft"],
            "sequences": raw["anomaly_sequences"].map(parse_anomaly_sequences),
            "classes": raw["class"].map(lambda s: [c.strip() for c in s.strip("[]").split(",")]),
            "num_values": raw["num_values"].astype(int),
        }
    )
    return out


def _channel_path(channel_id: str, split: str, data_dir: str | Path | None) -> Path:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if "/" in channel_id or "\\" in channel_id or ".." in channel_id:
        raise ValueError(f"unsafe channel id {channel_id!r}")
    return Path(data_dir or get_settings()["data_dir"]) / split / f"{channel_id}.npy"


def load_channel(channel_id: str, split: str = "test", data_dir: str | Path | None = None) -> np.ndarray:
    """Load the telemetry values (feature 0) for one channel as a 1-D float array."""
    path = _channel_path(channel_id, split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to download the dataset.")
    arr = np.load(path, allow_pickle=False)  # never unpickle untrusted files
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"{path}: expected shape (n_timesteps, n_features), got {arr.shape}")
    return arr[:, 0].astype(float)


def available_channels(data_dir: str | Path | None = None, labels: pd.DataFrame | None = None) -> list:
    """Channel ids that have both a train and a test file (and a label row, if labels are given)."""
    base = Path(data_dir or get_settings()["data_dir"])
    train = {p.stem for p in (base / "train").glob("*.npy")}
    test = {p.stem for p in (base / "test").glob("*.npy")}
    ids = train & test
    if labels is not None:
        ids &= set(labels["chan_id"])
    return sorted(ids)


def assign_timestamps(n: int, start_us: int, cadence_s: float) -> np.ndarray:
    """Return n spacecraft times in integer microseconds since the epoch, `cadence_s` apart.

    Integers (not floats) so that millions of samples never accumulate rounding drift.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if cadence_s <= 0:
        raise ValueError("cadence must be positive")
    step = int(round(cadence_s * 1_000_000))
    return start_us + np.arange(n, dtype=np.int64) * step
