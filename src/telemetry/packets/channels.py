"""
Channel registry and limit checking.

WHY THIS FILE EXISTS: telemetry is meaningless without units and limits. The registry maps a
channel name ("P-1") to the numbers that travel in packets (APID and channel index) and to its
limit bands, so the consumer, API and dashboard all agree.

The SMAP/MSL arrays are normalised (roughly -1..1) and shipped without engineering units or
operational limits, so this project DERIVES limits from each channel's training data
(0.5th-99.5th percentile = nominal band; that band widened by 25% of its span = warning band).
That is a documented stand-in for the real limits a mission database would hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NOMINAL = "nominal"
WARNING = "warning"
CRITICAL = "critical"


def build_registry(channels: dict) -> dict:
    """Build a registry from {channel_id: {"spacecraft": str, "train": 1-D array}}.

    Channel index and APID are assigned in sorted-name order so they are stable between runs:
    index = position (0-based), APID = index + 1 (APID 0 is avoided; 2047 is reserved for idle).
    """
    if len(channels) > 2046:
        raise ValueError("too many channels for 11-bit APIDs")
    registry = {}
    for index, channel_id in enumerate(sorted(channels)):
        info = channels[channel_id]
        train = np.asarray(info["train"], dtype=float)
        lo, hi = (float(np.percentile(train, 0.5)), float(np.percentile(train, 99.5))) if train.size else (-1.0, 1.0)
        margin = 0.25 * (hi - lo)
        registry[channel_id] = {
            "channel_id": channel_id,
            "index": index,
            "apid": index + 1,
            "spacecraft": info.get("spacecraft", "unknown"),
            "group": channel_id.split("-")[0],  # prefix letter; its subsystem meaning is not published
            "unit": "normalized",
            "nominal_min": lo,
            "nominal_max": hi,
            "warn_min": lo - margin,
            "warn_max": hi + margin,
        }
    return registry


def save_registry(registry: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(registry, indent=1, sort_keys=True))


def load_channel_registry(path: str | Path) -> dict:
    """Load channel metadata (id, index, apid, unit, nominal and warning limits) from JSON."""
    registry = json.loads(Path(path).read_text())
    for channel_id, ch in registry.items():
        for key in ("index", "apid", "nominal_min", "nominal_max", "warn_min", "warn_max"):
            if key not in ch:
                raise ValueError(f"channel {channel_id} is missing '{key}'")
        if not (ch["warn_min"] <= ch["nominal_min"] <= ch["nominal_max"] <= ch["warn_max"]):
            raise ValueError(f"channel {channel_id}: limits must nest (warn outside nominal)")
    return registry


def limit_state(value: float, channel: dict) -> str:
    """Return 'nominal', 'warning' or 'critical' (NaN is critical: a missing value is never fine)."""
    if value != value:
        return CRITICAL
    if channel["nominal_min"] <= value <= channel["nominal_max"]:
        return NOMINAL
    if channel["warn_min"] <= value <= channel["warn_max"]:
        return WARNING
    return CRITICAL


def by_apid(registry: dict) -> dict:
    """Reverse lookup: APID -> channel record."""
    return {ch["apid"]: ch for ch in registry.values()}
