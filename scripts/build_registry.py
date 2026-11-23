#!/usr/bin/env python3
"""Build data/channels.json (APID + nominal/warn limits per channel) from the TRAIN split.

    python scripts/build_registry.py --data-dir data/raw --out data/channels.json
    python scripts/build_registry.py --synthetic-seed 0 --out data/channels.json   # demo data

Limits come from the training data only (p0.5-p99.5 = nominal band), so nothing is learned
from the test split.
"""
import argparse
from pathlib import Path

from telemetry.packets.channels import build_registry, save_registry
from telemetry.replay.dataset_loader import available_channels, load_channel, load_labels
from telemetry.replay.synthetic import generate_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--synthetic-seed", type=int)
    ap.add_argument("--out", default="data/channels.json")
    ap.add_argument("--max-channels", type=int, default=0, help="0 = all channels found")
    a = ap.parse_args()
    labels = load_labels()
    if a.synthetic_seed is not None:
        data_dir = Path("data/synthetic")
        generate_benchmark(data_dir, labels, seed=a.synthetic_seed)
    elif a.data_dir:
        data_dir = Path(a.data_dir)
    else:
        ap.error("give --data-dir or --synthetic-seed")
    chans = available_channels(data_dir, labels)
    if a.max_channels:
        chans = chans[: a.max_channels]
    spec = {}
    for c in chans:
        row = labels[labels.chan_id == c].iloc[0]
        spec[c] = {"spacecraft": row.spacecraft, "train": load_channel(c, "train", data_dir)}
    save_registry(build_registry(spec), a.out)
    print(f"wrote {a.out} with {len(spec)} channels")


if __name__ == "__main__":
    main()
