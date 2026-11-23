#!/usr/bin/env python3
"""Re-run both detectors (same defaults) on chosen channels and save diagnostic figures.

    python scripts/make_figures.py --data-dir results/synthetic_seed0/_data --channels D-11 A-8 --out docs/figures

Include the failures, not only the wins: they are what show you understand the detector.
"""
import argparse
from pathlib import Path

from telemetry.anomaly.baseline import ZScoreDetector
from telemetry.anomaly.forecast_model import LSTMDetector
from telemetry.anomaly.run_benchmark import plot_channel
from telemetry.replay.dataset_loader import load_channel, load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    labels = load_labels()
    for c in a.channels:
        row = labels[labels.chan_id == c].iloc[0]
        train, test = load_channel(c, "train", a.data_dir), load_channel(c, "test", a.data_dir)
        z = ZScoreDetector().detect(test)
        lstm = LSTMDetector()
        lstm.fit(train)
        l = lstm.detect(test)
        plot_channel(Path(a.out) / f"{c}.png", c, test, row.sequences, {"zscore": (z.sequences, z.scores), "lstm": (l.sequences, l.scores)},
                     title_suffix=f"(synthetic data; class: {', '.join(row.classes)})")
        print("wrote", Path(a.out) / f"{c}.png")


if __name__ == "__main__":
    main()
