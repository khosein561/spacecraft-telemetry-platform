#!/usr/bin/env python3
"""Choose detector hyper-parameters on a TUNING dataset, never on the evaluation labels.

    python scripts/tune_hyperparameters.py --synthetic-seed 100 --out results/tuning
    python scripts/tune_hyperparameters.py --data-dir data/raw --channels A-1 B-1 ... --out results/tuning

Why a separate script: if you pick thresholds by looking at the labelled test anomalies you will
report flattering numbers that do not generalise. Tune on one set of channels (or a different
synthetic seed), then run `run_benchmark` ONCE on channels that were never used for tuning.

To keep it fast the LSTM is trained once per model configuration; the smoothing and pruning
settings (which only post-process the prediction errors) are swept on the cached errors.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from telemetry.anomaly.baseline import ZScoreDetector
from telemetry.anomaly.evaluate import aggregate_counts, metrics_from_counts, sequence_counts
from telemetry.anomaly.forecast_model import fit_forecaster, prediction_errors
from telemetry.anomaly.threshold import segmented_threshold, smooth_errors
from telemetry.replay.dataset_loader import available_channels, load_channel, load_labels
from telemetry.replay.synthetic import generate_benchmark

MODEL_GRID = [{"length": L, "hidden": h, "epochs": e} for L, h, e in itertools.product((32, 64), (32, 64), (8, 16))]
POST_GRID = [{"span": s, "pct": p} for s, p in itertools.product((5, 15, 30), (0.05, 0.13, 0.25))]
Z_GRID = [{"window": w, "threshold": t, "diff": d} for w, t, d in itertools.product((100, 200, 400), (2.0, 3.0, 4.0, 6.0, 8.0), (False, True))]


def score(counts):
    m = metrics_from_counts(aggregate_counts(counts))
    return {"f1": m["f1"] or 0.0, "precision": m["precision"], "recall": m["recall"], "false_alarms": m["false_alarms"], "detected": m["detected"], "true": m["true_sequences"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-seed", type=int)
    ap.add_argument("--data-dir")
    ap.add_argument("--channels", nargs="*")
    ap.add_argument("--n-channels", type=int, default=16)
    ap.add_argument("--out", default="results/tuning")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    if a.synthetic_seed is not None:
        data_dir = out / "_data"
        generate_benchmark(data_dir, labels, seed=a.synthetic_seed)
    else:
        data_dir = Path(a.data_dir)
    avail = available_channels(data_dir, labels)
    channels = a.channels or avail[1:: max(1, len(avail) // a.n_channels)][: a.n_channels]   # offset so it differs from run_benchmark's default picks
    print(f"tuning on {len(channels)} channels: {channels}")
    true = {c: labels[labels.chan_id == c].iloc[0].sequences for c in channels}
    test = {c: load_channel(c, "test", data_dir) for c in channels}
    train = {c: load_channel(c, "train", data_dir) for c in channels}

    z_results = []
    for cfg in Z_GRID:
        cs = [sequence_counts(ZScoreDetector(**cfg).detect(test[c]).sequences, true[c]) for c in channels]
        z_results.append({**cfg, **score(cs)})
    best_z = max(z_results, key=lambda r: (r["f1"], -r["false_alarms"]))

    lstm_results = []
    for mcfg in MODEL_GRID:
        errors = {}
        for c in channels:
            model, _ = fit_forecaster(train[c], mcfg["length"], mcfg["hidden"], mcfg["epochs"], seed=0)
            e = prediction_errors(model, test[c], mcfg["length"])
            e[: mcfg["length"]] = np.nanmedian(e)
            errors[c] = e
        for pcfg in POST_GRID:
            cs = []
            for c in channels:
                _, seqs = segmented_threshold(smooth_errors(errors[c], pcfg["span"]), 2000, min_pct_decrease=pcfg["pct"])
                cs.append(sequence_counts(seqs, true[c]))
            lstm_results.append({**mcfg, **pcfg, **score(cs)})
        print(f"  model {mcfg} done", flush=True)
    best_l = max(lstm_results, key=lambda r: (r["f1"], -r["false_alarms"]))

    result = {"channels": channels, "zscore_best": best_z, "lstm_best": best_l, "zscore_grid": z_results, "lstm_grid": lstm_results}
    (out / "tuning.json").write_text(json.dumps(result, indent=1))
    print("BEST zscore:", {k: best_z[k] for k in ("window", "threshold", "diff", "f1", "precision", "recall")})
    print("BEST lstm  :", {k: best_l[k] for k in ("length", "hidden", "epochs", "span", "pct", "f1", "precision", "recall")})


if __name__ == "__main__":
    main()
