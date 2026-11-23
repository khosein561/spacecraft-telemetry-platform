"""
Benchmark runner: z-score baseline vs LSTM forecaster, scored at the sequence level.

    python -m telemetry.anomaly.run_benchmark --data-dir data/raw --out results/real
    python -m telemetry.anomaly.run_benchmark --synthetic-seed 0 --out results/synthetic

WHY THIS FILE EXISTS: "I built an anomaly detector" is only believable with a repeatable
experiment: fixed seeds, the same channels for both detectors, labels used ONLY for scoring, and
a report that shows where the detectors fail. This script is that experiment.

Honesty rules baked in:
  * hyper-parameters are fixed BEFORE the evaluation run (z-score parameters may be tuned on a
    different synthetic seed via --tune-seed; nothing is ever tuned on the evaluation labels);
  * both detectors run on exactly the same channels;
  * every missed sequence is listed in the report, with its class.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..alerts.notify import AlertManager, Anomaly
from ..config import get_settings
from ..replay.dataset_loader import load_channel, load_labels
from ..replay.synthetic import benchmark_fingerprint, generate_benchmark
from .baseline import ZScoreDetector
from .evaluate import (
    aggregate_counts, detection_latencies, metrics_from_counts, overlaps, samples_to_days, sequence_counts,
)
from .forecast_model import LSTMDetector


def _alerts_after_dedupe(chan_id, detector_name, sequences, cadence_s, cooldown_s=1800.0):
    mgr = AlertManager(senders=[], cooldown_s=cooldown_s)
    step = int(cadence_s * 1_000_000)
    for s, e in sequences:
        mgr.handle(Anomaly(chan_id, detector_name, s * step, e * step))
    return mgr.sent + mgr.failed  # senders=[] -> 'failed' counts attempts that reached the manager's decision


def evaluate_channel(chan_id, train, test, true_seqs, classes, detectors: dict, cadence_s: float) -> dict:
    out = {"channel": chan_id, "n_test": int(len(test)), "days": samples_to_days(len(test), cadence_s),
           "true_sequences": [list(map(int, s)) for s in true_seqs], "classes": list(classes), "detectors": {}}
    for name, det in detectors.items():
        t0 = time.perf_counter()
        if hasattr(det, "fit"):
            det.fit(train)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        res = det.detect(test)
        counts = sequence_counts(res.sequences, true_seqs)
        lat = detection_latencies(res.sequences, true_seqs)
        missed = [{"sequence": list(map(int, t)), "class": c} for t, c in zip(true_seqs, classes)
                  if not any(overlaps(p, t) for p in res.sequences)]
        out["detectors"][name] = {
            "counts": counts,
            "predicted": [list(map(int, s)) for s in res.sequences],
            "latency_samples": lat,
            "missed": missed,
            "alerts_after_dedupe": _alerts_after_dedupe(chan_id, name, res.sequences, cadence_s),
            "fit_seconds": round(fit_s, 3),
            "detect_seconds": round(time.perf_counter() - t1, 3),
        }
        out["detectors"][name]["_scores"] = res.scores  # kept in memory for plots; stripped before saving JSON
    return out


def summarize(per_channel: list, detector: str, cadence_s: float) -> dict:
    counts = aggregate_counts([c["detectors"][detector]["counts"] for c in per_channel])
    m = metrics_from_counts(counts)
    days = sum(c["days"] for c in per_channel)
    lat = [x for c in per_channel for x in c["detectors"][detector]["latency_samples"]]
    by_class = {}
    for klass in ("point", "contextual"):
        total = detected = 0
        for c in per_channel:
            missed = {tuple(x["sequence"]) for x in c["detectors"][detector]["missed"]}
            for seq, k in zip(c["true_sequences"], c["classes"]):
                if k == klass:
                    total += 1
                    detected += tuple(seq) not in missed
        by_class[klass] = {"true": total, "detected": detected}
    return {
        **m,
        "channels": len(per_channel),
        "channel_days": round(days, 3),
        "false_alarms_per_channel_day": m["false_alarms"] / days if days else None,
        "alerts_after_dedupe": sum(c["detectors"][detector]["alerts_after_dedupe"] for c in per_channel),
        "mean_latency_samples": float(np.mean(lat)) if lat else None,
        "mean_latency_minutes": float(np.mean(lat)) * cadence_s / 60 if lat else None,
        "by_class": by_class,
    }


def run_benchmark(data_dir, labels, channels, cadence_s: float = 60.0, lstm_kwargs=None, zscore_kwargs=None,
                  progress=None) -> dict:
    lstm_kwargs, zscore_kwargs = lstm_kwargs or {}, zscore_kwargs or {}
    per_channel = []
    for i, chan in enumerate(channels):
        row = labels[labels["chan_id"] == chan].iloc[0]
        train, test = load_channel(chan, "train", data_dir), load_channel(chan, "test", data_dir)
        detectors = {"zscore": ZScoreDetector(**zscore_kwargs), "lstm": LSTMDetector(**lstm_kwargs)}
        per_channel.append(evaluate_channel(chan, train, test, row.sequences, row.classes, detectors, cadence_s))
        if progress:
            progress(i + 1, len(channels), chan)
    return {
        "cadence_s": cadence_s,
        "channels": list(channels),
        "summary": {d: summarize(per_channel, d, cadence_s) for d in ("zscore", "lstm")},
        "per_channel": per_channel,
    }


def tune_zscore(data_dir, labels, channels, windows=(100, 200, 400), thresholds=(3.0, 4.0, 5.0, 6.0, 8.0)) -> dict:
    """Pick z-score parameters on a TUNING dataset (a different synthetic seed than the evaluation one)
    by F1 at the sequence level. Returns {'window','threshold','f1', 'grid': [...]}."""
    grid = []
    for w in windows:
        for thr in thresholds:
            cs = []
            for chan in channels:
                row = labels[labels["chan_id"] == chan].iloc[0]
                res = ZScoreDetector(window=w, threshold=thr).detect(load_channel(chan, "test", data_dir))
                cs.append(sequence_counts(res.sequences, row.sequences))
            m = metrics_from_counts(aggregate_counts(cs))
            grid.append({"window": w, "threshold": thr, "f1": m["f1"] or 0.0, "precision": m["precision"], "recall": m["recall"]})
    best = max(grid, key=lambda g: g["f1"])
    return {"window": best["window"], "threshold": best["threshold"], "f1": best["f1"], "grid": grid}


def plot_channel(path, chan_id, test, true_seqs, per_detector: dict, title_suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1 + len(per_detector), 1, figsize=(11, 2.2 * (1 + len(per_detector))), sharex=True)
    axes[0].plot(test, lw=0.6, color="#1f2937")
    axes[0].set_ylabel("value")
    axes[0].set_title(f"{chan_id} {title_suffix}".strip(), fontsize=10)
    for ax in axes:
        for s, e in true_seqs:
            ax.axvspan(s, e, color="#f59e0b", alpha=0.25, lw=0)
    for ax, (name, (seqs, scores)) in zip(axes[1:], per_detector.items()):
        ax.plot(scores, lw=0.5, color="#2563eb")
        for s, e in seqs:
            ax.axvspan(s, e, ymin=0.0, ymax=0.12, color="#dc2626", alpha=0.9, lw=0)
        ax.set_ylabel(f"{name} score")
    axes[-1].set_xlabel("sample index (amber = labelled anomaly, red tick = detector alert)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def results_table(summary: dict) -> str:
    rows = ["| Detector | Precision | Recall | F1 | False alarms / channel-day | Mean latency (min) | Detected / true sequences |",
            "|---|---|---|---|---|---|---|"]
    for name in ("zscore", "lstm"):
        s = summary[name]
        rows.append(f"| {name} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
                    f"{_fmt(s['false_alarms_per_channel_day'], 2)} | {_fmt(s['mean_latency_minutes'], 1)} | "
                    f"{s['detected']} / {s['true_sequences']} |")
    return "\n".join(rows)


def save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(results, default=lambda o: None))  # drop numpy score arrays (not JSON)
    for c in clean["per_channel"]:
        for d in c["detectors"].values():
            d.pop("_scores", None)
    (out_dir / "results.json").write_text(json.dumps(clean, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="directory holding train/ and test/ (real SMAP/MSL data)")
    ap.add_argument("--synthetic-seed", type=int, help="generate and use the synthetic benchmark with this seed")
    ap.add_argument("--tune-seed", type=int, help="synthetic seed used ONLY to pick z-score parameters")
    ap.add_argument("--channels", nargs="*", help="channel ids (default: --max-channels spread across both spacecraft)")
    ap.add_argument("--max-channels", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=None, help="override the tuned default (16)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    s = get_settings()
    labels = load_labels()
    out = Path(args.out)
    work = out / "_data"
    if args.synthetic_seed is not None:
        generate_benchmark(work, labels, seed=args.synthetic_seed)
        data_dir, kind = work, f"synthetic (seed {args.synthetic_seed})"
    elif args.data_dir:
        data_dir, kind = Path(args.data_dir), "real SMAP/MSL"
    else:
        ap.error("give --data-dir (real data) or --synthetic-seed")
    from ..replay.dataset_loader import available_channels

    avail = available_channels(data_dir, labels)
    if args.channels:
        channels = args.channels
    else:  # spread evenly over the sorted list so both spacecraft and many subsystems are included
        step = max(1, len(avail) // args.max_channels)
        channels = avail[::step][: args.max_channels]

    zkw = {}
    tuning = None
    if args.tune_seed is not None:
        tdir = out / "_tune_data"
        generate_benchmark(tdir, labels, channels, seed=args.tune_seed)
        tuning = tune_zscore(tdir, labels, channels)
        zkw = {"window": tuning["window"], "threshold": tuning["threshold"]}
        print(f"z-score tuned on synthetic seed {args.tune_seed}: window={zkw['window']} threshold={zkw['threshold']} (F1 {tuning['f1']:.3f})")

    t0 = time.perf_counter()
    results = run_benchmark(data_dir, labels, channels, s["cadence_s"], ({"epochs": args.epochs} if args.epochs else {}), zkw,
                            progress=lambda i, n, c: print(f"  [{i}/{n}] {c}", flush=True))
    results.update({"dataset": kind, "zscore_params": zkw or "defaults", "tuning": tuning, "wall_seconds": round(time.perf_counter() - t0, 1),
                    "python": platform.python_version(), "dataset_sha256": benchmark_fingerprint(data_dir)})
    save_results(results, out)
    print(results_table(results["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
