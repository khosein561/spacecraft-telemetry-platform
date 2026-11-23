"""The experiment harness itself (tiny, fast): wiring, no leakage, determinism."""
import numpy as np
import pandas as pd

from telemetry.anomaly.run_benchmark import evaluate_channel, run_benchmark, summarize, tune_zscore
from telemetry.anomaly.baseline import ZScoreDetector
from telemetry.replay.synthetic import generate_benchmark


def tiny_labels():
    return pd.DataFrame({
        "chan_id": ["T-1", "T-2"], "spacecraft": ["SMAP", "MSL"], "num_values": [1500, 1500],
        "sequences": [[(600, 760)], [(300, 380), (1000, 1100)]],
        "classes": [["point"], ["contextual", "point"]],
    })


def test_runner_end_to_end_on_tiny_synthetic_data(tmp_path):
    labels = tiny_labels()
    generate_benchmark(tmp_path, labels, seed=3)
    res = run_benchmark(tmp_path, labels, ["T-1", "T-2"], 60.0, {"epochs": 2, "hidden": 8}, {})
    assert res["channels"] == ["T-1", "T-2"]
    for det in ("zscore", "lstm"):
        s = res["summary"][det]
        assert s["true_sequences"] == 3 and s["channels"] == 2
        assert s["detected"] + s["missed"] == 3
        assert s["by_class"]["point"]["true"] == 2 and s["by_class"]["contextual"]["true"] == 1
    assert res["summary"]["zscore"]["channel_days"] == res["summary"]["lstm"]["channel_days"]   # same data for both


def test_labels_are_only_used_for_scoring(tmp_path):
    """Changing the labels must not change what a detector predicts, only how it is scored."""
    labels = tiny_labels()
    generate_benchmark(tmp_path, labels, seed=3)
    from telemetry.replay.dataset_loader import load_channel
    test, train = load_channel("T-2", "test", tmp_path), load_channel("T-2", "train", tmp_path)
    a = evaluate_channel("T-2", train, test, [(300, 380)], ["point"], {"zscore": ZScoreDetector()}, 60.0)
    b = evaluate_channel("T-2", train, test, [(900, 910)], ["point"], {"zscore": ZScoreDetector()}, 60.0)
    assert a["detectors"]["zscore"]["predicted"] == b["detectors"]["zscore"]["predicted"]


def test_tuning_returns_a_parameter_grid_and_best_by_f1(tmp_path):
    labels = tiny_labels()
    generate_benchmark(tmp_path, labels, seed=9)
    t = tune_zscore(tmp_path, labels, ["T-1", "T-2"], windows=(100, 200), thresholds=(3.0, 6.0))
    assert len(t["grid"]) == 4 and t["f1"] == max(g["f1"] for g in t["grid"])


def test_same_seed_same_results(tmp_path):
    labels = tiny_labels()
    generate_benchmark(tmp_path, labels, seed=4)
    r1 = run_benchmark(tmp_path, labels, ["T-1"], 60.0, {"epochs": 2, "hidden": 8}, {})
    r2 = run_benchmark(tmp_path, labels, ["T-1"], 60.0, {"epochs": 2, "hidden": 8}, {})
    for det in ("zscore", "lstm"):
        assert r1["per_channel"][0]["detectors"][det]["predicted"] == r2["per_channel"][0]["detectors"][det]["predicted"]
