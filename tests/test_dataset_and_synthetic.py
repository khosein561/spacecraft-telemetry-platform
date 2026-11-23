"""Dataset loader, labels (real NASA file), synthetic benchmark and channel registry."""
import numpy as np
import pytest

from telemetry.packets.channels import build_registry, by_apid, limit_state, load_channel_registry, save_registry
from telemetry.replay.dataset_loader import (
    assign_timestamps, available_channels, load_channel, load_labels, parse_anomaly_sequences,
)
from telemetry.replay.synthetic import benchmark_fingerprint, generate_benchmark, make_channel


def test_real_label_file_parses():
    labels = load_labels()
    assert len(labels) == 82 and set(labels["spacecraft"]) == {"SMAP", "MSL"}
    assert (labels["spacecraft"] == "SMAP").sum() == 55
    for row in labels.itertuples():
        assert len(row.sequences) == len(row.classes) >= 1
        assert all(0 <= s <= e < row.num_values for s, e in row.sequences), row.chan_id
        assert set(row.classes) <= {"point", "contextual"}


def test_parse_anomaly_sequences_validates_and_never_executes_code():
    assert parse_anomaly_sequences("[[1, 5], [9, 9]]") == [(1, 5), (9, 9)]
    for bad in ("[[5, 1]]", "[[1]]", "[[-1, 4]]", "[['a', 2]]"):
        with pytest.raises(ValueError):
            parse_anomaly_sequences(bad)
    with pytest.raises((ValueError, SyntaxError)):
        parse_anomaly_sequences("__import__('os').system('echo pwned')")


def test_assign_timestamps_exact_integers():
    t = assign_timestamps(5, 1_000, 0.5)
    assert t.dtype == np.int64 and list(t) == [1000, 501000, 1001000, 1501000, 2001000]
    big = assign_timestamps(1_000_000, 0, 60.0)
    assert big[-1] == 999_999 * 60_000_000                       # no float drift over a million samples
    with pytest.raises(ValueError):
        assign_timestamps(3, 0, 0)


def test_synthetic_benchmark_has_the_real_dataset_layout(tmp_path):
    labels = load_labels()
    ids = ["P-1", "S-1", "E-2"]
    truth = generate_benchmark(tmp_path, labels, ids, seed=7, extra_features=25)
    assert available_channels(tmp_path, labels) == sorted(ids)
    raw = np.load(tmp_path / "test" / "P-1.npy")
    assert raw.shape[1] == 26 and raw.shape[0] == 8505 and raw.dtype == np.float32       # same shape idea as the real files
    test = load_channel("P-1", "test", tmp_path)                  # loader takes feature 0 only
    assert test.shape == (8505,)
    train = load_channel("P-1", "train", tmp_path)
    assert -1.001 <= train.min() and train.max() <= 1.001        # scaled like the real data
    assert len(truth["P-1"]["injected"]) == 3


def test_synthetic_is_deterministic_and_seed_dependent(tmp_path):
    labels = load_labels()
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    generate_benchmark(a, labels, ["P-1", "T-1"], seed=1)
    generate_benchmark(b, labels, ["P-1", "T-1"], seed=1)
    generate_benchmark(c, labels, ["P-1", "T-1"], seed=2)
    assert benchmark_fingerprint(a) == benchmark_fingerprint(b) != benchmark_fingerprint(c)


def test_anomalies_change_the_signal_only_inside_labelled_ranges():
    seqs, classes = [(300, 420), (800, 830)], ["contextual", "point"]
    _, clean, _ = make_channel("X-1", 1200, [], [], seed=4)
    _, dirty, info = make_channel("X-1", 1200, seqs, classes, seed=4)
    diff = np.flatnonzero(clean[:, 0] != dirty[:, 0])
    assert len(diff) > 0
    inside = np.zeros(1200, dtype=bool)
    for s, e in seqs:
        inside[s : e + 3] = True                                  # spikes may be up to 3 samples wide
    assert inside[diff].all()
    assert {i["class"] for i in info["injected"]} == {"contextual", "point"}


def test_loader_rejects_path_tricks_and_missing_files(tmp_path):
    for bad in ("../x", "a/b", "..\\x"):
        with pytest.raises(ValueError):
            load_channel(bad, "test", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_channel("NOPE-1", "test", tmp_path)
    with pytest.raises(ValueError):
        load_channel("P-1", "validation", tmp_path)


def test_loader_never_unpickles(tmp_path):
    (tmp_path / "test").mkdir()
    np.save(tmp_path / "test" / "EVIL-1.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_channel("EVIL-1", "test", tmp_path)


def test_registry_is_stable_and_limits_nest(tmp_path):
    rng = np.random.default_rng(1)
    chans = {n: {"spacecraft": "SMAP", "train": rng.normal(0, 0.3, 400)} for n in ("T-1", "A-1", "P-2")}
    reg = build_registry(chans)
    assert [reg[k]["index"] for k in ("A-1", "P-2", "T-1")] == [0, 1, 2]
    assert [reg[k]["apid"] for k in ("A-1", "P-2", "T-1")] == [1, 2, 3]
    path = tmp_path / "channels.json"
    save_registry(reg, path)
    assert load_channel_registry(path) == reg
    assert by_apid(reg)[2]["channel_id"] == "P-2"


def test_limit_states():
    ch = {"nominal_min": -1.0, "nominal_max": 1.0, "warn_min": -1.5, "warn_max": 1.5}
    assert limit_state(0.0, ch) == "nominal" and limit_state(1.0, ch) == "nominal"
    assert limit_state(1.2, ch) == "warning" and limit_state(-1.5, ch) == "warning"
    assert limit_state(1.6, ch) == "critical" and limit_state(float("nan"), ch) == "critical"


def test_bad_registry_file_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"X": {"index": 0, "apid": 1, "nominal_min": 0, "nominal_max": 1, "warn_min": 0.5, "warn_max": 2}}')
    with pytest.raises(ValueError):
        load_channel_registry(p)
