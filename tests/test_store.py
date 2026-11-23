"""Store behaviour against a real PostgreSQL."""
import numpy as np
from helpers import START_US


def test_insert_readings_is_idempotent_and_counts_inserted_rows(store):
    rows = [("A-1", START_US + i * 1_000_000, float(i), 0) for i in range(10)]
    assert store.insert_readings(rows) == 10
    assert store.insert_readings(rows) == 0                       # replay: nothing inserted
    assert store.insert_readings(rows + [("A-1", START_US + 99_000_000, 1.0, 0)]) == 1
    assert store.count("telemetry_readings") == 11


def test_same_timestamp_different_channels_are_separate_rows(store):
    assert store.insert_readings([("A-1", START_US, 1.0, 0), ("B-2", START_US, 2.0, 0)]) == 2


def test_microsecond_timestamps_round_trip_exactly(store):
    t = START_US + 123_456                                         # 0.123456 s past the second
    store.insert_readings([("A-1", t, 0.5, 0)])
    times, values = store.read_channel("A-1")
    assert int(times[0]) == t and values[0] == 0.5


def test_read_channel_range_and_order(store):
    rows = [("A-1", START_US + i * 60_000_000, float(i), 0) for i in (5, 1, 3, 2, 4)]  # inserted out of order
    store.insert_readings(rows)
    times, values = store.read_channel("A-1")
    assert list(values) == [1, 2, 3, 4, 5]
    t2, v2 = store.read_channel("A-1", START_US + 2 * 60_000_000, START_US + 4 * 60_000_000)
    assert list(v2) == [2, 3, 4]


def test_event_and_bad_packet_inserts_are_idempotent(store):
    ev = [(7, 100, "gap", 2, START_US), (7, 105, "duplicate", 0, START_US + 5)]
    assert store.insert_packet_events(ev) == 2 and store.insert_packet_events(ev) == 0
    bad = [(b"\x00\x01\x02", "malformed", None, None)]
    assert store.insert_bad_packets(bad) == 1 and store.insert_bad_packets(bad) == 0


def test_anomaly_records_are_idempotent(store):
    times = START_US + np.arange(100, dtype=np.int64) * 60_000_000
    assert store.record_anomalies("A-1", "zscore", [(10, 20), (50, 60)], times, scores=[3.1, 4.2], labeled=[True, False]) == 2
    assert store.record_anomalies("A-1", "zscore", [(10, 20)], times) == 0


def test_count_rejects_unknown_table(store):
    import pytest
    with pytest.raises(ValueError):
        store.count("pg_user; DROP TABLE x")
