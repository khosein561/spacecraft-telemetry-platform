"""Sequence-level scoring."""
import pytest

from telemetry.anomaly.evaluate import (
    aggregate_counts, detection_latencies, detection_latency, false_alarms_per_day, metrics_from_counts,
    overlaps, samples_to_days, sequence_counts, sequence_metrics,
)


def test_overlap_is_inclusive():
    assert overlaps((0, 5), (5, 9)) and overlaps((5, 9), (0, 5))
    assert not overlaps((0, 4), (5, 9))


def test_perfect_detection():
    m = sequence_metrics([(10, 20), (50, 60)], [(12, 15), (55, 58)])
    assert (m["detected"], m["missed"], m["false_alarms"]) == (2, 0, 0)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0


def test_miss_and_false_alarm():
    m = sequence_metrics([(10, 20), (100, 110)], [(12, 15), (55, 58)])
    assert (m["detected"], m["missed"], m["false_alarms"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_no_double_credit_and_one_prediction_can_cover_two_truths():
    many = sequence_counts([(10, 11), (12, 13), (14, 15)], [(10, 15)])
    assert many["detected"] == 1 and many["false_alarms"] == 0
    wide = sequence_counts([(0, 100)], [(10, 20), (60, 70)])
    assert wide["detected"] == 2 and wide["false_alarms"] == 0


def test_undefined_ratios_are_none_not_zero_or_one():
    nothing_predicted = sequence_metrics([], [(1, 2)])
    assert nothing_predicted["precision"] is None and nothing_predicted["recall"] == 0.0
    nothing_true = sequence_metrics([(1, 2)], [])
    assert nothing_true["recall"] is None and nothing_true["precision"] == 0.0


def test_aggregation_sums_counts_before_dividing():
    a = sequence_counts([(1, 2)], [(1, 2)])            # 1 detected
    b = sequence_counts([], [(5, 6)])                   # 1 missed, nothing predicted
    c = sequence_counts([(9, 9)], [])                   # 1 false alarm
    m = metrics_from_counts(aggregate_counts([a, b, c]))
    assert (m["detected"], m["missed"], m["false_alarms"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_false_alarms_per_day():
    assert false_alarms_per_day([(0, 1), (5, 6), (50, 51)], [(5, 6)], total_duration_days=2.0) == 1.0
    with pytest.raises(ValueError):
        false_alarms_per_day([], [], 0)


def test_latency_counts_from_true_start_and_clamps_early_alerts():
    assert detection_latencies([(15, 30)], [(10, 40)]) == [5]
    assert detection_latencies([(2, 30)], [(10, 40)]) == [0]          # alert began before the event: not late
    assert detection_latencies([(15, 16), (12, 13)], [(10, 40)]) == [2]   # earliest overlapping alert wins
    assert detection_latencies([], [(10, 40)]) == []
    assert detection_latency([(15, 30), (60, 70)], [(10, 40), (55, 80)]) == (5 + 5) / 2
    assert detection_latency([], [(1, 2)]) is None


def test_samples_to_days():
    assert samples_to_days(1440, 60) == 1.0


def test_pointwise_accuracy_would_have_been_misleading():
    """Why we score sequences: a detector that never fires has 99%+ point accuracy and recall 0."""
    n, true = 10_000, [(5000, 5050)]
    point_accuracy = 1 - sum(e - s + 1 for s, e in true) / n
    assert point_accuracy > 0.99
    assert sequence_metrics([], true)["recall"] == 0.0
