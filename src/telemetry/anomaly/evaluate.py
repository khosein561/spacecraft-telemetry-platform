"""
Sequence-level evaluation.

WHY THIS FILE EXISTS: point-by-point accuracy is misleading for anomalies. If 1% of points are
anomalous, a detector that never fires is "99% accurate". Operators care about EVENTS: did the
detector notice the incident, how soon, and how many false alarms did it cost them?

Overlap rule (stated in the README too):
  * sequences are INCLUSIVE index ranges (start, end) into the test array;
  * a TRUE sequence is DETECTED if at least one predicted sequence overlaps it;
  * a PREDICTED sequence is a FALSE ALARM if it overlaps no true sequence;
  * several predictions on one true sequence count once (no double credit), and one long
    prediction covering two true sequences detects both.

    precision = detected_true / (detected_true + false_alarms)
    recall    = detected_true / (detected_true + missed_true)
"""

from __future__ import annotations


def overlaps(a, b) -> bool:
    """Do inclusive ranges a=(s,e) and b=(s,e) share at least one index?"""
    return a[0] <= b[1] and b[0] <= a[1]


def sequence_counts(pred_seqs, true_seqs) -> dict:
    pred_seqs, true_seqs = list(pred_seqs), list(true_seqs)
    detected = sum(1 for t in true_seqs if any(overlaps(p, t) for p in pred_seqs))
    false_alarms = sum(1 for p in pred_seqs if not any(overlaps(p, t) for t in true_seqs))
    return {
        "true_sequences": len(true_seqs),
        "predicted_sequences": len(pred_seqs),
        "detected": detected,
        "missed": len(true_seqs) - detected,
        "false_alarms": false_alarms,
    }


def _ratio(num, den):
    return None if den == 0 else num / den


def metrics_from_counts(c: dict) -> dict:
    out = dict(c)
    out["precision"] = _ratio(c["detected"], c["detected"] + c["false_alarms"])
    out["recall"] = _ratio(c["detected"], c["true_sequences"])
    p, r = out["precision"], out["recall"]
    out["f1"] = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
    return out


def sequence_metrics(pred_seqs, true_seqs) -> dict:
    """Precision / recall / F1 on sequences for ONE channel (None where undefined, e.g. no predictions)."""
    return metrics_from_counts(sequence_counts(pred_seqs, true_seqs))


def aggregate_counts(per_channel_counts) -> dict:
    """Micro-average: add up the raw counts across channels, THEN compute ratios (a channel with no
    predictions must not silently count as precision 1.0 or 0.0)."""
    total = {"true_sequences": 0, "predicted_sequences": 0, "detected": 0, "missed": 0, "false_alarms": 0}
    for c in per_channel_counts:
        for k in total:
            total[k] += c[k]
    return total


def false_alarms_per_day(pred_seqs, true_seqs, total_duration_days: float) -> float:
    """False-alarm sequences divided by observed days. Operators feel this as alert fatigue."""
    if total_duration_days <= 0:
        raise ValueError("duration must be positive")
    return sequence_counts(pred_seqs, true_seqs)["false_alarms"] / total_duration_days


def detection_latencies(pred_seqs, true_seqs) -> list:
    """For each DETECTED true sequence: samples from its start to the first alert point inside it.

    An alert that began before the true sequence started counts as latency 0 (it was early, not late).
    """
    pred_seqs = list(pred_seqs)
    out = []
    for t in true_seqs:
        firsts = [max(p[0], t[0]) for p in pred_seqs if overlaps(p, t)]
        if firsts:
            out.append(min(firsts) - t[0])
    return out


def detection_latency(pred_seqs, true_seqs):
    """Mean latency in samples over detected sequences, or None if nothing was detected."""
    lat = detection_latencies(pred_seqs, true_seqs)
    return None if not lat else sum(lat) / len(lat)


def samples_to_days(n_samples: int, cadence_s: float) -> float:
    return n_samples * cadence_s / 86400.0
