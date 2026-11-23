"""Sequence tracking: gaps, wraparound, duplicates, reordering."""
import pytest

from telemetry.packets.sequence import (
    DUPLICATE, FIRST, GAP, IN_ORDER, OUT_OF_ORDER, SequenceTracker, StreamTrackers,
)


def feed(seqs):
    t = SequenceTracker()
    return t, [t.observe(s) for s in seqs]


def test_first_packet_is_just_first():
    _, ev = feed([100])
    assert ev[0].kind == FIRST


def test_in_order_stream_has_no_events():
    t, ev = feed([5, 6, 7, 8])
    assert [e.kind for e in ev] == [FIRST, IN_ORDER, IN_ORDER, IN_ORDER]
    assert t.lost == 0 and t.gaps == 0


def test_gap_detected():
    t, ev = feed([5, 8])
    assert ev[1].kind == GAP and ev[1].gap == 2          # 6 and 7 never arrived
    assert t.lost == 2


def test_wraparound_is_not_a_gap():
    t, ev = feed([16382, 16383, 0, 1])
    assert all(e.kind != GAP for e in ev) and t.lost == 0
    assert ev[2].kind == IN_ORDER                        # 16383 -> 0 is the normal next packet


def test_gap_across_wraparound_is_measured_modulo():
    _, ev = feed([16382, 2])                             # skipped 16383, 0, 1
    assert ev[1].kind == GAP and ev[1].gap == 3


def test_duplicate_and_out_of_order():
    _, ev = feed([5, 5])
    assert ev[1].kind == DUPLICATE
    _, ev = feed([8, 6])
    assert ev[1].kind == OUT_OF_ORDER


def test_late_arrival_fills_the_hole_so_nothing_is_lost():
    t, ev = feed([5, 8, 6, 7])
    assert [e.kind for e in ev] == [FIRST, GAP, OUT_OF_ORDER, OUT_OF_ORDER]
    assert all(e.filled_gap for e in ev[2:])
    assert t.lost == 0 and t.out_of_order == 2


def test_swapped_pair_is_one_gap_one_late_arrival():
    t, _ = feed([1, 2, 4, 3, 5])
    assert (t.gaps, t.out_of_order, t.duplicates, t.lost) == (1, 1, 0, 0)


def test_duplicate_of_a_late_arrival_is_still_a_duplicate():
    t, ev = feed([1, 3, 2, 2])
    assert ev[3].kind == DUPLICATE and t.lost == 0


def test_repeat_a_few_packets_later_is_recognised():
    t, ev = feed([1, 2, 3, 4, 2])
    assert ev[4].kind == DUPLICATE


def test_large_gap_is_counted_without_remembering_every_number():
    t, ev = feed([0, 6000])
    assert ev[1].gap == 5999 and t.lost == 5999
    assert len(t.missing) == 0                           # beyond MAX_TRACKED_GAP: counted, not remembered


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        SequenceTracker().observe(16384)


def test_each_apid_has_its_own_counter():
    s = StreamTrackers()
    s.observe(1, 10)
    s.observe(2, 500)
    assert s.observe(1, 11).kind == IN_ORDER
    assert s.observe(2, 501).kind == IN_ORDER
    assert s.totals()["received"] == 4


def test_full_wrap_cycle_twice_has_no_false_alarms():
    t = SequenceTracker()
    for i in range(16384 * 2 + 10):
        e = t.observe((16000 + i) % 16384)
    assert t.gaps == 0 and t.lost == 0 and t.duplicates == 0
