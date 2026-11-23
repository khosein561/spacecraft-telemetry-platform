"""End to end: replay with injected faults -> broker -> consumer -> database. Detected must equal injected."""
import pytest

from telemetry.broker import InMemoryBroker
from telemetry.consume.decoder_consumer import DecoderConsumer
from telemetry.packets.ccsds import SEQ_MODULO, decode_packet
from telemetry.replay.faults import FaultPlan, inject_faults
from telemetry.replay.producer import DEFAULT_SEQ_START, build_packets, replay_channel
from telemetry.replay.dataset_loader import assign_timestamps
from helpers import START_US, signal

N = 3000
CADENCE = 60.0


def run(registry, store, plan, n=N, channel="A-1", broker=None, seed=0):
    broker = broker or InMemoryBroker(partitions=3)
    values = signal(n, seed)
    published, truth, clean = replay_channel(broker, "telemetry.raw", registry, channel, values, START_US, CADENCE, plan)
    consumer = DecoderConsumer(broker, store, registry, batch_size=200)
    stats = consumer.run_until_idle()
    return values, truth, consumer, stats, published


def test_clean_stream_stores_every_reading_exactly(registry, store):
    values, truth, consumer, stats, published = run(registry, store, FaultPlan(seed=1))
    assert stats.readings_inserted == N == store.count("telemetry_readings")
    times, got = store.read_channel("A-1")
    assert (got == values).all()                                   # float32 on the wire, exact for float32 inputs
    assert list(times[:3]) == [START_US + i * 60_000_000 for i in range(3)]
    t = consumer.trackers.tracker(registry["A-1"]["apid"])
    assert (t.gaps, t.lost, t.duplicates, t.out_of_order) == (0, 0, 0, 0)
    assert store.count("packet_events") == 0 and store.count("bad_packets") == 0


def test_stream_crosses_the_14_bit_wraparound_without_false_gaps(registry, store):
    """3000 packets starting at 16300 wrap past 16383 -> 0. A correct tracker reports no gaps."""
    broker = InMemoryBroker()
    values = signal(N)
    ch = registry["A-1"]
    pk = build_packets(ch["apid"], ch["index"], values, assign_timestamps(N, START_US, CADENCE), DEFAULT_SEQ_START)
    seqs = [decode_packet(p).seq_count for p in pk]
    assert 16383 in seqs and 0 in seqs[seqs.index(16383):]        # it really wraps
    assert max(seqs) == SEQ_MODULO - 1
    run(registry, store, FaultPlan())
    assert store.packet_event_counts() == {}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_detected_faults_equal_injected_faults(registry, store, seed):
    plan = FaultPlan(seed=seed, drop_rate=0.03, corrupt_rate=0.02, dup_rate=0.02, reorder_rate=0.03)
    values, truth, consumer, stats, _ = run(registry, store, plan)
    t = consumer.trackers.tracker(registry["A-1"]["apid"])

    assert truth.dropped > 0 and truth.corrupted > 0 and truth.duplicated > 0 and truth.reordered > 0
    assert stats.crc_errors + (stats.bad_packets - stats.crc_errors) == truth.corrupted   # every bit flip caught
    assert stats.bad_packets == truth.corrupted
    assert t.lost == truth.never_delivered                         # dropped + corrupted never arrive
    assert t.duplicates == truth.duplicated
    assert t.out_of_order == truth.reordered                       # each swap = 1 gap + 1 late arrival
    assert store.count("telemetry_readings") == N - truth.never_delivered   # reordered/duplicated are not lost
    assert store.count("bad_packets") <= truth.corrupted           # identical garbage would be stored once


def test_reordered_packets_still_land_with_the_right_values(registry, store):
    plan = FaultPlan(seed=5, reorder_rate=0.2)
    values, truth, consumer, _, _ = run(registry, store, plan)
    times, got = store.read_channel("A-1")
    assert len(got) == N and (got == values).all()                 # ORDER BY time restores order


def test_events_recorded_in_database(registry, store):
    plan = FaultPlan(seed=2, drop_rate=0.05, dup_rate=0.05)
    _, truth, _, _, _ = run(registry, store, plan)
    counts = store.packet_event_counts()
    assert counts["duplicate"] == truth.duplicated
    assert counts["gap"] >= 1


def test_two_channels_are_tracked_independently(registry, store):
    broker = InMemoryBroker(partitions=2)
    for ch, seed, plan in (("A-1", 0, FaultPlan(seed=1, drop_rate=0.05)), ("B-2", 1, FaultPlan(seed=2))):
        replay_channel(broker, "telemetry.raw", registry, ch, signal(800, seed), START_US, CADENCE, plan)
    consumer = DecoderConsumer(broker, store, registry)
    consumer.run_until_idle()
    assert consumer.trackers.tracker(registry["A-1"]["apid"]).lost > 0
    assert consumer.trackers.tracker(registry["B-2"]["apid"]).lost == 0


def test_unknown_apid_and_garbage_go_to_dead_letter(registry, store):
    broker = InMemoryBroker()
    from telemetry.packets.ccsds import encode_packet
    broker.produce("telemetry.raw", b"\x01", encode_packet(1500, 0, START_US, [(0, 1.0)]))      # apid not in registry
    broker.produce("telemetry.raw", b"\x01", b"\x00\x01\x02")                                    # garbage
    broker.produce("telemetry.raw", b"\x01", encode_packet(2047, 0, START_US, []))                # idle packet
    stats = DecoderConsumer(broker, store, registry).run_until_idle()
    assert stats.bad_packets == 2 and stats.idle_skipped == 1 and stats.readings_inserted == 0
    assert store.count("bad_packets") == 2


def test_fault_injection_is_deterministic_and_validated():
    pk = [bytes([i]) * 20 for i in range(200)]
    plan = FaultPlan(seed=9, drop_rate=0.1, dup_rate=0.1, reorder_rate=0.1, corrupt_rate=0.1)
    assert inject_faults(pk, plan) == inject_faults(pk, plan)
    with pytest.raises(ValueError):
        FaultPlan(drop_rate=0.7, dup_rate=0.7)
