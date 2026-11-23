"""Replays and crashes must never create duplicate rows (Definition of Done: replays are idempotent)."""
from telemetry.broker import InMemoryBroker
from telemetry.consume.decoder_consumer import DecoderConsumer
from telemetry.replay.faults import FaultPlan
from telemetry.replay.producer import replay_channel
from helpers import START_US, signal

N = 1500
PLAN = FaultPlan(seed=3, drop_rate=0.03, corrupt_rate=0.02, dup_rate=0.03, reorder_rate=0.03)


def snapshot(store):
    return {t: store.count(t) for t in ("telemetry_readings", "packet_events", "bad_packets")}


def test_replaying_the_same_recording_twice_adds_nothing(registry, store):
    values = signal(N)
    for round_ in (1, 2):
        broker = InMemoryBroker()
        replay_channel(broker, "telemetry.raw", registry, "A-1", values, START_US, 60.0, PLAN)
        stats = DecoderConsumer(broker, store, registry).run_until_idle()
        if round_ == 1:
            first, first_stats = snapshot(store), stats
        else:
            assert stats.readings_inserted == 0 and stats.readings_seen == first_stats.readings_seen
    assert snapshot(store) == first                                   # identical row counts after the second replay


def test_same_stream_delivered_twice_through_one_broker(registry, store, engine):
    """A sender that retries the whole recording (every packet arrives twice) ends in the same database."""
    from sqlalchemy import text

    values = signal(N)
    twice = InMemoryBroker()
    for _ in range(2):
        replay_channel(twice, "telemetry.raw", registry, "A-1", values, START_US, 60.0, PLAN)
    DecoderConsumer(twice, store, registry).run_until_idle()
    after_twice = store.count("telemetry_readings")
    _, got_twice = store.read_channel("A-1")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE telemetry_readings, packet_events, bad_packets RESTART IDENTITY"))
    once = InMemoryBroker()
    replay_channel(once, "telemetry.raw", registry, "A-1", values, START_US, 60.0, PLAN)
    DecoderConsumer(once, store, registry).run_until_idle()
    _, got_once = store.read_channel("A-1")

    assert after_twice == store.count("telemetry_readings")
    assert (got_twice == got_once).all()


def test_crash_after_database_write_but_before_commit_redelivers_safely(registry, store):
    """At-least-once: the consumer writes a batch, 'dies' before committing, restarts, and sees the batch again."""
    broker = InMemoryBroker(partitions=2)
    values = signal(N)
    replay_channel(broker, "telemetry.raw", registry, "A-1", values, START_US, 60.0, FaultPlan(seed=1))
    consumer = DecoderConsumer(broker, store, registry, batch_size=300)

    batch = broker.poll("telemetry.raw", consumer.group, 300)
    # simulate: process (writes to the DB) but crash before broker.commit
    real_commit, calls = broker.commit, []
    broker.commit = lambda *a, **k: calls.append(1)                   # the commit never reaches the broker
    consumer.process_batch(batch)
    broker.commit = real_commit
    assert store.count("telemetry_readings") == 300 and calls == [1]

    broker.simulate_consumer_restart("telemetry.raw", consumer.group)  # uncommitted progress is forgotten
    restarted = DecoderConsumer(broker, store, registry, batch_size=300)   # fresh process, fresh tracker
    restarted.run_until_idle()
    assert store.count("telemetry_readings") == N                     # everything arrived exactly once
    assert broker.lag("telemetry.raw", consumer.group) == 0
    _, got = store.read_channel("A-1")
    assert (got == values).all()


def test_commit_only_after_write_succeeds(registry):
    """If the database write raises, the offset must NOT be committed (so the batch is retried)."""
    class ExplodingStore:
        def insert_readings(self, rows):
            raise RuntimeError("database down")
        insert_packet_events = insert_bad_packets = lambda self, rows: 0

    broker = InMemoryBroker()
    replay_channel(broker, "telemetry.raw", registry, "A-1", signal(50), START_US, 60.0)
    consumer = DecoderConsumer(broker, ExplodingStore(), registry)
    batch = broker.poll("telemetry.raw", consumer.group, 100)
    try:
        consumer.process_batch(batch)
    except RuntimeError:
        pass
    assert broker.lag("telemetry.raw", consumer.group) == 50          # nothing committed
