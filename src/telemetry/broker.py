"""
Message-broker abstraction: one small interface, two implementations.

WHY THIS FILE EXISTS: the real deployment uses a Kafka-compatible broker (Redpanda). Tests and
laptops without Docker need something that behaves the same way. Both implementations expose:

    produce(topic, key, value)          send a message; the KEY decides the partition
    flush()                             wait until sent messages are acknowledged
    poll(topic, group, max_messages)    fetch messages for a consumer group
    commit(topic, group, messages)      record "I have fully processed these"

The delivery contract is AT-LEAST-ONCE: offsets are committed only AFTER the database write, so a
crash between write and commit redelivers messages. That is safe here because the database write
is idempotent (UNIQUE(channel_id, time) + ON CONFLICT DO NOTHING).

PARTITION KEY: we key by APID, so all packets of one channel go to one partition and keep their
order. Sequence-gap detection is only meaningful if per-channel order is preserved.

`InMemoryBroker` is fully tested. `KafkaBroker` is written against confluent-kafka and unit-tested
with fakes; it has NOT been run against a live Redpanda/Kafka in the environment where this was built.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    topic: str
    partition: int
    offset: int
    key: bytes
    value: bytes


class InMemoryBroker:
    """A tiny partitioned log with consumer-group offsets. Deterministic and dependency-free."""

    def __init__(self, partitions: int = 3):
        if partitions < 1:
            raise ValueError("need at least one partition")
        self.partitions = partitions
        self._logs: dict = {}        # (topic, partition) -> list[(key, value)]
        self._committed: dict = {}   # (topic, group, partition) -> next offset to read after restart
        self._position: dict = {}    # (topic, group, partition) -> next offset to hand out

    def _partition_for(self, key: bytes) -> int:
        # Kafka's default partitioner uses murmur2; any stable hash gives the same guarantee
        # (same key -> same partition) and keeps tests deterministic.
        return zlib.crc32(key) % self.partitions

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        p = self._partition_for(key)
        self._logs.setdefault((topic, p), []).append((key, value))

    def flush(self) -> None:
        return None

    def poll(self, topic: str, group: str, max_messages: int = 500, timeout_s: float = 0.0) -> list:
        out = []
        progressed = True
        while len(out) < max_messages and progressed:
            progressed = False
            for p in range(self.partitions):  # round-robin so partitions interleave like a real consumer
                log = self._logs.get((topic, p), [])
                pos = self._position.get((topic, group, p), self._committed.get((topic, group, p), 0))
                if pos < len(log) and len(out) < max_messages:
                    key, value = log[pos]
                    out.append(Message(topic, p, pos, key, value))
                    self._position[(topic, group, p)] = pos + 1
                    progressed = True
        return out

    def commit(self, topic: str, group: str, messages: list) -> None:
        for m in messages:
            k = (topic, group, m.partition)
            self._committed[k] = max(self._committed.get(k, 0), m.offset + 1)

    def simulate_consumer_restart(self, topic: str, group: str) -> None:
        """Forget in-flight (uncommitted) progress, as a crashed consumer would."""
        for p in range(self.partitions):
            self._position.pop((topic, group, p), None)

    def lag(self, topic: str, group: str) -> int:
        total = 0
        for p in range(self.partitions):
            total += len(self._logs.get((topic, p), [])) - self._committed.get((topic, group, p), 0)
        return total


class KafkaBroker:
    """Kafka/Redpanda implementation (confluent-kafka). Offsets are committed manually."""

    def __init__(self, bootstrap: str, producer=None, consumer_factory=None):
        self.bootstrap = bootstrap
        self._producer = producer
        self._consumers: dict = {}
        self._consumer_factory = consumer_factory

    def _get_producer(self):
        if self._producer is None:
            from confluent_kafka import Producer

            self._producer = Producer({"bootstrap.servers": self.bootstrap, "enable.idempotence": True, "acks": "all"})
        return self._producer

    def _get_consumer(self, topic: str, group: str):
        key = (topic, group)
        if key not in self._consumers:
            if self._consumer_factory is not None:
                consumer = self._consumer_factory(group)
            else:
                from confluent_kafka import Consumer

                consumer = Consumer(
                    {
                        "bootstrap.servers": self.bootstrap,
                        "group.id": group,
                        "enable.auto.commit": False,  # commit only after the database write succeeds
                        "auto.offset.reset": "earliest",
                    }
                )
            consumer.subscribe([topic])
            self._consumers[key] = consumer
        return self._consumers[key]

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        producer = self._get_producer()
        while True:
            try:
                producer.produce(topic, key=key, value=value)
                break
            except BufferError:  # local queue full: let delivery callbacks drain, then retry
                producer.poll(0.5)
        producer.poll(0)

    def flush(self) -> None:
        remaining = self._get_producer().flush(30)
        if remaining:
            raise RuntimeError(f"{remaining} messages were not delivered within 30 s")

    def poll(self, topic: str, group: str, max_messages: int = 500, timeout_s: float = 1.0) -> list:
        consumer = self._get_consumer(topic, group)
        out = []
        for msg in consumer.consume(num_messages=max_messages, timeout=timeout_s):
            if msg.error():
                raise RuntimeError(f"kafka error: {msg.error()}")
            out.append(Message(msg.topic(), msg.partition(), msg.offset(), msg.key() or b"", msg.value()))
        return out

    def commit(self, topic: str, group: str, messages: list) -> None:
        if not messages:
            return
        from confluent_kafka import TopicPartition

        highest: dict = {}
        for m in messages:
            highest[m.partition] = max(highest.get(m.partition, -1), m.offset)
        offsets = [TopicPartition(topic, p, off + 1) for p, off in highest.items()]  # Kafka commits the NEXT offset
        self._get_consumer(topic, group).commit(offsets=offsets, asynchronous=False)
