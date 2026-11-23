"""Broker behaviour: partition keys keep per-channel order; Kafka adapter commits the NEXT offset."""
from telemetry.broker import InMemoryBroker, KafkaBroker, Message


def test_same_key_goes_to_one_partition_and_keeps_order():
    b = InMemoryBroker(partitions=4)
    for i in range(100):
        b.produce("t", b"chan-A", f"A{i}".encode())
        b.produce("t", b"chan-B", f"B{i}".encode())
    msgs = []
    while True:
        batch = b.poll("t", "g", 17)
        if not batch:
            break
        msgs += batch
    a = [m for m in msgs if m.key == b"chan-A"]
    assert len({m.partition for m in a}) == 1
    assert [m.value for m in a] == [f"A{i}".encode() for i in range(100)]


def test_uncommitted_messages_come_back_after_restart():
    b = InMemoryBroker(partitions=1)
    for i in range(5):
        b.produce("t", b"k", bytes([i]))
    first = b.poll("t", "g", 10)
    b.commit("t", "g", first[:3])
    b.simulate_consumer_restart("t", "g")
    again = b.poll("t", "g", 10)
    assert [m.value for m in again] == [bytes([3]), bytes([4])]


def test_groups_are_independent():
    b = InMemoryBroker(partitions=1)
    b.produce("t", b"k", b"x")
    assert len(b.poll("t", "g1", 10)) == 1 and len(b.poll("t", "g2", 10)) == 1


# ---- Kafka adapter, tested with fakes (a live broker was NOT available where this was written) ----
class FakeProducer:
    def __init__(self):
        self.sent, self.fail_first = [], 0

    def produce(self, topic, key=None, value=None):
        if self.fail_first:
            self.fail_first -= 1
            raise BufferError("queue full")
        self.sent.append((topic, key, value))

    def poll(self, t):
        return 0

    def flush(self, t):
        return 0


class FakeKafkaMsg:
    def __init__(self, p, o, k, v):
        self._a = (p, o, k, v)

    def error(self):
        return None

    def topic(self):
        return "t"

    def partition(self):
        return self._a[0]

    def offset(self):
        return self._a[1]

    def key(self):
        return self._a[2]

    def value(self):
        return self._a[3]


class FakeConsumer:
    def __init__(self):
        self.commits = []
        self.queue = [FakeKafkaMsg(0, 10, b"k", b"a"), FakeKafkaMsg(0, 11, b"k", b"b"), FakeKafkaMsg(1, 4, b"j", b"c")]

    def subscribe(self, topics):
        self.topics = topics

    def consume(self, num_messages, timeout):
        out, self.queue = self.queue[:num_messages], self.queue[num_messages:]
        return out

    def commit(self, offsets, asynchronous):
        self.commits.append((offsets, asynchronous))


def test_kafka_adapter_produces_retries_on_full_queue_and_polls():
    prod = FakeProducer()
    prod.fail_first = 2
    kb = KafkaBroker("x:9092", producer=prod, consumer_factory=lambda g: FakeConsumer())
    kb.produce("t", b"k", b"v")
    kb.flush()
    assert prod.sent == [("t", b"k", b"v")]
    msgs = kb.poll("t", "g", 10)
    assert msgs == [Message("t", 0, 10, b"k", b"a"), Message("t", 0, 11, b"k", b"b"), Message("t", 1, 4, b"j", b"c")]


def test_kafka_adapter_commits_next_offset_per_partition_synchronously():
    cons = FakeConsumer()
    kb = KafkaBroker("x:9092", producer=FakeProducer(), consumer_factory=lambda g: cons)
    msgs = kb.poll("t", "g", 10)
    kb.commit("t", "g", msgs)
    offsets, asynchronous = cons.commits[0]
    assert asynchronous is False
    assert {(o.partition, o.offset) for o in offsets} == {(0, 12), (1, 5)}   # highest offset + 1
