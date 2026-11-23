"""
Decoder consumer: broker bytes -> validated readings in the database.

WHY THIS FILE EXISTS: this is the ground-station ingest loop. For every message it must
  1. decode the packet (or quarantine it if it is damaged),
  2. check the sequence count so loss, repeats and reordering are noticed,
  3. write readings idempotently,
  4. ONLY THEN tell the broker the message is done (commit).

Step 4 after step 3 gives at-least-once delivery; the idempotent write in step 3 turns that into
"effectively once". If the process dies between 3 and 4, the messages come back and the database
simply ignores the readings it already has.

Known limitation (stated in the README): sequence-tracker state lives in this process. After a
restart the first packet of each stream is treated as a new start, so a gap that falls exactly on
the restart boundary is not reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import get_settings
from ..packets.ccsds import IDLE_APID, PacketCRCError, PacketError, decode_packet
from ..packets.channels import by_apid
from ..packets.sequence import DUPLICATE, GAP, OUT_OF_ORDER, StreamTrackers

log = logging.getLogger(__name__)
EVENT_KINDS = {GAP, DUPLICATE, OUT_OF_ORDER}


@dataclass
class BatchStats:
    messages: int = 0
    decoded: int = 0
    readings_seen: int = 0
    readings_inserted: int = 0
    bad_packets: int = 0
    crc_errors: int = 0
    events_inserted: int = 0
    idle_skipped: int = 0

    def add(self, other: "BatchStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


class DecoderConsumer:
    def __init__(self, broker, store, registry: dict, topic: str | None = None, group: str | None = None,
                 bad_topic: str | None = None, batch_size: int = 500):
        s = get_settings()
        self.broker, self.store = broker, store
        self.topic = topic or s["topic_raw"]
        self.group = group or s["consumer_group"]
        self.bad_topic = bad_topic
        self.batch_size = batch_size
        self.registry = registry
        self._apid_to_channel = by_apid(registry)
        self.trackers = StreamTrackers()

    def process_batch(self, messages) -> BatchStats:
        stats = BatchStats(messages=len(messages))
        readings, events, bad = [], [], []
        for m in messages:
            try:
                pkt = decode_packet(m.value)
            except PacketCRCError as exc:
                stats.crc_errors += 1
                bad.append((m.value, "crc mismatch", exc.apid, exc.seq_count))
                continue
            except PacketError as exc:
                bad.append((m.value, f"malformed: {exc}", None, None))
                continue
            if pkt.apid == IDLE_APID:
                stats.idle_skipped += 1
                continue
            channel = self._apid_to_channel.get(pkt.apid)
            if channel is None:
                bad.append((m.value, f"unknown apid {pkt.apid}", pkt.apid, pkt.seq_count))
                continue
            if any(index != channel["index"] for index, _ in pkt.readings):
                bad.append((m.value, "channel index does not match apid", pkt.apid, pkt.seq_count))
                continue

            stats.decoded += 1
            ev = self.trackers.observe(pkt.apid, pkt.seq_count)
            if ev.kind in EVENT_KINDS:
                events.append((pkt.apid, pkt.seq_count, ev.kind, ev.gap, pkt.time_us))
            if ev.kind == DUPLICATE:
                continue  # second copy of a packet we already stored: nothing new to write
            for _, value in pkt.readings:
                readings.append((channel["channel_id"], pkt.time_us, value, 0))

        stats.readings_seen = len(readings)
        stats.readings_inserted = self.store.insert_readings(readings)
        stats.events_inserted = self.store.insert_packet_events(events)
        stats.bad_packets = len(bad)
        if bad:
            self.store.insert_bad_packets(bad)
            if self.bad_topic:
                for raw, *_ in bad:
                    self.broker.produce(self.bad_topic, b"bad", raw)
                self.broker.flush()
        self.broker.commit(self.topic, self.group, messages)  # LAST: only after everything above succeeded
        return stats

    def run_until_idle(self, max_idle_polls: int = 2, timeout_s: float = 0.5) -> BatchStats:
        """Consume until the topic has been quiet for `max_idle_polls` polls. Returns aggregated stats."""
        total, idle = BatchStats(), 0
        while idle < max_idle_polls:
            batch = self.broker.poll(self.topic, self.group, self.batch_size, timeout_s)
            if not batch:
                idle += 1
                continue
            idle = 0
            total.add(self.process_batch(batch))
        return total


def main() -> int:  # pragma: no cover - needs a live broker
    import time

    from ..broker import KafkaBroker
    from ..packets.channels import load_channel_registry
    from ..store.timescale import TimescaleStore

    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    store = TimescaleStore.from_url(s["database_url"])
    store.init_schema()
    registry = load_channel_registry("data/channels.json")
    store.upsert_channels(registry)
    consumer = DecoderConsumer(KafkaBroker(s["kafka_bootstrap"]), store, registry, bad_topic=s["topic_bad"])
    while True:
        stats = consumer.run_until_idle(max_idle_polls=1, timeout_s=1.0)
        if stats.messages:
            log.info("batch: %s", stats)
        else:
            time.sleep(0.5)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
