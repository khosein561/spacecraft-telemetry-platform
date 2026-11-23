"""
Replay producer: recorded telemetry -> CCSDS packets -> broker.

WHY THIS FILE EXISTS: it plays the role of the spacecraft and the radio link. It reads one channel
of the dataset, wraps each sample in a space packet, optionally damages the stream (drops, bit
flips, repeats, swaps), and publishes the bytes to the broker keyed by APID.

Sequence numbers start at a configurable offset (default: near the 14-bit limit) so that
wraparound happens inside every demo instead of once in a blue moon.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from ..broker import InMemoryBroker, KafkaBroker
from ..config import get_settings
from ..packets.ccsds import SEQ_MODULO, encode_packet
from ..packets.channels import build_registry
from .dataset_loader import assign_timestamps, load_channel
from .faults import FaultPlan, FaultTruth, inject_faults

DEFAULT_SEQ_START = 16300  # 84 packets before the wrap, so every channel of >84 samples crosses it


def build_packets(apid: int, channel_index: int, values, times_us, seq_start: int = 0) -> list:
    """One packet per sample. Sequence counts increment per packet and wrap at 16384."""
    if len(values) != len(times_us):
        raise ValueError("values and times_us must have the same length")
    return [
        encode_packet(apid, (seq_start + i) % SEQ_MODULO, int(t), [(channel_index, float(v))])
        for i, (v, t) in enumerate(zip(values, times_us))
    ]


def publish(broker, topic: str, apid: int, stream: list) -> int:
    key = apid.to_bytes(2, "big")
    for raw in stream:
        broker.produce(topic, key, raw)
    broker.flush()
    return len(stream)


def replay_channel(broker, topic: str, registry: dict, channel_id: str, values, start_us: int, cadence_s: float,
                   plan: FaultPlan | None = None, seq_start: int = DEFAULT_SEQ_START):
    """Packetise, damage, publish. Returns (n_published, FaultTruth, packets_before_faults)."""
    ch = registry[channel_id]
    times = assign_timestamps(len(values), start_us, cadence_s)
    clean = build_packets(ch["apid"], ch["index"], values, times, seq_start)
    stream, truth = inject_faults(clean, plan) if plan else (clean, FaultTruth(len(clean), 0, 0, 0, 0))
    publish(broker, topic, ch["apid"], stream)
    return len(stream), truth, clean


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay one dataset channel as CCSDS packets into Kafka/Redpanda")
    ap.add_argument("--channel", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--corrupt", type=float, default=0.0)
    ap.add_argument("--dup", type=float, default=0.0)
    ap.add_argument("--reorder", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--registry", default="data/channels.json")
    args = ap.parse_args(argv)

    from ..packets.channels import load_channel_registry

    s = get_settings()
    values = load_channel(args.channel, args.split, s["data_dir"])
    if args.limit:
        values = values[: args.limit]
    registry = load_channel_registry(args.registry)
    start_us = int(datetime.fromisoformat(s["stream_start"]).timestamp() * 1_000_000)
    plan = FaultPlan(args.seed, args.drop, args.corrupt, args.dup, args.reorder)
    n, truth, _ = replay_channel(KafkaBroker(s["kafka_bootstrap"]), s["topic_raw"], registry, args.channel,
                                 values, start_us, s["cadence_s"], plan)
    print(f"published {n} packets for {args.channel}; injected faults: {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
