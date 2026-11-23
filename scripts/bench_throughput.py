#!/usr/bin/env python3
"""Measure packet encode/decode speed and end-to-end ingest speed (InMemoryBroker -> real Postgres).

    python scripts/bench_throughput.py --packets 50000 --readings-per-packet 1

Numbers depend on the machine; quote them WITH the machine description this script prints.
The ingest part needs a reachable PostgreSQL (same env vars as the tests); it uses a throwaway database.
"""
import argparse
import json
import platform
import time
import uuid

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from telemetry.broker import InMemoryBroker
from telemetry.config import get_settings
from telemetry.consume.decoder_consumer import DecoderConsumer
from telemetry.packets.ccsds import decode_packet, encode_packet
from telemetry.packets.channels import build_registry
from telemetry.replay.producer import publish
from telemetry.store.timescale import TimescaleStore


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=int, default=50_000)
    ap.add_argument("--readings-per-packet", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    n, k = a.packets, a.readings_per_packet
    t0 = 1_767_225_600_000_000
    rng = np.random.default_rng(0)

    packets, t_enc = timed(lambda: [encode_packet(100, i % 16384, t0 + i * 60_000_000, [(j, float(rng.normal())) for j in range(k)]) for i in range(n)])
    _, t_dec = timed(lambda: [decode_packet(p) for p in packets])
    res = {"machine": f"{platform.platform()} / python {platform.python_version()}", "packets": n, "readings_per_packet": k,
           "encode_packets_per_s": round(n / t_enc), "decode_packets_per_s": round(n / t_dec)}

    if k != 1:
        res["note"] = "ingest benchmark only runs with --readings-per-packet 1"
        print(json.dumps(res, indent=1))
        if a.out:
            open(a.out, "w").write(json.dumps(res, indent=1))
        return
    s = get_settings()
    admin_url = s["database_url"].rsplit("/", 1)[0] + "/postgres"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    name = f"bench_{uuid.uuid4().hex[:8]}"
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
        eng = create_engine(make_url(admin_url).set(database=name))
        store = TimescaleStore(eng)
        store.init_schema(timescale=False)
        rng2 = np.random.default_rng(1)
        reg = build_registry({f"CH-{j}": {"spacecraft": "SYN", "train": rng2.normal(size=500)} for j in range(k)})
        store.upsert_channels(reg)
        broker = InMemoryBroker()
        # one APID per channel when k == 1, so build via the registry's APIDs
        ch0 = reg["CH-0"]
        pk = [encode_packet(ch0["apid"], i % 16384, t0 + i * 60_000_000, [(ch0["index"], float(i % 100))]) for i in range(n)]
        publish(broker, s["topic_raw"], ch0["apid"], pk)
        cons = DecoderConsumer(broker, store, reg, batch_size=1000)
        stats, t_ing = timed(lambda: cons.run_until_idle())
        res.update({"ingest_packets_per_s": round(n / t_ing), "ingest_rows_stored": store.count("telemetry_readings"), "ingest_seconds": round(t_ing, 2),
                    "note": "single APID, in-memory broker, Postgres on same host: measures decode + sequence tracking + batched insert only, not Kafka"})
    except Exception as exc:  # pragma: no cover
        res["ingest_error"] = str(exc)[:200]
    finally:
        try:
            with admin.connect() as c:
                c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        except Exception:
            pass
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
