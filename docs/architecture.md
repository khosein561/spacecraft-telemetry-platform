# Architecture

```
 dataset / synthetic signal
        |  (replay/producer.py: one APID per channel, 60 s cadence)
        v
 CCSDS-style packets --(fault injector: drop / corrupt / dup / reorder, with exact ground truth)--+
        |                                                                                          |
        v                                                                                          |
 Kafka / Redpanda topic telemetry.raw   (key = APID -> per-channel order preserved)                 |
        |                                                                                          |
        v                                                                                          |
 consume/decoder_consumer.py                                                                       |
   decode + CRC  --bad--> bad_packets table (+ optional telemetry.bad topic)                       |
   SequenceTracker per APID -> packet_events (gap / duplicate / out_of_order)  <-- compared with ---+
   batched idempotent INSERT -> telemetry_readings (UNIQUE channel_id,time; ON CONFLICT DO NOTHING)
   commit offset LAST
        |
        v
 PostgreSQL (+ TimescaleDB extras: hypertable, retention, 1-minute continuous aggregate)
        |                         |
        v                         v
 anomaly/ (z-score, LSTM)    api/main.py (read-only)  ->  Grafana (provisioned datasource + dashboard)
        |
        v
 alerts/notify.py: dedupe + cooldown (spacecraft time) -> log / webhook -> alerts table
```

## Why each boundary exists

| Boundary | Reason |
|---|---|
| Packet codec is pure functions | No I/O, so it can be property-tested and can never take the pipeline down. |
| Broker is an interface (`InMemoryBroker`, `KafkaBroker`) | Every ingest test runs without Kafka; only a thin adapter is untested against a live broker. |
| Consumer commits last | A crash before the commit means redelivery; the idempotent write makes redelivery harmless. |
| Spacecraft time, not arrival time | Replays and late downlinks must land on the same rows. |
| Evaluation is separate from detection | Detectors return alert ranges; `evaluate.py` scores them against labels, so scoring bugs can't hide inside a model. |
| Core schema vs TimescaleDB extras | Core runs on plain PostgreSQL (and is what the tests exercise); extras are optional. |

## Data model (see `sql/schema.sql`)

`channels` (APID, limits) - `telemetry_readings` (time, channel_id, value, quality) - `packet_events` - `bad_packets` - `anomalies` - `alerts`.

## Not built / out of scope

Authentication, TLS, multi-node Kafka, schema registry, model registry/retraining, multivariate models, real mission data formats (this uses a simplified CCSDS-style layout, not a flight-qualified implementation).
