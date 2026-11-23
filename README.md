# Spacecraft Telemetry Ingestion and Anomaly Detection Platform

A ground-segment-style pipeline: **CCSDS-style space packets -> Kafka/Redpanda -> validated, idempotent storage in PostgreSQL/TimescaleDB -> anomaly detection (rolling z-score baseline vs. a PyTorch LSTM forecaster) -> deduplicated alerts -> read-only API + Grafana dashboard**, with a fault-injection harness that knows exactly which packets it dropped, corrupted, duplicated or reordered, so the ingest layer can be checked against ground truth.

> **Honest status.** Everything below marked *verified* was run and tested in a Linux sandbox. The detector results are on a **synthetic** benchmark (generated signals with the *real* NASA SMAP/MSL label geometry). **No results on the real NASA data are reported here**; one command below runs them. Docker Compose, a live Redpanda broker, the TimescaleDB extras and the Grafana dashboard were **written but not run** (no Docker in the build sandbox).

## What was verified (this is what the tests actually prove)

| Claim | Evidence |
|---|---|
| Packet encode/decode round-trips; CRC-16/CCITT-FALSE matches the standard check value (0x29B1); never crashes on garbage | `tests/test_ccsds_roundtrip.py` (incl. property tests) |
| Sequence tracker handles 14-bit wraparound (16383 -> 0 is *not* a gap), gaps, duplicates, late arrivals | `tests/test_sequence_gaps.py` |
| **Detected faults == injected faults**, for 4 random seeds: `lost == dropped + corrupted`, `duplicates == duplicated`, `out_of_order == reordered`, `readings stored == N - never_delivered` | `tests/test_faults_and_pipeline.py` |
| Replaying the same stream twice stores nothing new; a crash before offset commit redelivers safely (at-least-once + idempotent writes = effectively-once) | `tests/test_idempotent_replay.py` |
| Bad packets are dead-lettered, not fatal | `tests/test_faults_and_pipeline.py`, `tests/test_store.py` |
| Detector metrics code (sequence-level overlap scoring, latency, false alarms / channel-day) | `tests/test_evaluate.py` |
| Alert dedupe / cooldown in spacecraft time; failed delivery does not start the cooldown | `tests/test_alerts.py` |

**121 tests pass** against a real PostgreSQL 16 (throwaway database per run). I also broke the code on purpose six ways (no CRC check, no `ON CONFLICT`, wrong length field, no duplicate counting, no sequence wraparound, commit-before-write) and the tests failed each time.

Throughput on a 2-core sandbox (Python 3.13), measured by `scripts/bench_throughput.py`: ~50,000 packets/s encode, ~42,000 packets/s decode (1 reading per packet); ~17,400 packets/s end-to-end ingest from an in-memory broker into Postgres (decode + sequence tracking + batched idempotent insert). This does **not** include Kafka.

## Detector results (SYNTHETIC data - read this)

Single final run on synthetic seed 0, 12 channels never used for tuning (hyper-parameters were chosen on a different seed, 100, on 16 other channels), same channels for both detectors, 59 channel-days:

| Detector | Precision | Recall | F1 | False alarms / channel-day | Detected / true sequences |
|---|---|---|---|---|---|
| Rolling z-score (window 400, thr 4) | 0.25 | 0.17 | 0.20 | 0.10 | 2 / 12 |
| LSTM forecaster + dynamic threshold | 0.60 | 0.50 | 0.55 | 0.07 | 6 / 12 |

Why you should not over-read this: only 12 true sequences (huge error bars); the anomalies are ones *I* injected; and the benchmark's own limits are in `docs/evaluation_report.md`, including the failure cases (the LSTM is blind to a smooth contextual change because the new signal is *easier* to predict). The point of the table is a fair, reproducible comparison pipeline, not a claim about NASA data.

## Quickstart (no Docker needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU torch is fine
pytest                              # store tests need a local PostgreSQL; see docs/runbook.md. Without one they SKIP - check the skip count!

# detector benchmark on synthetic data (about 90 s on 2 CPU cores)
python -m telemetry.anomaly.run_benchmark --synthetic-seed 0 --max-channels 12 --out results/my_run
```

### Run on the REAL NASA SMAP/MSL data (do this before claiming real-data results)

```bash
bash scripts/fetch_smap_msl.sh                       # needs a free Kaggle API token; see the script
python -m telemetry.anomaly.run_benchmark --data-dir data/raw --max-channels 12 --out results/real
```
Check `data/README.md` for the expected file layout, then put the numbers you actually get into this README.

### Full stack (UNTESTED compose file)

```bash
python scripts/build_registry.py --synthetic-seed 0 --out data/channels.json   # or --data-dir data/raw
docker compose up -d redpanda timescaledb grafana
export DATA_DIR=data/synthetic     # where build_registry --synthetic-seed wrote the arrays (use data/raw for the real data)
python -m telemetry.replay.producer --channel A-1 --limit 5000 --drop 0.02 --dup 0.01
python -m telemetry.consume.decoder_consumer
```

## Design decisions (say these in an interview)

- **Packets**: 6-byte primary header (11-bit APID, 14-bit sequence count, "length minus one"), 8-byte microsecond timestamp, readings, CRC-16. The decoder requires the buffer length to match the length field *exactly*, so a lying length can't make it over-read.
- **Sequence tracking mod 16384**: forward distance < 8192 = gap, else behind (duplicate / late). A "missing set" lets a late packet fill its hole and be counted as reordering rather than a new fault.
- **Kafka key = APID**: all packets of a channel land on one partition, so per-channel order is preserved. Sequence counts are per-APID, so this matches the data model.
- **At-least-once + idempotent store**: commit the offset only after the DB write; `UNIQUE (channel_id, time)` + `ON CONFLICT DO NOTHING` makes redelivery harmless. Time is *spacecraft* time from the packet, not arrival time.
- **Dead-letter**: damaged packets go to a `bad_packets` table (and optional topic) with the reason; one bad packet never stops the stream.
- **Evaluation**: sequence-level (a detected *event* counts, as in Hundman et al.), with false alarms per channel-day and detection latency, because per-point F1 rewards flagging long ranges.
- **Dynamic threshold**: simplified non-parametric method from the Hundman et al. (KDD 2018) paper; differences are listed in the evaluation report.

## Limitations

- Synthetic-data results only; real-data run left to you (instructions above).
- Single-process, single-node; no TLS/auth on Kafka or the API; the API is read-only and unauthenticated.
- One-step LSTM forecaster per channel; no multivariate model, no model registry or retraining policy.
- SMAP/MSL timestamps are not in the data; the project assumes a fixed 60 s cadence.
- Docker Compose, live Kafka (`KafkaBroker` tested only against fakes), TimescaleDB hypertable/retention/continuous-aggregate SQL, and the Grafana dashboard were **not executed**.

## Layout

```
src/telemetry/packets/   ccsds.py (encode/decode), sequence.py (gap tracking), channels.py (registry, limits)
src/telemetry/replay/    dataset_loader, synthetic benchmark, faults (ground-truth injector), producer
src/telemetry/consume/   decoder_consumer.py (decode -> validate -> store -> commit)
src/telemetry/store/     timescale.py (+ sql/schema.sql, sql/timescale_extras.sql)
src/telemetry/anomaly/   baseline (z-score), forecast_model (LSTM), threshold, evaluate, run_benchmark
src/telemetry/alerts/    notify.py (dedupe/cooldown, webhook)
src/telemetry/orbit/     passes.py (stretch: pass prediction + store-and-forward ordering, synthetic TLE)
src/telemetry/api/       main.py (read-only FastAPI)
docs/                    architecture, runbook, evaluation_report, failure_drills, CODE_CHEAT_SHEET
```

## Credits and AI disclosure

Labels (`data/labeled_anomalies.csv`) are from NASA/JPL's `telemanom` repository (Apache-2.0). The method follows Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018. This starting codebase was written with AI assistance (Claude). **Edit this sentence so it is true for your repo** (for example: what you re-implemented yourself, what you changed), and be ready to explain every file; see `docs/CODE_CHEAT_SHEET.md`.
