# Spacecraft Telemetry Platform

A Python ground-segment platform for ingesting CCSDS-style telemetry, storing time-series readings, detecting anomalies, and delivering deduplicated alerts.

```text
Telemetry packets -> Kafka/Redpanda -> decode and validate -> PostgreSQL/TimescaleDB
                                                            |
                                                            v
                                                anomaly detection + alerts
                                                            |
                                                            v
                                                  FastAPI + Grafana
```

The platform also includes fault injection and ground-truth checks, making it possible to verify that packet loss, corruption, duplication, and reordering are detected correctly rather than only observing the end state.

## What the system demonstrates

- Decoding and validating a packet format with a CRC-16 integrity check
- Tracking sequence gaps across 14-bit counters that wrap from `16383` to `0`
- Processing Kafka-style streams with at-least-once delivery and idempotent database writes
- Comparing a rolling z-score baseline with a PyTorch LSTM forecaster
- Scoring detections at the event level with precision, recall, latency, and false alarms
- Exposing read-only telemetry endpoints and a Grafana dashboard
- Injecting known faults and asserting that processing behavior matches the injected ground truth

## Architecture

The pipeline has three main stages:

1. **Replay and transport** replay synthetic or NASA-style channel data, wrap readings in packets, and publish them to Kafka/Redpanda.
2. **Validation and storage** decode each packet, verify its contents, track sequence health, and insert readings without duplicate rows.
3. **Analysis and reporting** run detectors, record anomalies and alerts, and serve both API and dashboard views.

The data model contains channels, readings, packet events, bad packets, anomalies, and alerts. See [Architecture](docs/architecture.md) for the full data flow and boundary decisions.

## Validation

The test suite covers packet encoding, sequence wraparound, fault accounting, replay safety, storage behavior, anomaly evaluation, and alert delivery. The PostgreSQL-backed suite currently passes 121 tests when a local PostgreSQL 16 instance is available.

Measured on a 2-core Python 3.13 environment:

| Operation | Throughput |
|---|---:|
| Packet encoding | ~50,000 packets/s |
| Packet decoding | ~42,000 packets/s |
| In-memory ingest through decode, sequence tracking, and PostgreSQL insert | ~17,400 packets/s |

Kafka transport is outside the local benchmark. `scripts/bench_throughput.py` can reproduce the measurements.

## Detector evaluation

A reproducible synthetic benchmark uses channel geometries derived from the NASA SMAP/MSL label layout. Generated signals and injected anomalies are used so the comparison can run without downloading external datasets.

| Detector | Precision | Recall | F1 | False alarms / channel-day | Detected / 12 sequences |
|---|---:|---:|---:|---:|---:|
| Rolling z-score, window 400 | 0.25 | 0.17 | 0.20 | 0.10 | 2 |
| LSTM forecaster with dynamic threshold | 0.60 | 0.50 | 0.55 | 0.07 | 6 |

These results cover 12 held-out channels and are intended to demonstrate the benchmark and evaluation pipeline. They are not a claim about live spacecraft performance. The sample is small, and the known limitations are documented in [Evaluation Report](docs/evaluation_report.md).

## Getting started

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

PostgreSQL-backed tests require a local PostgreSQL role with `CREATEDB` permissions. Setup and troubleshooting steps are in [Runbook](docs/runbook.md).

### Run the synthetic benchmark

```bash
python -m telemetry.anomaly.run_benchmark --synthetic-seed 0 --max-channels 12 --out results/my_run
```

The benchmark is completed in roughly 90 seconds on a 2-core environment.

### Run the full stack

```bash
python scripts/build_registry.py --synthetic-seed 0 --out data/channels.json
docker compose up -d redpanda timescaledb grafana
export DATA_DIR=data/synthetic
python -m telemetry.replay.producer --channel A-1 --limit 5000 --drop 0.02 --dup 0.01
python -m telemetry.consume.decoder_consumer
```

Docker Compose, the live Redpanda integration, and TimescaleDB extras were not available in the local validation environment. Docker and service configuration remain in the repository for deployment.

### Optional NASA SMAP/MSL benchmark

```bash
bash scripts/fetch_smap_msl.sh
python -m telemetry.anomaly.run_benchmark --data-dir data/raw --max-channels 12 --out results/real
```

The Kaggle API and dataset layout are described in [data/README.md](data/README.md).

## Key implementation decisions

- **Packet integrity:** a packet's declared length must exactly match its payload length, and CRC-16/CCITT-FALSE verifies the final bytes.
- **Sequence tracking:** modulo-16384 arithmetic distinguishes wraparound, gaps, duplicates, and late packets.
- **Kafka partitioning:** APID is used as the message key so readings for one channel remain in order.
- **Delivery semantics:** offsets commit only after the database write, while a unique channel/time constraint prevents duplicate replay rows.
- **Fault handling:** damaged packets are recorded in a dead-letter table and do not stop the remaining stream.
- **Evaluation:** detection ranges, false alarms per channel-day, and latency are measured separately from raw model scores.
- **Alert behavior:** overlapping anomalies collapse into one alert, and spacecraft-time cooldown prevents rapid repeated delivery.

## Scope and limitations

- The synthetic benchmark uses injected anomalies and is separate from real NASA data results.
- The deployment is a single-process example with no Kafka authentication or API authentication.
- The current forecaster is one-step-ahead and per channel rather than a multivariate model.
- SMAP/MSL sources do not include timestamps, so the benchmark assumes a fixed 60-second cadence.
- Live Kafka, TimescaleDB hypertables, continuous aggregates, and Grafana were configured but not exercised in every local test environment.

## Project structure

```text
src/telemetry/packets/   packet format, sequence tracking, channel registry
src/telemetry/replay/    dataset loading, synthetic signals, fault injection, producer
src/telemetry/consume/   decode, validation, storage, and offset handling
src/telemetry/store/     PostgreSQL/TimescaleDB persistence
src/telemetry/anomaly/   baseline, forecaster, threshold, and evaluation code
src/telemetry/alerts/    deduplication, cooldown, and notification delivery
src/telemetry/orbit/     orbital-pass and ordering utilities
src/telemetry/api/       read-only FastAPI application
sql/                    core and TimescaleDB-specific schemas
scripts/                benchmarking and benchmark preparation tools
docs/                   architecture, operations, evaluation, and fault drills
```

## References

- NASA/JPL `telemanom` labels, Apache-2.0
- K. Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," KDD 2018
