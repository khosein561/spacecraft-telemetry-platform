# Code Cheat Sheet: spacecraft telemetry platform

**How to use this:** interviewers ask *"why did you build it this way?"* far more than *"what does line 40 do?"*. For each file: **Why** (the problem), **What** (mechanics), **Say** (an interview-ready sentence), **Learn it in** (curriculum unit). Read it next to the code, then close the code and explain each file out loud. If you can't, you don't own it yet. Rewrite files yourself before you put this on your GitHub, and be honest that AI helped you get started.

Curriculum codes (Master Learning System vault): SHARED-06 Python Foundations - SHARED-05 Probability - CS-04 Algorithms - CS-06 SQL & Relational Databases - CS-07 Computer Networking - CS-08 Data Engineering Core - CS-09 Cloud & Analytics Engineering - CS-11 Machine Learning Foundations (wk 34) - PORTFOLIO-6 (wk 38, this project's brief).

---

## 0. The 30-second pitch

> "I built a spacecraft telemetry pipeline. Readings are encoded as CCSDS-style space packets with a CRC and sequence counts, streamed through Kafka, decoded and stored idempotently in Postgres. To prove the ingest layer works I built a fault injector that knows exactly what it dropped, corrupted, duplicated and reordered, and the tests assert detected equals injected. On top of that I compared a rolling z-score baseline with an LSTM forecaster using sequence-level scoring on NASA's anomaly labels - on synthetic data so far, and I'm upfront about that."

Numbers you may quote (all measured, see README): **121 tests** against a real Postgres - **4 random fault seeds, detected == injected** - **~17k packets/s** ingest in-memory broker to Postgres (2-core sandbox) - synthetic benchmark **LSTM 6/12 vs z-score 2/12** detected, 12 channels, one seed (say "synthetic" and "small sample").

---

## 1. The big picture: one reading's life

1. `replay/producer.py` turns a channel's values into packets (`ccsds.py`), optionally damaged by `faults.py`, and publishes keyed by APID.
2. `consume/decoder_consumer.py` decodes (CRC check), tracks sequence counts (`sequence.py`), writes readings idempotently (`store/timescale.py`), dead-letters bad packets, then commits the offset.
3. `anomaly/` scores the stored series (z-score or LSTM + dynamic threshold); `evaluate.py` scores detections against labels.
4. `alerts/notify.py` deduplicates alerts; `api/main.py` and Grafana read the results.

---

## 2. Packets: `src/telemetry/packets/`

### `ccsds.py`
- **Why:** ground software turns bytes into numbers; everything downstream depends on this being exact and never crashing on garbage.
- **What:** 6-byte primary header (3-bit version, type, secondary-header flag, 11-bit APID, 2 sequence-flag bits, 14-bit sequence count, 16-bit length = data bytes **minus one**), 8-byte microsecond timestamp, readings (u16 channel index + f32), trailing CRC-16/CCITT-FALSE. `decode_packet` requires the buffer length to match the length field exactly, so a lying length cannot make it read past the end. CRC failure raises `PacketCRCError`, other problems `PacketError` (both `ValueError`).
- **Say:** "The length field is bytes minus one - the classic off-by-one. I never trust it: the buffer has to match exactly, and anything malformed becomes a typed error that gets dead-lettered, never a crash."
- **Learn it in:** CS-07 (headers, bit fields, checksums, framing); CS-05 (binary layout, endianness); SHARED-06 (`struct`).

### `sequence.py`
- **Why:** a space link drops, repeats and reorders packets; the 14-bit sequence count is how the ground notices.
- **What:** arithmetic is modulo 16384, so 16383 -> 0 is normal. Forward distance < 8192 = gap; otherwise the packet is behind (duplicate or late). A `missing` set lets a late packet fill its hole and count as *reordering*, not a new fault. `lost` = holes still open + untracked huge gaps (above 4,096 we count but stop remembering each number, to bound memory).
- **Say:** "With only 14 bits you can't tell 'four behind' from '16,380 ahead', so I use a half-space convention - the same idea as TCP sequence-number comparison."
- **Learn it in:** CS-07 (sequence numbers, ordering, retransmission); CS-04 (modular arithmetic, sets, bounded memory); SHARED-05 only loosely.

### `channels.py`
- **Why:** you need a channel -> APID map and limits for dashboards/alerts, built without peeking at test data.
- **What:** `build_registry` assigns indexes and APIDs in sorted order and sets the nominal band from the **train** split's 0.5-99.5 percentiles (warn band = +-25% of span).
- **Say:** "Limits come from training data only, so nothing is learned from the test split."
- **Learn it in:** CS-08 (reference/dimension data); SHARED-05 (percentiles).

---

## 3. Replay and fault injection: `src/telemetry/replay/`

### `dataset_loader.py`
- **Why:** one place that reads labels/arrays defensively. **What:** `parse_anomaly_sequences` uses `ast.literal_eval` (never `eval`) and validates ranges; `load_channel` takes column 0, `allow_pickle=False`, rejects path tricks; `assign_timestamps` makes integer microseconds from a fixed cadence (the dataset has no timestamps - documented assumption).
- **Say:** "Loading `.npy` with pickle enabled can execute code, so I turn it off; channel ids are validated so they can't escape the data directory."
- **Learn it in:** SHARED-06 (files, numpy); CS-08 (ingestion hygiene, validation).

### `synthetic.py`
- **Why:** the real arrays need a download; a pipeline needs data from minute one. **What:** same file layout as the real dataset, real label geometry, generated signals (periodic, two-tone, stepwise, drift), injected spikes (point) and four contextual styles (level shift, amplitude collapse, flat-line, fast oscillation). Seeded and fingerprinted.
- **Say:** "Results on this say the harness works and compare methods fairly; they say nothing certain about real spacecraft."
- **Learn it in:** SHARED-05 (random generators, seeds); CS-11 (train/test splits, leakage).

### `faults.py`
- **Why:** you can only claim gap detection works if you know exactly what you injected. **What:** `inject_faults` gives each packet at most one fault (drop / bit-flip corrupt / duplicate / swap with neighbour) and returns `FaultTruth`. Tests assert detected == injected.
- **Say:** "A corrupted packet fails CRC, so to the sequence tracker it looks lost - `lost == dropped + corrupted` is the invariant I test."
- **Learn it in:** CS-08 (data-quality testing); SHARED-05 (random rates); CS-04.

### `producer.py`
- **Why/What:** packetises a channel, applies a fault plan, publishes with key = APID. **Say:** "Key = APID keeps one channel on one partition, so order is preserved per channel; sequence counts are per-APID anyway." **Learn it in:** CS-08 (Kafka partitions, keys, ordering).

---

## 4. Ingest: `broker.py`, `consume/decoder_consumer.py`

### `broker.py`
- **Why:** test everything without a Kafka server. **What:** `InMemoryBroker` (partitions by key, per-group offsets, `simulate_consumer_restart`) and `KafkaBroker` (confluent-kafka, synchronous commit of *next* offset). The Kafka adapter is tested only with fakes - not against a live broker.
- **Learn it in:** CS-08 (consumer groups, offsets, at-least-once); CS-04 (hashing keys to partitions).

### `decoder_consumer.py`
- **Why:** the ground-station ingest loop. **What:** per batch: decode -> bad packets to `bad_packets` (+ optional topic) -> track sequence -> batched insert -> **commit last**.
- **Say:** "Offsets are committed only after the database write succeeds, and the write is idempotent, so a crash gives redelivery, not loss or duplicates - effectively-once."
- **Learn it in:** CS-08 (idempotency, delivery semantics, dead-letter queues); CS-06 (transactions).

---

## 5. Storage: `store/timescale.py`, `sql/`

- **Why:** correctness in the database, not in hope. **What:** `UNIQUE (channel_id, time)` + `ON CONFLICT DO NOTHING`; batch inserts via `INSERT ... SELECT FROM unnest(arrays)` so row counts are accurate; times converted with exact `timedelta` microseconds (float conversion would lose precision); `sql/timescale_extras.sql` (hypertable, retention, continuous aggregate) is separate and **was not run**.
- **Say:** "Spacecraft time, not arrival time, is the key, so replays and late downlinks land on the same rows."
- **Learn it in:** CS-06 (constraints, upserts, indexes); CS-08 (time-series storage, idempotent loads); CS-09 (managed time-series, retention).

---

## 6. Detection: `src/telemetry/anomaly/`

### `baseline.py` (z-score)
- **Why:** always build the simple baseline first, or you can't tell if the model earned its complexity. **What:** rolling mean/std over a window that **excludes the current point**; streaming and vectorised forms agree; `flags_to_sequences` merges nearby flags. Weak on periodic signals because the swing inflates the rolling std.
- **Learn it in:** SHARED-05 (mean, std, z-scores); CS-10 (outliers).

### `forecast_model.py` (LSTM)
- **What:** PyTorch one-step-ahead forecaster per channel, trained on the train split with a validation holdout (best weights kept); `prediction_errors` returns NaN for the first `length` points; the detector sets those to the median so the train/test junction can't make a fake spike.
- **Say:** "The forecaster learns 'normal'; the anomaly score is how surprised it is. That's why it can't see an anomaly that is *easier* to predict than normal - I show that failure case in the report."
- **Learn it in:** CS-11 (sequence models, validation, overfitting); CS-10.

### `threshold.py`
- **What:** EWMA smoothing + non-parametric dynamic threshold chosen from the errors alone (no labels), with pruning; simplified from Hundman et al. (KDD 2018); per 2,000-point segment. **Say:** "A fixed cut-off breaks when channels have different noise; this picks the threshold from the error distribution." **Learn it in:** CS-10, CS-11.

### `evaluate.py`
- **Why:** per-point F1 rewards flagging long ranges; spacecraft ops care about events and alarm fatigue. **What:** overlap rule, micro-averaged counts, false alarms per channel-day, detection latency, per-class breakdown. **Say:** "A true sequence is detected if any alert overlaps it; any alert overlapping nothing is a false alarm." **Caveat to volunteer:** the rule is lenient on long labelled ranges. **Learn it in:** CS-11 (precision/recall), CS-10.

### `run_benchmark.py`
- Runs both detectors on the same channels, writes `results.json` with dataset fingerprint, timings, parameters; plots figures. **Learn it in:** CS-11 (evaluation protocol), CS-12 (reproducibility).

---

## 7. Operations: `alerts/`, `api/`, `orbit/`, `grafana/`

- `alerts/notify.py`: dedupe + cooldown measured in **spacecraft time**; a failed delivery does not start the cooldown; webhook accepts http/https only. *Say:* "A flapping alert pages people to death, so overlapping anomalies on one channel collapse into one alert." *Learn it in:* CS-08 (monitoring/alerting), CS-07 (HTTP).
- `api/main.py`: read-only FastAPI (`/health`, `/channels`, `/channels/{id}/readings`, `/anomalies`, `/packet-health`). *Learn it in:* CS-07, CS-09. Unauthenticated - say so.
- `orbit/passes.py` (stretch): pass prediction over Seattle with skyfield/sgp4 on a synthetic TLE, and store-and-forward ordering. *Learn it in:* CS-04.
- `grafana/`, `docker-compose.yml`: written, **not run**.

---

## 8. Tests: `tests/`

121 tests; the store tests need a real Postgres (they **skip** without one: check `pytest -rs`). Highest value: `test_faults_and_pipeline.py` (detected == injected, 4 seeds), `test_idempotent_replay.py` (replay twice = no new rows; crash before commit), `test_sequence_gaps.py` (wraparound), `test_ccsds_roundtrip.py` (CRC check value 0x29B1, garbage input). I also broke the code six ways on purpose and confirmed the tests fail each time (see `docs/failure_drills.md`).

---

## 9. Likely interview questions

1. *Why Kafka partitioned by APID?* Order per channel; sequence counts are per APID.
2. *What does "length minus one" mean and why does it matter?* Off-by-one in framing; show the decoder check.
3. *How do you avoid duplicates on replay?* Unique key + `ON CONFLICT DO NOTHING`, commit after write.
4. *Your LSTM missed the contextual anomalies - why?* Forecast error measures unpredictability, not difference.
5. *Are your detector results real?* Synthetic so far; here is the command for real data; small sample.
6. *What would you do next?* Real SMAP/MSL run, multiple seeds, a distribution-shift detector, auth/TLS, live Kafka tests.
