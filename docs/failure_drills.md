# Failure drills (each one is an automated test)

| Drill | What is injected | Expected | Test |
|---|---|---|---|
| Lossy link | 3% drop, 2% corrupt, 2% dup, 3% reorder, 4 seeds | `lost == dropped + corrupted`; `duplicates == duplicated`; `out_of_order == reordered`; rows stored == sent - never delivered | `test_detected_faults_equal_injected_faults` |
| Sequence wraparound | stream starts at seq 16300 and crosses 16383 -> 0 | no false gap | `test_sequence_gaps.py`, `test_faults_and_pipeline.py` |
| Replay twice | same stream published twice | zero new rows | `test_idempotent_replay.py` |
| Crash before commit | consumer restarts without committing | messages redelivered, no duplicates stored | `test_idempotent_replay.py`, `test_broker.py` |
| Garbage bytes | truncated, wrong length, bad version, bit flips | `PacketError`/`PacketCRCError`, never a crash | `test_ccsds_roundtrip.py` |
| Alert flapping | repeated anomaly on one channel | one alert per cooldown window; failed delivery does not start cooldown | `test_alerts.py` |

## Mutation sanity checks run on the code (each made the test suite fail)
CRC verification disabled; `ON CONFLICT DO NOTHING` removed; "length minus one" dropped; duplicate counter removed; sequence difference without modulo; offset commit moved before the database write.
(Note: removing the modulo from the *expected* value alone does not change behaviour, because the later subtraction is also modulo 16384. That mutant is equivalent, not a test gap.)

## Not drilled
Live Kafka broker restart, database outage with a real network, disk full, TimescaleDB retention policy.
