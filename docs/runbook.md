# Runbook

## Local PostgreSQL for the tests
Store tests create a throwaway database, so the role needs `CREATEDB`:

```bash
sudo -u postgres psql -c "CREATE ROLE telemetry LOGIN CREATEDB PASSWORD 'change_me'"
export POSTGRES_PASSWORD=change_me     # or copy .env.example to .env
pytest -rs                             # -rs shows skip reasons
```
**If pytest reports skips, the database tests did not run.** Use the skip summary and local PostgreSQL configuration to verify that the store suite executed.

## Common problems

| Symptom | Likely cause | Check |
|---|---|---|
| `password authentication failed` | role/password mismatch | `.env` vs the role you created |
| Tables missing after `init_schema` | schema script did not execute on the target connection | `TimescaleStore._run_script` must use the raw driver connection |
| Consumer lag grows | decode errors or slow DB | `bad_packets` count; `SELECT count(*) FROM telemetry_readings` over time |
| Readings fewer than expected | drops/corruption in the stream | `packet_events` kinds and the `/packet-health` endpoint; compare with the producer's fault flags |
| Zero detections on a channel | threshold too high / signal too smooth | `docs/evaluation_report.md` failure cases; plot with `scripts/make_figures.py` |
| Dashboard empty | demo data is stamped from 2026-01-01 | widen the Grafana time range |

## Re-running an ingest
Safe: replays are idempotent. Re-run `replay/producer.py` with the same arguments and `SELECT count(*)` must not change.

## Tuning detectors
`python scripts/tune_hyperparameters.py --synthetic-seed 100 --out results/tuning` (or `--data-dir` with a *separate* set of channels). Then run `run_benchmark` once on different channels. Never tune on the channels you report.
