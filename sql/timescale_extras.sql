-- TimescaleDB-only features. Applied automatically by the timescale/timescaledb container
-- (docker-compose mounts it into docker-entrypoint-initdb.d) or by `python -m telemetry.store.timescale init`
-- when the extension is available.
--
-- STATUS: NOT executed in the environment where this project was built (no TimescaleDB there).
-- The statements follow the documented TimescaleDB 2.x API; if one fails, the core schema still works.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Partition readings by day on the time column. UNIQUE (channel_id, time) already includes `time`,
-- which TimescaleDB requires of unique indexes on hypertables.
SELECT create_hypertable('telemetry_readings', 'time',
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists => TRUE, migrate_data => TRUE);

-- Keep raw readings for a year (demo data is replayed with 2026 timestamps, so a 30-day policy
-- would delete it immediately; a real mission would pick this from its retention requirements).
SELECT add_retention_policy('telemetry_readings', INTERVAL '365 days', if_not_exists => TRUE);

-- 1-minute rollup per channel for fast dashboards.
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       channel_id,
       min(value) AS min_value,
       max(value) AS max_value,
       avg(value) AS avg_value,
       count(*)   AS samples
FROM telemetry_readings
GROUP BY bucket, channel_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('telemetry_1min',
       start_offset => INTERVAL '3 hours', end_offset => INTERVAL '1 minute',
       schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);
