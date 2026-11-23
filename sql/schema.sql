-- Core schema. Works on any PostgreSQL 14+. TimescaleDB-only extras (hypertable, retention,
-- continuous aggregate) live in sql/timescale_extras.sql so the project also runs on plain Postgres.
--
-- Idempotency lives in the schema: UNIQUE (channel_id, time) means replaying a recording can never
-- create a second copy of a reading; inserts use ON CONFLICT DO NOTHING.

CREATE TABLE IF NOT EXISTS channels (
    channel_id    TEXT PRIMARY KEY,
    apid          INT  NOT NULL UNIQUE,
    channel_index INT  NOT NULL,
    spacecraft    TEXT,
    grp           TEXT,
    unit          TEXT,
    nominal_min   DOUBLE PRECISION,
    nominal_max   DOUBLE PRECISION,
    warn_min      DOUBLE PRECISION,
    warn_max      DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS telemetry_readings (
    time       TIMESTAMPTZ      NOT NULL,   -- SPACECRAFT time from the packet, not arrival time
    channel_id TEXT             NOT NULL,
    value      DOUBLE PRECISION NOT NULL,
    quality    SMALLINT         NOT NULL DEFAULT 0,
    UNIQUE (channel_id, time)
);
CREATE INDEX IF NOT EXISTS idx_readings_time ON telemetry_readings (time DESC);

-- One row per sequence-tracking event (not per packet): gaps, repeats, late arrivals.
-- packet_time is part of the key so replaying the same stream produces the same rows.
CREATE TABLE IF NOT EXISTS packet_events (
    event_id    BIGSERIAL PRIMARY KEY,
    apid        INT         NOT NULL,
    seq_count   INT         NOT NULL CHECK (seq_count BETWEEN 0 AND 16383),
    kind        TEXT        NOT NULL CHECK (kind IN ('gap', 'duplicate', 'out_of_order')),
    gap         INT         NOT NULL DEFAULT 0,
    packet_time TIMESTAMPTZ NOT NULL,
    UNIQUE (apid, seq_count, kind, packet_time)
);

-- Dead-letter table for packets that could not be decoded (bad CRC, wrong length, ...).
CREATE TABLE IF NOT EXISTS bad_packets (
    raw_sha1  TEXT PRIMARY KEY,             -- hash of the bytes: the same garbage is stored once
    reason    TEXT  NOT NULL,
    apid      INT,                          -- best-effort, NOT trustworthy for corrupted packets
    seq_count INT,
    raw       BYTEA NOT NULL,
    seen_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id BIGSERIAL PRIMARY KEY,
    channel_id TEXT        NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time   TIMESTAMPTZ NOT NULL,
    detector   TEXT        NOT NULL,
    score      DOUBLE PRECISION,
    labeled    BOOLEAN,                     -- true if it overlaps a ground-truth label (evaluation only)
    UNIQUE (channel_id, detector, start_time)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    BIGSERIAL PRIMARY KEY,
    channel_id  TEXT        NOT NULL,
    detector    TEXT        NOT NULL,
    severity    TEXT        NOT NULL,
    event_time  TIMESTAMPTZ NOT NULL,       -- spacecraft time of the anomaly that triggered it
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    destination TEXT        NOT NULL,
    status      TEXT        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_channel_time ON alerts (channel_id, event_time DESC);
