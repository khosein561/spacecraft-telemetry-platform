"""
Database access for telemetry (TimescaleDB, or plain PostgreSQL).

WHY THIS FILE EXISTS: one place owns every SQL statement the application runs, so idempotency
(`ON CONFLICT DO NOTHING`) and time handling are done once and tested once.

Time: packets carry integer microseconds since the epoch. They are converted with timedelta
arithmetic (exact) rather than float division (which rounds at microsecond scale).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SQL_DIR = Path(__file__).resolve().parents[3] / "sql"
BATCH = 5000


def us_to_datetime(us: int) -> datetime:
    return EPOCH + timedelta(microseconds=int(us))


def datetime_to_us(dt: datetime) -> int:
    delta = dt - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _run_script(engine, path: Path) -> None:
    """Run a multi-statement .sql file through the raw DBAPI cursor (SQLAlchemy would read '%' as a parameter)."""
    raw = engine.raw_connection()
    try:
        dbapi = raw.driver_connection  # the real psycopg2 connection (not the pool wrapper)
        dbapi.autocommit = True  # continuous aggregates cannot be created inside a transaction block
        try:
            with dbapi.cursor() as cur:
                cur.execute(path.read_text())
        finally:
            dbapi.autocommit = False  # hand the connection back to the pool in its normal state
    finally:
        raw.close()


class TimescaleStore:
    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> "TimescaleStore":
        return cls(create_engine(url, pool_pre_ping=True))

    # ---------- schema ----------
    def init_schema(self, timescale: bool | None = None) -> bool:
        """Create tables. Returns True if the TimescaleDB extras were applied.

        timescale=None: use them only if the extension is available on this server.
        """
        _run_script(self.engine, SQL_DIR / "schema.sql")
        if timescale is None:
            with self.engine.connect() as conn:
                timescale = bool(conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")).first())
        if timescale:
            _run_script(self.engine, SQL_DIR / "timescale_extras.sql")
        return bool(timescale)

    def upsert_channels(self, registry: dict) -> None:
        with self.engine.begin() as conn:
            for ch in registry.values():
                conn.execute(
                    text(
                        """INSERT INTO channels (channel_id, apid, channel_index, spacecraft, grp, unit,
                                                 nominal_min, nominal_max, warn_min, warn_max)
                           VALUES (:channel_id, :apid, :index, :spacecraft, :group, :unit,
                                   :nominal_min, :nominal_max, :warn_min, :warn_max)
                           ON CONFLICT (channel_id) DO UPDATE SET
                               nominal_min = EXCLUDED.nominal_min, nominal_max = EXCLUDED.nominal_max,
                               warn_min = EXCLUDED.warn_min, warn_max = EXCLUDED.warn_max"""
                    ),
                    ch,
                )

    # ---------- writes (all idempotent) ----------
    def insert_readings(self, rows) -> int:
        """rows: iterable of (channel_id, time_us, value, quality). Returns rows actually inserted."""
        rows = list(rows)
        inserted = 0
        sql = text(
            """INSERT INTO telemetry_readings (time, channel_id, value, quality)
               SELECT * FROM unnest(CAST(:t AS timestamptz[]), CAST(:c AS text[]),
                                    CAST(:v AS double precision[]), CAST(:q AS smallint[]))
               ON CONFLICT (channel_id, time) DO NOTHING"""
        )
        with self.engine.begin() as conn:
            for i in range(0, len(rows), BATCH):
                chunk = rows[i : i + BATCH]
                res = conn.execute(
                    sql,
                    {
                        "t": [us_to_datetime(r[1]) for r in chunk],
                        "c": [r[0] for r in chunk],
                        "v": [float(r[2]) for r in chunk],
                        "q": [int(r[3]) for r in chunk],
                    },
                )
                inserted += res.rowcount
        return inserted

    def insert_packet_events(self, rows) -> int:
        """rows: (apid, seq_count, kind, gap, packet_time_us)."""
        rows = list(rows)
        if not rows:
            return 0
        inserted = 0
        with self.engine.begin() as conn:
            for r in rows:
                res = conn.execute(
                    text(
                        """INSERT INTO packet_events (apid, seq_count, kind, gap, packet_time)
                           VALUES (:a, :s, :k, :g, :t)
                           ON CONFLICT (apid, seq_count, kind, packet_time) DO NOTHING"""
                    ),
                    {"a": r[0], "s": r[1], "k": r[2], "g": r[3], "t": us_to_datetime(r[4])},
                )
                inserted += res.rowcount
        return inserted

    def insert_bad_packets(self, rows) -> int:
        """rows: (raw_bytes, reason, apid_or_None, seq_or_None). The hash of the bytes is the key."""
        inserted = 0
        with self.engine.begin() as conn:
            for raw, reason, apid, seq in rows:
                res = conn.execute(
                    text(
                        """INSERT INTO bad_packets (raw_sha1, reason, apid, seq_count, raw)
                           VALUES (:h, :r, :a, :s, :raw) ON CONFLICT (raw_sha1) DO NOTHING"""
                    ),
                    {"h": hashlib.sha1(raw).hexdigest(), "r": reason, "a": apid, "s": seq, "raw": raw[:1024]},
                )
                inserted += res.rowcount
        return inserted

    def record_anomalies(self, channel_id: str, detector: str, sequences, times_us, scores=None, labeled=None) -> int:
        """sequences: (start_idx, end_idx) pairs into times_us. Idempotent on (channel, detector, start)."""
        inserted = 0
        with self.engine.begin() as conn:
            for i, (s, e) in enumerate(sequences):
                res = conn.execute(
                    text(
                        """INSERT INTO anomalies (channel_id, start_time, end_time, detector, score, labeled)
                           VALUES (:c, :s, :e, :d, :sc, :l)
                           ON CONFLICT (channel_id, detector, start_time) DO NOTHING"""
                    ),
                    {
                        "c": channel_id, "s": us_to_datetime(int(times_us[s])), "e": us_to_datetime(int(times_us[e])),
                        "d": detector, "sc": None if scores is None else float(scores[i]),
                        "l": None if labeled is None else bool(labeled[i]),
                    },
                )
                inserted += res.rowcount
        return inserted

    def insert_alert(self, alert) -> None:
        """alert: telemetry.alerts.notify.Alert (duck-typed). Alerts are an audit log, so not de-duplicated here."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO alerts (channel_id, detector, severity, event_time, destination, status)
                       VALUES (:c, :d, :sev, :t, :dest, :st)"""
                ),
                {"c": alert.channel_id, "d": alert.detector, "sev": alert.severity,
                 "t": us_to_datetime(alert.event_us), "dest": alert.destination, "st": alert.status},
            )

    # ---------- reads ----------
    def count(self, table: str) -> int:
        if table not in {"telemetry_readings", "packet_events", "bad_packets", "anomalies", "alerts", "channels"}:
            raise ValueError(f"unknown table {table!r}")
        with self.engine.connect() as conn:
            return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()

    def read_channel(self, channel_id: str, start_us: int | None = None, end_us: int | None = None):
        """Return (times_us int64 array, values float array) ordered by time."""
        clauses, params = ["channel_id = :c"], {"c": channel_id}
        if start_us is not None:
            clauses.append("time >= :s")
            params["s"] = us_to_datetime(start_us)
        if end_us is not None:
            clauses.append("time <= :e")
            params["e"] = us_to_datetime(end_us)
        with self.engine.connect() as conn:
            rows = conn.execute(text(f"SELECT time, value FROM telemetry_readings WHERE {' AND '.join(clauses)} ORDER BY time"), params).fetchall()
        times = np.array([datetime_to_us(r[0]) for r in rows], dtype=np.int64)
        values = np.array([r[1] for r in rows], dtype=float)
        return times, values

    def list_anomalies(self, channel_id: str | None = None, detector: str | None = None, limit: int = 1000) -> list:
        clauses, params = ["TRUE"], {"lim": int(limit)}
        if channel_id:
            clauses.append("channel_id = :c")
            params["c"] = channel_id
        if detector:
            clauses.append("detector = :d")
            params["d"] = detector
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT channel_id, detector, start_time, end_time, score, labeled FROM anomalies WHERE {' AND '.join(clauses)} ORDER BY start_time LIMIT :lim"),
                params,
            ).fetchall()
        return [dict(zip(("channel_id", "detector", "start_time", "end_time", "score", "labeled"), r)) for r in rows]

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar_one() == 1

    def packet_event_counts(self) -> dict:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT kind, count(*) FROM packet_events GROUP BY kind")).fetchall()
        return {k: n for k, n in rows}


if __name__ == "__main__":  # python -m telemetry.store.timescale init
    import sys

    from ..config import get_settings

    if sys.argv[1:] == ["init"]:
        ts = TimescaleStore.from_url(get_settings()["database_url"]).init_schema()
        print("schema ready" + (" (TimescaleDB extras applied)" if ts else " (plain PostgreSQL; TimescaleDB not available)"))
    else:
        print("usage: python -m telemetry.store.timescale init")
