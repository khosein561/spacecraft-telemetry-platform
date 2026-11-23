"""
Settings loader.

WHY THIS FILE EXISTS: every process (producer, consumer, detector, API) needs the same few
settings. Reading them in one place, from environment variables with safe local defaults, means
nothing is hard-coded and secrets never live in the code. A `.env` file is read if present.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime; tests set env vars directly
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_settings(env: dict | None = None) -> dict:
    """Read env vars (Kafka, Postgres, alert webhook, data paths) with sensible local defaults."""
    e = os.environ if env is None else env
    s = {
        "kafka_bootstrap": e.get("KAFKA_BOOTSTRAP", "localhost:9092"),
        "topic_raw": e.get("TOPIC_RAW", "telemetry.raw"),
        "topic_bad": e.get("TOPIC_BAD", "telemetry.bad"),
        "consumer_group": e.get("CONSUMER_GROUP", "telemetry-decoder"),
        "pg_host": e.get("POSTGRES_HOST", "localhost"),
        "pg_port": e.get("POSTGRES_PORT", "5432"),
        "pg_db": e.get("POSTGRES_DB", "telemetry"),
        "pg_user": e.get("POSTGRES_USER", "telemetry"),
        "pg_password": e.get("POSTGRES_PASSWORD", "change_me"),
        "alert_webhook_url": e.get("ALERT_WEBHOOK_URL", ""),
        "data_dir": e.get("DATA_DIR", str(PROJECT_ROOT / "data" / "raw")),
        "labels_path": e.get("LABELS_PATH", str(PROJECT_ROOT / "data" / "labeled_anomalies.csv")),
        "cadence_s": float(e.get("CADENCE_SECONDS", "60")),
        "stream_start": e.get("STREAM_START", "2026-01-01T00:00:00+00:00"),
    }
    s["database_url"] = e.get("TELEMETRY_DATABASE_URL") or (
        f"postgresql+psycopg2://{s['pg_user']}:{s['pg_password']}@{s['pg_host']}:{s['pg_port']}/{s['pg_db']}"
    )
    return s
