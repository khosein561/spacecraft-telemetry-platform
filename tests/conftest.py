"""Shared fixtures. Store tests run against a REAL PostgreSQL (a throwaway database per test session)."""
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from telemetry.packets.channels import build_registry
from telemetry.store.timescale import TimescaleStore

ROOT = Path(__file__).resolve().parents[1]


def _admin_url() -> str:
    url = os.environ.get("TELEMETRY_TEST_ADMIN_URL")
    if url:
        return url
    try:
        from dotenv import dotenv_values

        env = dotenv_values(ROOT / ".env")
    except Exception:  # pragma: no cover
        env = {}
    g = lambda k, d: os.environ.get(k) or env.get(k) or d
    return f"postgresql+psycopg2://{g('POSTGRES_USER','telemetry')}:{g('POSTGRES_PASSWORD','change_me')}@{g('POSTGRES_HOST','localhost')}:{g('POSTGRES_PORT','5432')}/postgres"


@pytest.fixture(scope="session")
def engine():
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    name = f"telemetry_test_{uuid.uuid4().hex[:8]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL not available for store tests: {exc}")
    url = make_url(_admin_url()).set(database=name)
    eng = create_engine(url, pool_pre_ping=True)
    TimescaleStore(eng).init_schema(timescale=False)
    yield eng
    eng.dispose()
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture()
def store(engine):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE telemetry_readings, packet_events, bad_packets, anomalies, alerts, channels RESTART IDENTITY"))
    return TimescaleStore(engine)


@pytest.fixture()
def registry():
    rng = np.random.default_rng(0)
    return build_registry({
        "A-1": {"spacecraft": "SMAP", "train": rng.normal(0, 0.3, 500)},
        "B-2": {"spacecraft": "MSL", "train": rng.normal(0, 0.3, 500)},
    })
