"""Read-only API."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from telemetry.api.main import create_app
from telemetry.store.timescale import TimescaleStore
from helpers import START_US


@pytest.fixture()
def client(store, registry):
    store.insert_readings([("A-1", START_US + i * 60_000_000, v, 0) for i, v in enumerate([0.0, 0.1, 5.0, -0.2])])
    times = START_US + np.arange(10, dtype=np.int64) * 60_000_000
    store.record_anomalies("A-1", "zscore", [(2, 3)], times, scores=[6.5], labeled=[True])
    return TestClient(create_app(store, registry))


def test_health_and_channels(client):
    assert client.get("/health").json() == {"status": "ok"}
    ids = {c["channel_id"] for c in client.get("/channels").json()}
    assert ids == {"A-1", "B-2"}


def test_readings_include_limit_state_and_respect_limit(client):
    body = client.get("/channels/A-1/readings").json()
    assert body["count"] == 4 and [p["state"] for p in body["points"]][2] == "critical"   # 5.0 is far outside the limits
    assert client.get("/channels/A-1/readings?limit=2").json()["count"] == 2


def test_time_filter_needs_a_timezone(client):
    assert client.get("/channels/A-1/readings?start=2026-01-01T00:01:00Z&end=2026-01-01T00:02:00Z").json()["count"] == 2
    assert client.get("/channels/A-1/readings?start=2026-01-01T00:01:00").status_code == 422


def test_unknown_channel_and_bad_limits(client):
    assert client.get("/channels/NOPE/readings").status_code == 404
    assert client.get("/channels/A-1/readings?limit=0").status_code == 422
    assert client.get("/channels/A-1/readings?limit=999999").status_code == 422
    assert client.get("/channels/'; DROP TABLE x;--/readings").status_code == 404


def test_anomalies_and_packet_health(client):
    rows = client.get("/anomalies?channel=A-1").json()
    assert len(rows) == 1 and rows[0]["detector"] == "zscore" and rows[0]["labeled"] is True
    assert client.get("/anomalies?channel=B-2").json() == []
    assert client.get("/packet-health").json()["readings"] == 4


def test_health_reports_503_when_the_database_is_down(registry):
    dead = TimescaleStore(create_engine("postgresql+psycopg2://nobody:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}))
    assert TestClient(create_app(dead, registry)).get("/health").status_code == 503
