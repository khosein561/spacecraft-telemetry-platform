"""
Read-only HTTP API over the telemetry database.

WHY THIS FILE EXISTS: operators and dashboards need a stable, typed way to ask "what is channel X
doing?" without writing SQL. Grafana talks to the database directly; this API is for scripts and
for demonstrating the data model. It is READ-ONLY on purpose: there is no endpoint that writes
telemetry, because the only way data enters the system is through packets.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy.exc import OperationalError

from ..config import get_settings
from ..packets.channels import limit_state
from ..store.timescale import TimescaleStore, datetime_to_us

MAX_POINTS = 10_000


def create_app(store: TimescaleStore, registry: dict) -> FastAPI:
    app = FastAPI(title="Telemetry API", version="0.1.0")

    @app.get("/health")
    def health():
        try:
            store.ping()
        except OperationalError:
            raise HTTPException(503, "database unavailable")
        return {"status": "ok"}

    @app.get("/channels")
    def channels():
        return [{k: ch[k] for k in ("channel_id", "apid", "spacecraft", "group", "unit")} for ch in registry.values()]

    @app.get("/channels/{channel_id}/readings")
    def readings(channel_id: str, start: datetime | None = None, end: datetime | None = None,
                 limit: int = Query(1000, ge=1, le=MAX_POINTS)):
        ch = registry.get(channel_id)
        if ch is None:
            raise HTTPException(404, f"unknown channel {channel_id!r}")
        for name, value in (("start", start), ("end", end)):
            if value is not None and value.tzinfo is None:
                raise HTTPException(422, f"{name} must include a timezone, e.g. 2026-01-01T00:00:00Z")
        times, values = store.read_channel(
            channel_id, None if start is None else datetime_to_us(start), None if end is None else datetime_to_us(end)
        )
        times, values = times[:limit], values[:limit]
        return {
            "channel_id": channel_id,
            "count": len(values),
            "points": [{"time_us": int(t), "value": float(v), "state": limit_state(float(v), ch)} for t, v in zip(times, values)],
        }

    @app.get("/anomalies")
    def anomalies(channel: str | None = None, detector: str | None = None, limit: int = Query(1000, ge=1, le=MAX_POINTS)):
        return store.list_anomalies(channel, detector, limit)

    @app.get("/packet-health")
    def packet_health():
        return {
            "readings": store.count("telemetry_readings"),
            "events": store.packet_event_counts(),
            "bad_packets": store.count("bad_packets"),
        }

    return app


def build_default_app() -> FastAPI:  # pragma: no cover - needs a live database
    from ..packets.channels import load_channel_registry

    return create_app(TimescaleStore.from_url(get_settings()["database_url"]), load_channel_registry("data/channels.json"))


try:  # `uvicorn telemetry.api.main:app` needs a module-level app; import must not fail without a database
    app = build_default_app()
except Exception:  # pragma: no cover
    app = None
