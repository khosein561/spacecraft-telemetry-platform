"""
Alerting with de-duplication and cooldown.

WHY THIS FILE EXISTS: an anomaly detector can fire dozens of times for one incident, and an
on-call operator who is paged 40 times for one event learns to ignore pages. The AlertManager
turns a stream of detections into a small number of alerts:

  * DE-DUPLICATION: anomalies on the same (channel, detector) that overlap or follow a recent one
    belong to the same incident, so they do not page again;
  * COOLDOWN: after an alert, the same (channel, detector) stays quiet for `cooldown_s`;
  * SEVERITY: 'critical' if the score is at least `critical_score`, else 'warning'.

Time is SPACECRAFT time (the anomaly's own timestamps), not the wall clock, so replaying a week of
data in seconds behaves exactly like living through that week.

A failed delivery (webhook down) is recorded as 'failed' and does NOT start the cooldown, so the
next anomaly gets another chance to reach a human. Sending never raises into the detector.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Anomaly:
    channel_id: str
    detector: str
    start_us: int
    end_us: int
    score: float = 0.0


@dataclass(frozen=True)
class Alert:
    channel_id: str
    detector: str
    severity: str
    event_us: int
    destination: str
    status: str  # 'sent' | 'failed'
    text: str


class LogSender:
    name = "log"

    def send(self, alert: Alert) -> None:
        log.warning("ALERT %s", alert.text)


class WebhookSender:
    """POST JSON to a Slack/Discord/Teams-style incoming webhook (http/https URLs only)."""

    name = "webhook"

    def __init__(self, url: str, opener=urllib.request.urlopen, timeout: float = 5.0):
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("webhook URL must start with http:// or https://")
        self.url, self._open, self.timeout = url, opener, timeout

    def send(self, alert: Alert) -> None:
        body = json.dumps({"text": alert.text, "content": alert.text}).encode()  # 'text' = Slack, 'content' = Discord
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with self._open(req, timeout=self.timeout) as resp:
            if getattr(resp, "status", 200) >= 300:
                raise RuntimeError(f"webhook returned HTTP {resp.status}")


class AlertManager:
    def __init__(self, senders=None, cooldown_s: float = 1800.0, critical_score: float = 8.0, recorder=None):
        self.senders = list(senders) if senders is not None else [LogSender()]
        self.cooldown_us = int(cooldown_s * 1_000_000)
        self.critical_score = critical_score
        self.recorder = recorder  # optional callable(Alert) that persists alerts
        self._last_end: dict = {}   # (channel, detector) -> end of the incident we last alerted on
        self.sent = 0
        self.suppressed = 0
        self.failed = 0

    def _severity(self, a: Anomaly) -> str:
        return "critical" if a.score >= self.critical_score else "warning"

    def handle(self, a: Anomaly):
        """Return the Alert that was sent, or None if suppressed (or if every destination failed)."""
        key = (a.channel_id, a.detector)
        last_end = self._last_end.get(key)
        if last_end is not None and a.start_us <= last_end + self.cooldown_us:
            self.suppressed += 1
            self._last_end[key] = max(last_end, a.end_us)       # the incident is still going: extend its quiet period
            return None
        severity = self._severity(a)
        text = f"[{severity.upper()}] {a.channel_id}: {a.detector} anomaly (score {a.score:.1f})"
        results = []
        for sender in self.senders:
            try:
                sender.send(Alert(a.channel_id, a.detector, severity, a.start_us, sender.name, "sent", text))
                results.append((sender.name, "sent"))
            except Exception as exc:  # noqa: BLE001 - a broken webhook must never crash detection
                log.error("alert delivery via %s failed: %s", sender.name, exc)
                results.append((sender.name, "failed"))
        delivered = any(status == "sent" for _, status in results)
        alerts = [Alert(a.channel_id, a.detector, severity, a.start_us, name, status, text) for name, status in results]
        if self.recorder:
            for al in alerts:
                self.recorder(al)
        if delivered:
            self._last_end[key] = a.end_us
            self.sent += 1
            return next(al for al in alerts if al.status == "sent")
        self.failed += 1                                          # nobody was told: do not start the cooldown
        return None

    def process(self, anomalies) -> list:
        return [x for x in (self.handle(a) for a in sorted(anomalies, key=lambda a: (a.start_us, a.channel_id))) if x]
