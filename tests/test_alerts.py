"""Alert de-duplication, cooldown, failure handling."""
import json

import pytest

from telemetry.alerts.notify import Alert, AlertManager, Anomaly, WebhookSender

MIN = 60_000_000
T0 = 1_767_225_600_000_000


class Recorder:
    name = "rec"

    def __init__(self, fail=False):
        self.alerts, self.fail = [], fail

    def send(self, alert):
        if self.fail:
            raise ConnectionError("down")
        self.alerts.append(alert)


def a(ch="A-1", det="lstm", start=0, end=10, score=3.0):
    return Anomaly(ch, det, T0 + start * MIN, T0 + end * MIN, score)


def test_first_anomaly_alerts_and_overlapping_ones_are_suppressed():
    r = Recorder()
    m = AlertManager([r], cooldown_s=1800)
    assert m.handle(a(start=0, end=10)) is not None
    assert m.handle(a(start=5, end=12)) is None                    # overlaps the incident
    assert m.handle(a(start=20, end=25)) is None                   # within 30 min of its end
    assert (m.sent, m.suppressed, len(r.alerts)) == (1, 2, 1)


def test_realerts_after_cooldown_expires():
    m = AlertManager([Recorder()], cooldown_s=1800)
    m.handle(a(start=0, end=10))
    assert m.handle(a(start=10 + 31, end=10 + 40)) is not None


def test_ongoing_incident_keeps_extending_the_quiet_period():
    m = AlertManager([Recorder()], cooldown_s=1800)
    m.handle(a(start=0, end=10))
    for k in range(1, 6):                                          # a new fragment every 25 min for two hours
        assert m.handle(a(start=10 + 25 * k, end=10 + 25 * k + 5)) is None
    assert m.sent == 1


def test_channels_and_detectors_are_independent():
    r = Recorder()
    m = AlertManager([r])
    m.handle(a("A-1", "lstm", 0, 10))
    assert m.handle(a("B-2", "lstm", 0, 10)) is not None
    assert m.handle(a("A-1", "zscore", 0, 10)) is not None
    assert len(r.alerts) == 3


def test_severity_from_score():
    m = AlertManager([Recorder()], critical_score=8.0)
    assert m.handle(a(score=3.0)).severity == "warning"
    assert m.handle(a("B-2", score=9.5)).severity == "critical"


def test_failed_delivery_does_not_start_the_cooldown_and_never_raises():
    bad, m = Recorder(fail=True), None
    m = AlertManager([bad], cooldown_s=1800)
    assert m.handle(a(start=0, end=10)) is None and m.failed == 1
    bad.fail = False
    assert m.handle(a(start=11, end=15)) is not None               # the retry gets through


def test_one_failing_destination_does_not_block_the_other():
    good, bad = Recorder(), Recorder(fail=True)
    m = AlertManager([bad, good])
    assert m.handle(a()) is not None and len(good.alerts) == 1


def test_process_sorts_by_time_so_input_order_does_not_matter():
    m = AlertManager([Recorder()])
    out = m.process([a(start=100, end=110), a(start=0, end=10), a(start=35, end=40)])
    assert [x.event_us for x in out] == [T0, T0 + 100 * MIN]       # minute 35 is 25 min after the first incident ended: inside the 30 min cooldown


def test_recorder_receives_every_delivery_attempt():
    log = []
    m = AlertManager([Recorder(fail=True), Recorder()], recorder=log.append)
    m.handle(a())
    assert sorted(x.status for x in log) == ["failed", "sent"]


def test_webhook_payload_and_url_validation():
    sent = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *e): return False

    def fake_open(req, timeout):
        sent["url"], sent["body"], sent["ct"] = req.full_url, json.loads(req.data), req.headers["Content-type"]
        return Resp()

    WebhookSender("https://hooks.example.test/x", opener=fake_open).send(Alert("A-1", "lstm", "warning", 0, "webhook", "sent", "hello"))
    assert sent["body"] == {"text": "hello", "content": "hello"} and sent["ct"] == "application/json"
    for bad in ("file:///etc/passwd", "ftp://x", "javascript:alert(1)"):
        with pytest.raises(ValueError):
            WebhookSender(bad)


def test_alerts_are_persisted(store):
    m = AlertManager([Recorder()], recorder=store.insert_alert)
    m.handle(a())
    assert store.count("alerts") == 1
