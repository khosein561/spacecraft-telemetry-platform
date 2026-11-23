"""
Satellite pass prediction and contact-window gating (stretch goal).

WHY THIS FILE EXISTS: a real low-earth-orbit spacecraft can only talk to one ground station during
short "passes" overhead. In between, it records data on board and sends it (store-and-forward)
when a pass begins. This module predicts passes from orbital elements (a TLE) with skyfield/SGP4
and models that behaviour: samples are held until the next pass and then released in order.

NOTE ON DATA: the SMAP/MSL arrays have no orbit. The TLE used in tests/demos is SYNTHETIC
(`synthetic_leo_tle`, an ISS-like 51.6 degree orbit generated from elements). It is not a real
satellite and says nothing about when any real mission is overhead. For real predictions, fetch
a current TLE from CelesTrak (`fetch_tle`, network required, not exercised by the tests).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone

SEATTLE = (47.6062, -122.3321)


@dataclass(frozen=True)
class Pass:
    rise_us: int
    culminate_us: int
    set_us: int
    max_elevation_deg: float

    def contains(self, t_us: int) -> bool:
        return self.rise_us <= t_us <= self.set_us


def _to_us(t) -> int:
    dt = t.utc_datetime()
    return int(round(dt.timestamp() * 1_000_000))


def synthetic_leo_tle(epoch: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc), mean_motion_rev_per_day: float = 15.5,
                      inclination_deg: float = 51.64) -> tuple:
    """Build a valid two-line element set for an invented ISS-like orbit (offline, deterministic)."""
    from sgp4.api import WGS72, Satrec
    from sgp4.exporter import export_tle

    sat = Satrec()
    days = (epoch - datetime(1949, 12, 31, tzinfo=timezone.utc)).total_seconds() / 86400
    sat.sgp4init(WGS72, "i", 99001, days, 3e-5, 0.0, 0.0, 0.0006, math.radians(90), math.radians(inclination_deg),
                 math.radians(10), mean_motion_rev_per_day * 2 * math.pi / 1440, math.radians(120))
    return export_tle(sat)


def predict_passes(line1: str, line2: str, start: datetime, end: datetime, latlon=SEATTLE, min_elevation_deg: float = 10.0) -> list:
    """Complete passes (rise -> culmination -> set) over `latlon` between start and end, in time order.

    Passes cut off by the edges of the window are dropped rather than reported with invented times.
    """
    from skyfield.api import EarthSatellite, load, wgs84

    if end <= start:
        raise ValueError("end must be after start")
    ts = load.timescale(builtin=True)  # builtin: no network download needed
    sat = EarthSatellite(line1, line2, "SAT", ts)
    site = wgs84.latlon(*latlon)
    t0, t1 = ts.from_datetime(start), ts.from_datetime(end)
    times, events = sat.find_events(site, t0, t1, altitude_degrees=min_elevation_deg)
    passes, current = [], {}
    for t, ev in zip(times, events):
        if ev == 0:
            current = {"rise": _to_us(t)}
        elif ev == 1 and "rise" in current:
            alt, _, _ = (sat - site).at(t).altaz()
            current["culm"], current["alt"] = _to_us(t), float(alt.degrees)
        elif ev == 2 and "culm" in current:
            passes.append(Pass(current["rise"], current["culm"], _to_us(t), current["alt"]))
            current = {}
    return passes


def in_contact(t_us: int, passes: list) -> bool:
    """Is spacecraft time t inside any pass? (passes must be sorted and non-overlapping)"""
    i = bisect_right([p.rise_us for p in passes], t_us) - 1
    return i >= 0 and passes[i].contains(t_us)


def delivery_times(times_us, passes: list) -> list:
    """Store-and-forward: when does each sample reach the ground?

    A sample recorded during a pass is delivered immediately; one recorded between passes waits for
    the next pass to begin. Samples after the final pass stay on board (None).
    """
    rises = [p.rise_us for p in passes]
    out = []
    for t in times_us:
        t = int(t)
        if in_contact(t, passes):
            out.append(t)
            continue
        j = bisect_right(rises, t)
        out.append(rises[j] if j < len(rises) else None)
    return out


def downlink_order(times_us, passes: list) -> list:
    """Indices of the samples that get delivered, in the order the ground receives them:
    by delivery time, then by spacecraft time (so each pass dumps its backlog in recorded order)."""
    deliv = delivery_times(times_us, passes)
    delivered = [i for i, d in enumerate(deliv) if d is not None]
    return sorted(delivered, key=lambda i: (deliv[i], int(times_us[i])))


def fetch_tle(url: str = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle") -> tuple:  # pragma: no cover
    """Download a real TLE (network required)."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=20) as resp:
        lines = [ln.strip() for ln in resp.read().decode().splitlines() if ln.strip()]
    return lines[-2], lines[-1]
