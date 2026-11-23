"""Pass prediction (offline, synthetic orbit) and store-and-forward downlink."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from telemetry.orbit.passes import SEATTLE, Pass, delivery_times, downlink_order, in_contact, predict_passes, synthetic_leo_tle

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def passes():
    l1, l2 = synthetic_leo_tle()
    return predict_passes(l1, l2, START, START + timedelta(days=1), SEATTLE, 10.0)


def test_tle_is_valid_and_deterministic():
    l1, l2 = synthetic_leo_tle()
    assert l1.startswith("1 ") and l2.startswith("2 ") and len(l1) == 69 and len(l2) == 69
    assert (l1, l2) == synthetic_leo_tle()


def test_passes_are_sane(passes):
    assert 3 <= len(passes) <= 8                                    # a 51.6 degree LEO over Seattle: a handful per day
    for p in passes:
        assert p.rise_us < p.culminate_us < p.set_us
        assert 2 * 60e6 < p.set_us - p.rise_us < 15 * 60e6          # LEO passes last minutes, not hours
        assert 10.0 <= p.max_elevation_deg <= 90.0
    assert all(passes[i].set_us < passes[i + 1].rise_us for i in range(len(passes) - 1))   # ordered, non-overlapping


def test_equatorial_orbit_never_rises_over_seattle():
    """Geometry check: a 0.5 degree inclination orbit stays near the equator, far from latitude 47.6 N."""
    l1, l2 = synthetic_leo_tle(inclination_deg=0.5)
    equatorial = predict_passes(l1, l2, START, START + timedelta(days=1), SEATTLE, 10.0)
    assert len(equatorial) == 0


def test_bad_window_rejected():
    l1, l2 = synthetic_leo_tle()
    with pytest.raises(ValueError):
        predict_passes(l1, l2, START, START)


def test_in_contact_boundaries():
    ps = [Pass(100, 150, 200, 40.0), Pass(500, 550, 600, 60.0)]
    assert in_contact(100, ps) and in_contact(200, ps) and in_contact(550, ps)
    assert not in_contact(99, ps) and not in_contact(201, ps) and not in_contact(450, ps) and not in_contact(700, ps)


def test_store_and_forward_delivery():
    ps = [Pass(1000, 1500, 2000, 40.0), Pass(5000, 5500, 6000, 60.0)]
    times = [100, 1200, 3000, 5200, 9000]
    d = delivery_times(times, ps)
    assert d == [1000, 1200, 5000, 5200, None]                      # waits for the next pass; after the last pass: stays on board
    order = downlink_order(times, ps)
    assert order == [0, 1, 2, 3]                                     # backlog released in recorded order; sample 4 never delivered


def test_downlinked_stream_preserves_recorded_order_within_each_pass(passes):
    rng = np.random.default_rng(0)
    t0 = int(START.timestamp() * 1_000_000)
    times = t0 + np.sort(rng.integers(0, 86_400_000_000, 3000))
    order = downlink_order(times, passes)
    deliv = delivery_times(times, passes)
    assert len(order) == sum(d is not None for d in deliv)
    seq = [int(times[i]) for i in order]
    assert seq == sorted(seq)                                        # store-and-forward never reorders a single channel
    assert all(deliv[i] >= times[i] for i in order)
