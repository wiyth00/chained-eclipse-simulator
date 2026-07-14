"""Focused tests for ephemeris loading and independent eclipse enumeration."""

from pathlib import Path

import pytest

from chained_eclipse.ephemeris import enumerate_real_solar_eclipses, load_ephemeris

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context():
    return load_ephemeris(ROOT / "data" / "ephemeris")


def test_de440s_is_cached_locally(context) -> None:
    assert context.kernel_path.exists()
    assert context.kernel_path.stat().st_size > 30_000_000


def test_first_post_epoch_eclipses_are_numerically_recovered(context) -> None:
    events = enumerate_real_solar_eclipses(
        context,
        context.time_utc("2026-07-10T00:00:00Z"),
        context.time_utc("2027-09-01T00:00:00Z"),
    )
    assert [(event.event_id, event.eclipse_type) for event in events] == [
        ("20260812_real", "total"),
        ("20270206_real", "annular"),
        ("20270802_real", "total"),
    ]
    assert events[0].latitude_deg == pytest.approx(65.225, abs=0.03)
    assert events[0].longitude_deg == pytest.approx(-25.2283, abs=0.03)

