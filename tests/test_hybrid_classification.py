"""Regression tests for hybrid (annular-total) eclipse classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from chained_eclipse.ephemeris import enumerate_real_solar_eclipses, load_ephemeris

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context():
    return load_ephemeris(PROJECT_ROOT / "data" / "ephemeris")


def _single_event(context, start_utc: str, end_utc: str):
    events = enumerate_real_solar_eclipses(
        context, context.time_utc(start_utc), context.time_utc(end_utc)
    )
    assert len(events) == 1
    return events[0]


def test_2031_nov_14_is_classified_hybrid(context) -> None:
    """NASA/GSFC lists 2031 Nov 14 as hybrid; sign-at-greatest alone says total.

    This test fails on main, where every central eclipse is labelled by the
    sign of the core radius at greatest eclipse only.
    """

    event = _single_event(context, "2031-11-10T00:00:00Z", "2031-11-19T00:00:00Z")
    assert event.event_id == "20311114_real"
    assert event.eclipse_type == "hybrid"
    assert event.latitude_deg is not None
    assert event.longitude_deg is not None


def test_2026_aug_12_remains_total(context) -> None:
    event = _single_event(context, "2026-08-08T00:00:00Z", "2026-08-17T00:00:00Z")
    assert event.event_id == "20260812_real"
    assert event.eclipse_type == "total"


def test_2027_feb_06_remains_annular(context) -> None:
    event = _single_event(context, "2027-02-02T00:00:00Z", "2027-02-11T00:00:00Z")
    assert event.event_id == "20270206_real"
    assert event.eclipse_type == "annular"
