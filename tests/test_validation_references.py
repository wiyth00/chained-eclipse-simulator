"""Timing-only validation behavior for non-central (partial) NASA references."""

from __future__ import annotations

from pathlib import Path

import pytest

from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.models import RealEclipse
from chained_eclipse.validation import NASA_GSFC_REFERENCES, _validate_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context():
    return load_ephemeris(PROJECT_ROOT / "data" / "ephemeris")


def _partial_reference():
    matches = [item for item in NASA_GSFC_REFERENCES if item.eclipse_type == "partial"]
    assert len(matches) == 1
    return matches[0]


def test_partial_reference_identity() -> None:
    reference = _partial_reference()
    assert reference.event_id == "20250329_real"
    assert reference.latitude_deg is None
    assert reference.longitude_deg is None
    assert reference.central_duration_s is None


def test_first_reference_is_pre_epoch_2024_total() -> None:
    first = NASA_GSFC_REFERENCES[0]
    assert first.event_id == "20240408_real"
    assert first.eclipse_type == "total"
    # Pre-epoch anchor: earlier than the model epoch (2026-07-10).
    assert first.published_greatest_tdt.startswith("2024-")


def test_partial_event_validates_without_coordinates(context) -> None:
    """Fails on main: _validate_one raises for events without a central line."""

    reference = _partial_reference()
    event = RealEclipse(
        event_id=reference.event_id,
        maximum_utc="2025-03-29T10:47:24.700Z",
        eclipse_type="partial",
        latitude_deg=None,
        longitude_deg=None,
        axis_distance_km=7_000.0,
        penumbra_margin_km=100.0,
        core_margin_km=100.0,
        solar_altitude_deg=None,
    )
    result = _validate_one(context, reference, event)
    assert result.coordinate_error_wgs84_km is None
    assert result.model_central_duration_s is None
    assert result.central_duration_error_s is None
    assert result.coordinate_within_target is None
    assert result.type_matches
    # The fabricated maximum reuses NASA's UT label as a UTC string, so its
    # TT differs from the published TDT by roughly Delta-T minus the
    # TT-minus-UTC offset (about -1.8 s here) -- well inside the 60 s target.
    assert result.timing_within_target
    assert result.passed == (result.type_matches and result.timing_within_target)
    assert result.passed is True


def test_central_reference_still_requires_central_event(context) -> None:
    """A central reference matched to a partial event must still raise."""

    central = NASA_GSFC_REFERENCES[0]
    event = RealEclipse(
        event_id=central.event_id,
        maximum_utc="2024-04-08T18:17:18.300Z",
        eclipse_type="partial",
        latitude_deg=None,
        longitude_deg=None,
        axis_distance_km=7_000.0,
        penumbra_margin_km=100.0,
        core_margin_km=100.0,
        solar_altitude_deg=None,
    )
    with pytest.raises(ValueError, match="not classified as a central eclipse"):
        _validate_one(context, central, event)
