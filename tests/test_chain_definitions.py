"""Regression tests for the at-least-one-total chained-eclipse requirement."""

from dataclasses import replace

from chained_eclipse.models import LocalCircumstances
from chained_eclipse.search import build_event


def _local(eclipse_type: str, maximum_utc: str) -> LocalCircumstances:
    return LocalCircumstances(
        body="test",
        eclipse_type=eclipse_type,
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        c1_utc="2030-01-01T11:00:00Z",
        c2_utc=None,
        maximum_utc=maximum_utc,
        c3_utc=None,
        c4_utc="2030-01-01T13:00:00Z",
        magnitude=0.7,
        obscuration=0.6,
        central_duration_s=0.0,
        solar_altitude_deg=30.0,
        center_separation_deg=0.1,
        solar_angular_diameter_deg=0.53,
        moon_angular_diameter_deg=0.50,
    )


def test_partial_only_pair_fails_every_definition_even_when_tracks_overlap() -> None:
    real = _local("partial", "2030-01-01T12:00:00Z")
    second = _local("partial", "2030-01-01T12:30:00Z")

    event = build_event(
        "partial-only",
        real,
        second,
        track_distance_km=10.0,
        latitude_deg=0.0,
        longitude_deg=0.0,
    )

    assert event.definition_a is False
    assert event.definition_b is False
    assert not any(event.thresholds.values())


def test_one_local_total_enables_same_location_and_regional_definitions() -> None:
    real = replace(
        _local("total", "2030-01-01T12:00:00Z"),
        c2_utc="2030-01-01T11:59:00Z",
        c3_utc="2030-01-01T12:01:00Z",
        central_duration_s=120.0,
    )
    second = _local("partial", "2030-01-01T12:30:00Z")

    event = build_event(
        "one-total",
        real,
        second,
        track_distance_km=10.0,
        latitude_deg=0.0,
        longitude_deg=0.0,
    )

    assert event.definition_a is True
    assert event.definition_b is True
    assert all(event.thresholds.values())
