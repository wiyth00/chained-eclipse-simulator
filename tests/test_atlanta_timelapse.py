"""Numerical tests for the Atlanta disk-union calculation."""

from __future__ import annotations

import pytest

from chained_eclipse.atlanta_timelapse import combined_obscuration_fraction


def _disk(east: float, north: float, radius: float) -> dict[str, float]:
    return {
        "east_deg": east,
        "north_deg": north,
        "angular_radius_deg": radius,
        "distance_km": 1.0,
    }


def test_combined_obscuration_counts_overlapping_moons_once() -> None:
    disks = {
        "sun": _disk(0.0, 0.0, 1.0),
        "real_moon": _disk(0.0, 0.0, 0.5),
        "second_moon": _disk(0.0, 0.0, 0.75),
    }

    assert combined_obscuration_fraction(disks) == pytest.approx(0.75**2, abs=2e-4)


def test_combined_obscuration_is_zero_when_both_moons_miss() -> None:
    disks = {
        "sun": _disk(0.0, 0.0, 1.0),
        "real_moon": _disk(3.0, 0.0, 0.5),
        "second_moon": _disk(-3.0, 0.0, 0.75),
    }

    assert combined_obscuration_fraction(disks) == 0.0
