"""Regression checks for eclipse detection in the coupled four-body model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chained_eclipse.coupled_eclipse import (
    CoupledEphemeris,
    coupled_apparent_geometry,
    coupled_central_point,
    coupled_sky_plane_disks,
    generate_coupled_track,
    scan_coupled_eclipses,
)
from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.models import OrbitalElements


ROOT = Path(__file__).resolve().parents[1]


def _elements() -> OrbitalElements:
    payload = yaml.safe_load((ROOT / "config" / "optimized_system.yaml").read_text())
    return OrbitalElements(**payload["orbital_elements"])


def test_added_moon_backreaction_moves_the_2026_real_eclipse() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    elements = _elements()
    end = "2026-08-14T00:00:00Z"

    massless = CoupledEphemeris(
        context, replace(elements, mass_kg=0.0), end, sample_step_seconds=600.0
    )
    control_event = scan_coupled_eclipses(
        massless,
        "real_moon",
        "2026-08-01T00:00:00Z",
        end,
        step_seconds=300.0,
    )[0]
    assert control_event.axis_maximum_utc.startswith("2026-08-12T17:46")

    coupled = CoupledEphemeris(context, elements, end, sample_step_seconds=600.0)
    coupled_event = scan_coupled_eclipses(
        coupled,
        "real_moon",
        "2026-08-01T00:00:00Z",
        end,
        step_seconds=300.0,
    )[0]
    assert coupled_event.eclipse_type == "total"
    assert coupled_event.axis_maximum_utc.startswith("2026-08-12T20:55")
    central_track = generate_coupled_track(
        coupled,
        "real_moon",
        float(context.time_utc(coupled_event.axis_maximum_utc).tt),
        half_window_hours=1.0,
        step_seconds=600.0,
    )
    assert len(central_track["tt_jd"]) > 5
    assert np.all(np.isfinite(central_track["signed_core_radius_km"]))
    assert abs(coupled.metadata["relative_energy_error"]) < 1e-12


def test_partial_eclipse_gets_a_maximum_surface_track() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    end = "2026-08-22T12:00:00Z"
    coupled = CoupledEphemeris(context, _elements(), end, sample_step_seconds=600.0)
    event = scan_coupled_eclipses(
        coupled,
        "second_moon",
        "2026-08-20T00:00:00Z",
        end,
        step_seconds=300.0,
    )[0]
    assert event.eclipse_type == "partial"

    track = generate_coupled_track(
        coupled,
        "second_moon",
        float(context.time_utc(event.axis_maximum_utc).tt),
        half_window_hours=2.0,
        step_seconds=300.0,
        partial_step_seconds=300.0,
    )
    assert len(track["tt_jd"]) >= 3
    assert np.all(np.isfinite(track["latitude_deg"]))
    assert np.all(np.isfinite(track["longitude_deg"]))
    assert np.all(np.isnan(track["signed_core_radius_km"]))


def test_earth_rotation_offset_shifts_longitude_without_changing_local_geometry() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    end = "2026-08-13T12:00:00Z"
    baseline = CoupledEphemeris(context, _elements(), end, sample_step_seconds=600.0)
    event = scan_coupled_eclipses(
        baseline,
        "real_moon",
        "2026-08-12T00:00:00Z",
        end,
        step_seconds=300.0,
    )[0]
    maximum_tt = float(context.time_utc(event.axis_maximum_utc).tt)
    baseline_point = coupled_central_point(baseline, "real_moon", maximum_tt)
    assert baseline_point is not None

    class OffsetEphemeris:
        def __getattr__(self, name):
            return getattr(baseline, name)

        @staticmethod
        def longitude_offset_deg(tt_jd):
            return np.zeros_like(np.asarray(tt_jd, dtype=float)) + 10.0

    shifted = OffsetEphemeris()
    shifted_point = coupled_central_point(shifted, "real_moon", maximum_tt)
    assert shifted_point is not None
    assert shifted_point[0] == pytest.approx(baseline_point[0], abs=1.0e-10)
    longitude_delta = (shifted_point[1] - baseline_point[1] + 180.0) % 360.0 - 180.0
    assert longitude_delta == pytest.approx(10.0, abs=1.0e-10)
    assert shifted_point[3] == pytest.approx(baseline_point[3], abs=1.0e-10)

    baseline_geometry = coupled_apparent_geometry(
        baseline,
        "real_moon",
        maximum_tt,
        baseline_point[0],
        baseline_point[1],
    )
    shifted_geometry = coupled_apparent_geometry(
        shifted,
        "real_moon",
        maximum_tt,
        shifted_point[0],
        shifted_point[1],
    )
    assert shifted_geometry.separation_rad == pytest.approx(
        baseline_geometry.separation_rad, abs=1.0e-12
    )
    assert shifted_geometry.solar_altitude_deg == pytest.approx(
        baseline_geometry.solar_altitude_deg, abs=1.0e-10
    )

    disks = coupled_sky_plane_disks(
        baseline,
        maximum_tt,
        baseline_point[0],
        baseline_point[1],
    )
    assert disks["sun"]["east_deg"] == pytest.approx(0.0, abs=1.0e-14)
    assert disks["sun"]["north_deg"] == pytest.approx(0.0, abs=1.0e-14)
    sky_offset_deg = np.hypot(
        disks["real_moon"]["east_deg"],
        disks["real_moon"]["north_deg"],
    )
    assert sky_offset_deg == pytest.approx(
        np.degrees(baseline_geometry.separation_rad),
        rel=1.0e-4,
    )
