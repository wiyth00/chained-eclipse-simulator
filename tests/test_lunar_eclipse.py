"""Regression checks for coupled lunar-eclipse geometry."""

from __future__ import annotations

from pathlib import Path

import yaml

from chained_eclipse.coupled_eclipse import CoupledEphemeris
from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.lunar_eclipse import find_lunar_eclipses
from chained_eclipse.models import OrbitalElements


ROOT = Path(__file__).resolve().parents[1]


def test_july_2026_second_moon_total_lunar_eclipse() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    config = yaml.safe_load((ROOT / "config" / "optimized_system.yaml").read_text())
    elements = OrbitalElements(**config["orbital_elements"])
    ephemeris = CoupledEphemeris(
        context,
        elements,
        "2026-07-31T00:00:00Z",
        sample_step_seconds=300.0,
    )
    events = find_lunar_eclipses(
        ephemeris,
        "2026-07-29T00:00:00Z",
        "2026-07-31T00:00:00Z",
    )

    assert len(events) == 1
    event = events[0]
    assert event.body == "second_moon"
    assert event.eclipse_type == "total"
    assert event.maximum_utc.startswith("2026-07-30T08:53:51")
    assert 5_700.0 < event.totality_duration_s < 5_730.0
    assert 7.3 < event.other_moon_angular_separation_deg < 7.4
