"""Fast tests for the 3-D animation's physical data plumbing."""

from __future__ import annotations

import json

import numpy as np
import yaml

from chained_eclipse.animation import _cone_surface, load_animation_inputs
from chained_eclipse.constants import SUN_RADIUS_KM


def test_load_animation_inputs_uses_fixed_event_and_saved_elements(tmp_path) -> None:
    events = {
        "fixed_system_events": [
            {
                "event_id": "fixed-1",
                "real_eclipse": {"maximum_utc": "2026-08-12T17:45:56Z"},
                "second_eclipse": {"maximum_utc": "2026-08-12T17:57:57Z"},
                "best_latitude_deg": 65.2,
                "best_longitude_deg": -25.2,
            }
        ]
    }
    config = {
        "orbital_elements": {
            "semimajor_axis_km": 180_000.0,
            "eccentricity": 0.04,
            "inclination_deg": 5.0,
            "longitude_ascending_node_deg": 119.0,
            "argument_periapsis_deg": 320.0,
            "mean_anomaly_deg": 115.0,
            "epoch_utc": "2026-07-10T00:00:00Z",
        }
    }
    events_path = tmp_path / "events.json"
    config_path = tmp_path / "optimized.yaml"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    event, elements = load_animation_inputs(events_path, config_path)

    assert event.event_id == "fixed-1"
    assert event.latitude_deg == 65.2
    assert elements.semimajor_axis_km == 180_000.0
    assert elements.mean_anomaly_deg == 115.0


def test_core_cone_uses_physical_opening_angle() -> None:
    moon_position = np.asarray((100_000.0, 0.0, 0.0))
    axis = np.asarray((-1.0, 0.0, 0.0))
    moon_radius = 1_000.0
    sun_distance = 150_000_000.0

    x, y, z, signed_radius = _cone_surface(
        moon_position,
        axis,
        moon_radius,
        sun_distance,
        near_earth_only=False,
        radial_samples=5,
        azimuth_samples=12,
    )

    angle = np.arcsin((SUN_RADIUS_KM - moon_radius) / sun_distance)
    expected_at_moon = moon_radius / np.cos(angle)
    assert np.isclose(signed_radius[0], expected_at_moon)
    assert x.shape == y.shape == z.shape == (5, 12)
    assert signed_radius[-1] < signed_radius[0]

