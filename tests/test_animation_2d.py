"""Fast numerical tests for the 2-D eclipse-footprint renderer."""

from __future__ import annotations

import numpy as np

from chained_eclipse.animation_2d import SurfaceGrid, _overlap_fraction


def test_overlap_fraction_covers_total_partial_and_none() -> None:
    separation = np.asarray((0.0, 0.75, 2.1))
    sun_radius = np.ones(3)
    moon_radius = np.ones(3)
    result = _overlap_fraction(separation, sun_radius, moon_radius)

    assert result[0] == 1.0
    assert 0.0 < result[1] < 1.0
    assert result[2] == 0.0


def test_surface_grid_detects_aligned_daylight_eclipse() -> None:
    grid = SurfaceGrid(step_deg=5.0)
    latitude = np.radians(60.0)
    longitude = np.radians(-20.0)
    direction = np.asarray(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    sun = direction * 149_600_000.0
    moon = direction * 180_000.0
    obscuration, central = grid.obscuration(sun, moon, 838.0)

    assert obscuration.shape == grid.shape
    assert float(np.max(obscuration)) > 0.95
    assert np.any(central)
    assert np.all(obscuration >= 0.0)
    assert np.all(obscuration <= 1.0)
