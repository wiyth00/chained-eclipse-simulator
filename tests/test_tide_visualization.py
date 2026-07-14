from __future__ import annotations

import numpy as np
import pytest

from chained_eclipse.tide_visualization import (
    EARTH_MEAN_RADIUS_KM,
    STANDARD_GRAVITY_KM_S2,
    TideBodyState,
    equilibrium_tide_height_m,
    exact_equilibrium_tide_height_m,
    surface_unit_vectors,
    wgs84_surface_positions,
)
from chained_eclipse.tides_spin import G_KM3_KG_S2


def _state(unit: tuple[float, float, float], amplitude: float) -> TideBodyState:
    return TideBodyState(
        name="test",
        mass_kg=1.0,
        distance_km=1.0,
        subpoint_latitude_deg=0.0,
        subpoint_longitude_deg=0.0,
        subpoint_amplitude_m=amplitude,
        unit_itrs=np.asarray(unit, dtype=float),
    )


def test_degree_two_tide_has_two_high_bulges_and_equatorial_lows() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = equilibrium_tide_height_m(points, (_state((1.0, 0.0, 0.0), 0.4),))

    assert result == pytest.approx([0.4, 0.4, -0.2, -0.2])


def test_aligned_moon_fields_add_linearly() -> None:
    point = np.asarray([[1.0, 0.0, 0.0]])
    result = equilibrium_tide_height_m(
        point,
        (_state((1.0, 0.0, 0.0), 0.36), _state((-1.0, 0.0, 0.0), 0.39)),
    )

    assert result[0] == pytest.approx(0.75)


def test_surface_grid_is_global_and_unit_normalized() -> None:
    longitude, latitude, unit = surface_unit_vectors(1.0)

    assert longitude.shape == latitude.shape == unit.shape[:2] == (181, 361)
    assert longitude[0, 0] == -180.0
    assert longitude[0, -1] == 180.0
    assert latitude[0, 0] == -90.0
    assert latitude[-1, 0] == 90.0
    assert np.linalg.norm(unit, axis=-1) == pytest.approx(np.ones((181, 361)))


def test_exact_tide_is_zero_mean_and_close_to_degree_two_in_far_field() -> None:
    longitude, latitude, unit = surface_unit_vectors(5.0)
    positions, weights = wgs84_surface_positions(longitude, latitude)
    state = _state((1.0, 0.0, 0.0), 0.4)
    mass_kg = 7.342e22
    distance_km = 3_844_000.0
    amplitude_m = (
        G_KM3_KG_S2
        * mass_kg
        * EARTH_MEAN_RADIUS_KM**2
        / (STANDARD_GRAVITY_KM_S2 * distance_km**3)
        * 1_000.0
    )
    state = TideBodyState(
        name=state.name,
        mass_kg=mass_kg,
        distance_km=distance_km,
        subpoint_latitude_deg=state.subpoint_latitude_deg,
        subpoint_longitude_deg=state.subpoint_longitude_deg,
        subpoint_amplitude_m=amplitude_m,
        unit_itrs=state.unit_itrs,
    )
    exact = exact_equilibrium_tide_height_m(positions, weights, (state,))
    degree_two = equilibrium_tide_height_m(unit, (state,))

    assert np.average(exact, weights=weights) == pytest.approx(0.0, abs=1.0e-12)
    # At ten lunar distances, higher degrees are well below one percent.
    assert np.max(np.abs(exact - degree_two)) < 1.0e-5
