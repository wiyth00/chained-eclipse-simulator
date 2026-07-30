"""Cross-checks between the project's duplicate numeric implementations.

The repository intentionally carries two Kepler solvers (vectorized in
``orbital_dynamics``, scalar in ``stability``, which documents standalone
capability) and two element->state converters.  These tests pin the pairs to
each other so any future divergence fails CI, and they pin the canonical
pyproj/Karney geodesic to the hand-rolled Vincenty inverse that previously
lived in ``validation.py``, preserved below as an independent reference.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from chained_eclipse.constants import MU_EARTH_KM3_S2, WGS84_A_KM, WGS84_F
from chained_eclipse.models import OrbitalElements
from chained_eclipse.orbital_dynamics import elements_to_state, solve_kepler
from chained_eclipse.stability import _solve_kepler, orbital_elements_to_icrf_state
from chained_eclipse.validation import NASA_GSFC_REFERENCES, wgs84_geodesic_distance_km

_REFERENCE_SITES = tuple(
    (reference.latitude_deg, reference.longitude_deg)
    for reference in NASA_GSFC_REFERENCES
    if reference.latitude_deg is not None and reference.longitude_deg is not None
)


def _vincenty_reference_km(
    latitude_1_deg: float,
    longitude_1_deg: float,
    latitude_2_deg: float,
    longitude_2_deg: float,
) -> float:
    """Vincenty inverse distance on the WGS84 reference ellipsoid.

    This is the implementation removed from ``validation.py`` when
    ``pyproj.Geod`` became the project's single canonical geodesic.  It is
    kept verbatim as an independent reference for the cross-check below.
    Validation points are well away from Vincenty's near-antipodal
    non-convergence case.
    """

    if latitude_1_deg == latitude_2_deg and longitude_1_deg == longitude_2_deg:
        return 0.0

    semi_major = WGS84_A_KM
    flattening = WGS84_F
    semi_minor = semi_major * (1.0 - flattening)
    phi_1 = math.radians(latitude_1_deg)
    phi_2 = math.radians(latitude_2_deg)
    longitude_delta = math.radians(
        (longitude_2_deg - longitude_1_deg + 180.0) % 360.0 - 180.0
    )
    reduced_1 = math.atan((1.0 - flattening) * math.tan(phi_1))
    reduced_2 = math.atan((1.0 - flattening) * math.tan(phi_2))
    sin_u1, cos_u1 = math.sin(reduced_1), math.cos(reduced_1)
    sin_u2, cos_u2 = math.sin(reduced_2), math.cos(reduced_2)
    longitude = longitude_delta

    for _ in range(100):
        sin_longitude = math.sin(longitude)
        cos_longitude = math.cos(longitude)
        sin_sigma = math.hypot(
            cos_u2 * sin_longitude,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_longitude,
        )
        if sin_sigma == 0.0:
            return 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_longitude
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_longitude / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
        cos_two_sigma_m = (
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            if cos_sq_alpha > 1e-16
            else 0.0
        )
        coefficient = flattening / 16.0 * cos_sq_alpha * (
            4.0 + flattening * (4.0 - 3.0 * cos_sq_alpha)
        )
        previous = longitude
        longitude = longitude_delta + (1.0 - coefficient) * flattening * sin_alpha * (
            sigma
            + coefficient
            * sin_sigma
            * (
                cos_two_sigma_m
                + coefficient
                * cos_sigma
                * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
            )
        )
        if abs(longitude - previous) < 1e-12:
            break
    else:
        raise RuntimeError("WGS84 Vincenty inverse failed to converge")

    reduced_sq = cos_sq_alpha * (
        (semi_major * semi_major - semi_minor * semi_minor) / (semi_minor * semi_minor)
    )
    series_a = 1.0 + reduced_sq / 16_384.0 * (
        4_096.0 + reduced_sq * (-768.0 + reduced_sq * (320.0 - 175.0 * reduced_sq))
    )
    series_b = reduced_sq / 1_024.0 * (
        256.0 + reduced_sq * (-128.0 + reduced_sq * (74.0 - 47.0 * reduced_sq))
    )
    delta_sigma = series_b * sin_sigma * (
        cos_two_sigma_m
        + series_b
        / 4.0
        * (
            cos_sigma * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
            - series_b
            / 6.0
            * cos_two_sigma_m
            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
            * (-3.0 + 4.0 * cos_two_sigma_m * cos_two_sigma_m)
        )
    )
    return semi_minor * series_a * (sigma - delta_sigma)


def _wrapped_difference_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def test_kepler_solvers_agree_on_grid() -> None:
    """Both Kepler solvers must agree to < 1e-12 rad across (e, M) space."""

    eccentricities = (0.0, 0.01, 0.04, 0.2, 0.5, 0.79, 0.8, 0.85, 0.9, 0.95, 0.99)
    worst = 0.0
    for eccentricity in eccentricities:
        for mean_deg in np.linspace(0.0, 359.5, 720):
            mean = math.radians(float(mean_deg))
            vectorized = float(solve_kepler(mean, eccentricity))
            scalar = _solve_kepler(mean, eccentricity)
            worst = max(worst, _wrapped_difference_rad(vectorized, scalar))
    assert worst < 1e-12


def test_solve_kepler_accepts_scalar_high_eccentricity() -> None:
    """Regression: scalar input with e > 0.8 raised TypeError before the fix.

    NumPy ufuncs collapse 0-d inputs to scalars, so the high-eccentricity
    starting-guess item assignment failed for scalar mean anomalies.
    """

    eccentric = float(solve_kepler(1.0, 0.9))
    assert abs(eccentric - 0.9 * math.sin(eccentric) - 1.0) < 1e-12


def test_elements_to_state_accepts_high_eccentricity() -> None:
    """Regression: elements_to_state crashed for e > 0.8 before the fix."""

    elements = OrbitalElements(eccentricity=0.9, mean_anomaly_deg=57.0)
    state = elements_to_state(elements)
    assert state.shape == (6,)
    assert bool(np.all(np.isfinite(state)))


@pytest.mark.parametrize("eccentricity", [0.04, 0.9])
def test_solve_kepler_vectorized_matches_scalar_calls(eccentricity: float) -> None:
    means = np.linspace(0.0, 2.0 * np.pi, 97)
    vectorized = np.asarray(solve_kepler(means, eccentricity))
    one_by_one = np.asarray([float(solve_kepler(mean, eccentricity)) for mean in means])
    assert float(np.max(np.abs(vectorized - one_by_one))) < 1e-13


def test_element_to_state_paths_agree() -> None:
    """The two element->state converters must agree to < 1 mm and < 1 um/s."""

    worst_position_km = 0.0
    worst_velocity_km_s = 0.0
    for a, e, inc, node, argp, mean in itertools.product(
        (8_000.0, 42_164.0, 180_000.0, 384_400.0),
        (0.0, 0.04, 0.3, 0.7, 0.9),
        (0.0, 5.0, 28.5, 63.4, 90.0, 150.0),
        (0.0, 45.0, 123.4, 270.0),
        (0.0, 90.0, 251.0),
        (0.0, 10.0, 179.0, 181.0, 350.0),
    ):
        elements = OrbitalElements(
            semimajor_axis_km=a,
            eccentricity=e,
            inclination_deg=inc,
            longitude_ascending_node_deg=node,
            argument_periapsis_deg=argp,
            mean_anomaly_deg=mean,
        )
        primary = elements_to_state(elements)
        secondary = orbital_elements_to_icrf_state(elements, MU_EARTH_KM3_S2)
        worst_position_km = max(
            worst_position_km, float(np.max(np.abs(primary[:3] - secondary[:3])))
        )
        worst_velocity_km_s = max(
            worst_velocity_km_s, float(np.max(np.abs(primary[3:] - secondary[3:])))
        )
    assert worst_position_km < 1e-6
    assert worst_velocity_km_s < 1e-9


def test_geod_matches_vincenty_reference_for_validation_scale_pairs() -> None:
    """Canonical Geod vs the Vincenty reference: < 1 mm at validation scales."""

    offsets_deg = (0.0004, 0.01, 0.05, 0.2)
    for latitude, longitude in _REFERENCE_SITES:
        for offset in offsets_deg:
            for delta_lat, delta_lon in ((offset, 0.0), (0.0, offset), (offset, -offset)):
                expected = _vincenty_reference_km(
                    latitude, longitude, latitude + delta_lat, longitude + delta_lon
                )
                actual = wgs84_geodesic_distance_km(
                    latitude, longitude, latitude + delta_lat, longitude + delta_lon
                )
                assert abs(actual - expected) < 1e-6


def test_geod_matches_vincenty_reference_for_long_pairs() -> None:
    """Continental-scale pairs between reference sites agree to < 1 cm."""

    for (lat_1, lon_1), (lat_2, lon_2) in itertools.combinations(_REFERENCE_SITES, 2):
        expected = _vincenty_reference_km(lat_1, lon_1, lat_2, lon_2)
        actual = wgs84_geodesic_distance_km(lat_1, lon_1, lat_2, lon_2)
        assert abs(actual - expected) < 1e-5


def test_geodesic_zero_and_symmetry() -> None:
    latitude, longitude = _REFERENCE_SITES[0]
    assert wgs84_geodesic_distance_km(latitude, longitude, latitude, longitude) == 0.0
    forward = wgs84_geodesic_distance_km(latitude, longitude, -12.5, 101.25)
    backward = wgs84_geodesic_distance_km(-12.5, 101.25, latitude, longitude)
    assert abs(forward - backward) < 1e-9
