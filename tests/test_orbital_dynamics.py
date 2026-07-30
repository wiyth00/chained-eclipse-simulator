"""Checks for mass calculation, element conversion, and two-body propagation."""

import math

import numpy as np
import pytest

from chained_eclipse.constants import SECOND_MOON_MASS_KG
from chained_eclipse.models import OrbitalElements, spherical_mass_kg
from chained_eclipse.orbital_dynamics import (
    elements_to_state,
    mean_motion_rad_s,
    propagate_two_body,
    state_to_elements,
)


def test_second_moon_mass_matches_radius_density_calculation() -> None:
    assert SECOND_MOON_MASS_KG == pytest.approx(8.2430310159e21, rel=1e-10)


def test_custom_bulk_properties_recalculate_default_mass() -> None:
    elements = OrbitalElements(radius_km=1_000.0, density_kg_m3=2_500.0)

    assert elements.mass_kg == pytest.approx(spherical_mass_kg(1_000.0, 2_500.0))
    assert elements.mass_kg != SECOND_MOON_MASS_KG


def test_explicit_independently_constrained_mass_is_retained() -> None:
    elements = OrbitalElements(
        radius_km=1_000.0,
        density_kg_m3=2_500.0,
        mass_kg=9.0e21,
    )

    assert elements.mass_kg == 9.0e21


def test_explicit_nominal_mass_is_retained_with_custom_radius() -> None:
    elements = OrbitalElements(
        radius_km=2_514.0,
        density_kg_m3=3_344.0,
        mass_kg=SECOND_MOON_MASS_KG,
    )

    assert elements.mass_kg == SECOND_MOON_MASS_KG


def test_massless_control_is_retained() -> None:
    elements = OrbitalElements(mass_kg=0.0)

    assert elements.mass_kg == 0.0


@pytest.mark.parametrize(
    ("radius_km", "density_kg_m3"),
    [
        (0.0, 2_500.0),
        (-1.0, 2_500.0),
        (math.inf, 2_500.0),
        (1_000.0, 0.0),
        (1_000.0, -1.0),
        (1_000.0, math.nan),
    ],
)
def test_invalid_bulk_properties_are_rejected(
    radius_km: float, density_kg_m3: float
) -> None:
    with pytest.raises(ValueError):
        OrbitalElements(radius_km=radius_km, density_kg_m3=density_kg_m3)


def test_invalid_explicit_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        OrbitalElements(mass_kg=-1.0)


def test_element_state_round_trip() -> None:
    elements = OrbitalElements(
        longitude_ascending_node_deg=119.7,
        argument_periapsis_deg=320.5,
        mean_anomaly_deg=115.4,
    )
    recovered = state_to_elements(elements_to_state(elements), epoch_utc=elements.epoch_utc)
    assert recovered.semimajor_axis_km == pytest.approx(elements.semimajor_axis_km)
    assert recovered.eccentricity == pytest.approx(elements.eccentricity)
    assert recovered.inclination_deg == pytest.approx(elements.inclination_deg)
    assert recovered.longitude_ascending_node_deg == pytest.approx(
        elements.longitude_ascending_node_deg
    )
    assert recovered.argument_periapsis_deg == pytest.approx(
        elements.argument_periapsis_deg
    )
    assert recovered.mean_anomaly_deg == pytest.approx(elements.mean_anomaly_deg)


def test_two_body_state_repeats_after_one_orbital_period() -> None:
    elements = OrbitalElements()
    period = 2.0 * np.pi / mean_motion_rad_s(elements.semimajor_axis_km)
    initial = elements_to_state(elements)
    final = propagate_two_body(elements, period)
    assert np.linalg.norm(final[:3] - initial[:3]) < 1e-6
    assert np.linalg.norm(final[3:] - initial[3:]) < 1e-10
