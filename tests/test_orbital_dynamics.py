"""Checks for mass calculation, element conversion, and two-body propagation."""

import numpy as np
import pytest

from chained_eclipse.constants import SECOND_MOON_MASS_KG
from chained_eclipse.models import OrbitalElements
from chained_eclipse.orbital_dynamics import (
    elements_to_state,
    mean_motion_rad_s,
    propagate_two_body,
    state_to_elements,
)


def test_second_moon_mass_matches_radius_density_calculation() -> None:
    assert SECOND_MOON_MASS_KG == pytest.approx(8.2430310159e21, rel=1e-10)


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

