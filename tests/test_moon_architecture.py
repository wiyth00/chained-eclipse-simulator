"""Checks for the hierarchical binary-moon initial conditions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from chained_eclipse.constants import (
    EARTH_MASS_KG,
    MU_EARTH_KM3_S2,
    REAL_MOON_MASS_KG,
)
from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.moon_architecture import (
    architecture_diagnostics,
    architecture_from_config,
    binary_moon_states_icrf,
    elements_from_config,
)
from chained_eclipse.stability import StabilityConfig, run_stability_check


ROOT = Path(__file__).resolve().parents[1]


def _configuration():
    payload = yaml.safe_load(
        (ROOT / "config" / "bound_binary_giant.yaml").read_text(encoding="utf-8")
    )
    architecture = architecture_from_config(payload)
    assert architecture is not None
    return elements_from_config(payload), architecture


def test_binary_config_passes_analytic_hierarchy_screens() -> None:
    elements, architecture = _configuration()
    diagnostics = architecture_diagnostics(architecture, elements)

    assert diagnostics["periods"]["outer_barycenter_period_days"] == pytest.approx(39.7510472458)
    assert diagnostics["periods"]["mutual_orbit_period_days"] == pytest.approx(4.13924162047)
    assert diagnostics["hierarchy"]["mardling_aarseth_screen_pass"]
    assert diagnostics["hierarchy"]["mutual_orbit_inside_half_binary_hill"]
    assert diagnostics["earth_orbit"]["outer_semimajor_axis_inside_half_earth_hill"]
    assert diagnostics["independent_orbit_screen"][
        "second_moon_inner_axis_must_be_below_km"
    ] == pytest.approx(149_071.047)
    assert diagnostics["independent_orbit_screen"][
        "second_moon_outer_axis_must_be_above_km"
    ] == pytest.approx(991_227.762)
    assert not diagnostics["independent_orbit_screen"]["outer_solution_inside_prograde_solar_limit"]
    assert diagnostics["collision"]["minimum_surface_gap_km"] > 35_000.0
    assert 0.55 < diagnostics["apparent_diameters_deg"]["second_moon_min"] < 0.60


def test_binary_states_preserve_barycenter_and_mutual_vectors() -> None:
    elements, architecture = _configuration()
    project_g = MU_EARTH_KM3_S2 / EARTH_MASS_KG
    states = binary_moon_states_icrf(
        architecture,
        gravitational_constant_km3_kg_s2=project_g,
        earth_mass_kg=EARTH_MASS_KG,
        real_moon_mass_kg=REAL_MOON_MASS_KG,
        second_moon_mass_kg=float(elements.mass_kg),
    )
    pair_mass = REAL_MOON_MASS_KG + float(elements.mass_kg)
    recovered_barycenter = (
        REAL_MOON_MASS_KG * states["real_moon"] + float(elements.mass_kg) * states["second_moon"]
    ) / pair_mass

    assert recovered_barycenter == pytest.approx(states["moon_pair_barycenter"])
    assert states["second_moon"] - states["real_moon"] == pytest.approx(
        states["second_relative_to_real"]
    )
    assert np.linalg.norm(states["second_relative_to_real"][:3]) == pytest.approx(
        39_654.603, rel=1.0e-6
    )


def test_binary_architecture_remains_bound_for_one_year() -> None:
    elements, architecture = _configuration()
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    result = run_stability_check(
        context,
        elements,
        config=StabilityConfig(
            years=1.0,
            sample_interval_days=10.0,
            ias15_epsilon=1.0e-10,
        ),
        binary_architecture=architecture,
    )

    assert result.stable
    assert result.completed
    assert result.architecture == "hierarchical binary moons"
    assert result.binary_moons is not None
    assert not result.ejection_detected
    assert not result.collision_detected
    assert result.min_moon_moon_distance_km > 35_000.0
    assert max(result.binary_moons["mutual_orbit"]["eccentricity"]) < 0.15
