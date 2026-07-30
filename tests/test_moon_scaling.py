"""Fast checks for the distant-giant similarity transformation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from chained_eclipse.cli import (
    DEFAULT_CONFIG,
    _baseline_elements,
    _updates_project_optimized_config,
)
from chained_eclipse.constants import (
    REAL_MOON_RADIUS_KM,
    SECOND_MOON_MASS_KG,
)
from chained_eclipse.coupled_eclipse import body_radius_km
from chained_eclipse.models import OrbitalElements
from chained_eclipse.moon_scaling import scale_moon, scaling_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def test_three_x_constant_density_preserves_angular_size_and_tide_scale() -> None:
    reference = OrbitalElements()
    giant = scale_moon(reference, 3.0)
    diagnostics = scaling_diagnostics(reference, giant)

    assert giant.semimajor_axis_km == 540_000.0
    assert giant.radius_km == 2_514.0
    assert giant.mass_kg == pytest.approx(27.0 * reference.mass_kg)
    assert diagnostics["similarities"]["angular_diameter_ratio"] == pytest.approx(1.0)
    assert diagnostics["similarities"]["perigee_tide_strength_ratio"] == pytest.approx(1.0)
    assert diagnostics["scaled"]["mass_in_real_moons"] == pytest.approx(3.03135, rel=1e-5)
    assert diagnostics["dynamical_screen"]["earth_barycenter_outside_earth"] is True
    assert diagnostics["dynamical_screen"]["widely_separated_heuristic_pass"] is False


def test_fixed_mass_mode_is_an_optics_only_control() -> None:
    reference = OrbitalElements()
    giant = scale_moon(reference, 3.0, mass_mode="fixed-mass")
    diagnostics = scaling_diagnostics(reference, giant)

    assert giant.mass_kg == SECOND_MOON_MASS_KG
    assert diagnostics["similarities"]["angular_diameter_ratio"] == pytest.approx(1.0)
    assert diagnostics["similarities"]["perigee_tide_strength_ratio"] == pytest.approx(1.0 / 27.0)


def test_distant_giant_config_loads_its_explicit_physics() -> None:
    payload = yaml.safe_load((ROOT / "config" / "distant_giant.yaml").read_text())
    elements = _baseline_elements(payload)

    assert elements.semimajor_axis_km == 540_000.0
    assert elements.radius_km == 2_514.0
    assert elements.mass_kg == pytest.approx(2.225618374301551e23)


def test_saved_distant_giant_orientation_matches_the_solved_design() -> None:
    payload = yaml.safe_load((ROOT / "config" / "distant_giant_optimized.yaml").read_text())
    elements = OrbitalElements(**payload["orbital_elements"])

    assert elements.radius_km == 2_514.0
    assert elements.longitude_ascending_node_deg == pytest.approx(136.9766039166)
    assert elements.argument_periapsis_deg == pytest.approx(303.3911929566)
    assert elements.mean_anomaly_deg == pytest.approx(169.7689781825)


def test_coupled_geometry_uses_configured_second_moon_radius() -> None:
    ephemeris = SimpleNamespace(elements=OrbitalElements(radius_km=2_514.0))

    assert body_radius_km(ephemeris, "real_moon") == REAL_MOON_RADIUS_KM
    assert body_radius_km(ephemeris, "second_moon") == 2_514.0


def test_custom_scenario_does_not_replace_canonical_optimized_config() -> None:
    assert _updates_project_optimized_config(DEFAULT_CONFIG)
    assert not _updates_project_optimized_config(ROOT / "config" / "distant_giant.yaml")
