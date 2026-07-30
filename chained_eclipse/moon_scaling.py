"""Scale the hypothetical moon while preserving its geocentric angular size.

If orbital distance and physical radius are multiplied by the same factor,
the body's angular diameter is unchanged at corresponding orbital phases.
This module makes that similarity transformation explicit and reports the
dynamical quantities that do *not* remain similar.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import yaml

from .constants import (
    EARTH_MASS_KG,
    MU_EARTH_KM3_S2,
    REAL_MOON_MASS_KG,
    SUN_MASS_KG,
    WGS84_A_KM,
)
from .models import OrbitalElements, spherical_mass_kg

MassMode = Literal["constant-density", "fixed-mass"]
AU_KM = 149_597_870.7
REAL_MOON_SEMIMAJOR_AXIS_KM = 384_400.0


def scale_moon(
    elements: OrbitalElements,
    scale_factor: float,
    *,
    mass_mode: MassMode = "constant-density",
) -> OrbitalElements:
    """Return a linearly scaled orbit-and-radius counterfactual.

    ``constant-density`` is the physically direct transformation: mass grows
    as the cube of the scale factor. ``fixed-mass`` is an optics-only control
    whose implied bulk density falls as the inverse cube.
    """

    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("scale_factor must be finite and positive")
    if mass_mode not in {"constant-density", "fixed-mass"}:
        raise ValueError(f"unsupported mass mode: {mass_mode}")
    radius_km = elements.radius_km * scale_factor
    mass_kg = (
        spherical_mass_kg(radius_km, elements.density_kg_m3)
        if mass_mode == "constant-density"
        else elements.mass_kg
    )
    return replace(
        elements,
        semimajor_axis_km=elements.semimajor_axis_km * scale_factor,
        radius_km=radius_km,
        mass_kg=mass_kg,
    )


def _bulk_density_kg_m3(elements: OrbitalElements) -> float:
    volume_m3 = 4.0 / 3.0 * math.pi * (elements.radius_km * 1_000.0) ** 3
    return elements.mass_kg / volume_m3


def _orbital_period_days(
    elements: OrbitalElements,
    *,
    include_second_moon_mass: bool,
) -> float:
    project_g = MU_EARTH_KM3_S2 / EARTH_MASS_KG
    gravitational_parameter = MU_EARTH_KM3_S2
    if include_second_moon_mass:
        gravitational_parameter += project_g * elements.mass_kg
    period_seconds = (
        2.0 * math.pi * math.sqrt(elements.semimajor_axis_km**3 / gravitational_parameter)
    )
    return period_seconds / 86_400.0


def _mutual_hill_spacing(elements: OrbitalElements) -> float:
    mean_axis = 0.5 * (REAL_MOON_SEMIMAJOR_AXIS_KM + elements.semimajor_axis_km)
    mutual_hill_radius = mean_axis * (
        (REAL_MOON_MASS_KG + elements.mass_kg) / (3.0 * EARTH_MASS_KG)
    ) ** (1.0 / 3.0)
    return abs(elements.semimajor_axis_km - REAL_MOON_SEMIMAJOR_AXIS_KM) / (mutual_hill_radius)


def scaling_diagnostics(
    reference: OrbitalElements,
    scaled: OrbitalElements,
) -> dict[str, object]:
    """Return compact optical, tidal, and stability diagnostics."""

    reference_perigee = reference.semimajor_axis_km * (1.0 - reference.eccentricity)
    scaled_perigee = scaled.semimajor_axis_km * (1.0 - scaled.eccentricity)
    angular_diameter_deg = 2.0 * math.degrees(math.asin(scaled.radius_km / scaled_perigee))
    reference_angular_diameter_deg = 2.0 * math.degrees(
        math.asin(reference.radius_km / reference_perigee)
    )
    tide_ratio = (scaled.mass_kg / scaled_perigee**3) / (reference.mass_kg / reference_perigee**3)
    barycenter_km = scaled.semimajor_axis_km * scaled.mass_kg / (EARTH_MASS_KG + scaled.mass_kg)
    earth_system_mass = EARTH_MASS_KG + REAL_MOON_MASS_KG + scaled.mass_kg
    hill_radius_km = AU_KM * (earth_system_mass / (3.0 * SUN_MASS_KG)) ** (1.0 / 3.0)
    prograde_rule_of_thumb_km = 0.5 * hill_radius_km
    mutual_spacing = _mutual_hill_spacing(scaled)
    return {
        "linear_scale_factor": scaled.semimajor_axis_km / reference.semimajor_axis_km,
        "reference": {
            "semimajor_axis_km": reference.semimajor_axis_km,
            "radius_km": reference.radius_km,
            "mass_kg": reference.mass_kg,
            "perigee_km": reference_perigee,
            "geocentric_perigee_angular_diameter_deg": (reference_angular_diameter_deg),
            "restricted_test_particle_period_days": _orbital_period_days(
                reference,
                include_second_moon_mass=False,
            ),
            "massive_two_body_period_days": _orbital_period_days(
                reference,
                include_second_moon_mass=True,
            ),
        },
        "scaled": {
            "semimajor_axis_km": scaled.semimajor_axis_km,
            "radius_km": scaled.radius_km,
            "mass_kg": scaled.mass_kg,
            "mass_in_real_moons": scaled.mass_kg / REAL_MOON_MASS_KG,
            "bulk_density_kg_m3": _bulk_density_kg_m3(scaled),
            "perigee_km": scaled_perigee,
            "geocentric_perigee_angular_diameter_deg": angular_diameter_deg,
            "restricted_test_particle_period_days": _orbital_period_days(
                scaled,
                include_second_moon_mass=False,
            ),
            "massive_two_body_period_days": _orbital_period_days(
                scaled,
                include_second_moon_mass=True,
            ),
            "earth_barycenter_distance_km": barycenter_km,
        },
        "similarities": {
            "angular_diameter_ratio": angular_diameter_deg / reference_angular_diameter_deg,
            "perigee_tide_strength_ratio": tide_ratio,
        },
        "dynamical_screen": {
            "earth_system_hill_radius_km": hill_radius_km,
            "prograde_half_hill_rule_of_thumb_km": prograde_rule_of_thumb_km,
            "semimajor_axis_inside_half_hill_radius": (
                scaled.semimajor_axis_km < prograde_rule_of_thumb_km
            ),
            "real_to_second_moon_mutual_hill_spacing": mutual_spacing,
            "widely_separated_heuristic_pass": mutual_spacing > 3.5,
            "earth_barycenter_outside_earth": barycenter_km > WGS84_A_KM,
            "interpretation": (
                "The Hill checks are screening heuristics, not a stability proof; "
                "the coupled REBOUND integration is authoritative for this project."
            ),
        },
    }


def load_elements(path: str | Path) -> OrbitalElements:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "orbital_elements" in payload:
        return OrbitalElements(**payload["orbital_elements"])
    moon = payload["second_moon"]
    radius_km = float(moon["radius_km"])
    density_kg_m3 = float(moon["density_kg_m3"])
    return OrbitalElements(
        semimajor_axis_km=float(moon["semimajor_axis_km"]),
        eccentricity=float(moon["eccentricity"]),
        inclination_deg=float(moon["inclination_deg"]),
        epoch_utc=str(payload["model"]["epoch_utc"]),
        radius_km=radius_km,
        density_kg_m3=density_kg_m3,
        mass_kg=float(moon.get("mass_kg", spherical_mass_kg(radius_km, density_kg_m3))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument(
        "--mass-mode",
        choices=("constant-density", "fixed-mass"),
        default="constant-density",
    )
    args = parser.parse_args(argv)
    reference = load_elements(args.config)
    scaled = scale_moon(reference, args.scale_factor, mass_mode=args.mass_mode)
    print(json.dumps(scaling_diagnostics(reference, scaled), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
