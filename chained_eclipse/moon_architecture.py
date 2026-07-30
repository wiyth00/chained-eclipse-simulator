"""Hierarchical initial conditions for a bound binary-moon system.

The original simulator places both moons on independent Earth-centred
osculating orbits.  A very massive outer moon cannot be packed safely beside
the real Moon in that architecture.  This module supplies the alternative
Jacobi hierarchy:

1. the real and giant moons orbit their mutual barycenter;
2. that barycenter orbits Earth;
3. the Earth--moon hierarchy orbits the Sun in the coupled REBOUND model.

The Keplerian elements here are initial conditions, not prescribed tracks.
After the epoch every body is propagated self-consistently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .constants import (
    EARTH_MASS_KG,
    MU_EARTH_KM3_S2,
    REAL_MOON_MASS_KG,
    REAL_MOON_RADIUS_KM,
    SUN_MASS_KG,
    WGS84_A_KM,
)
from .models import OrbitalElements, spherical_mass_kg
from .orbital_dynamics import elements_to_state


AU_KM = 149_597_870.7
REAL_MOON_SEMIMAJOR_AXIS_KM = 384_400.0


@dataclass(frozen=True, slots=True)
class KeplerOrbit:
    """One relative Kepler orbit in the mean ecliptic J2000 frame."""

    semimajor_axis_km: float
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    longitude_ascending_node_deg: float = 0.0
    argument_periapsis_deg: float = 0.0
    mean_anomaly_deg: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.semimajor_axis_km) or self.semimajor_axis_km <= 0.0:
            raise ValueError("semimajor_axis_km must be finite and positive")
        if not math.isfinite(self.eccentricity) or not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must satisfy 0 <= e < 1")
        for name in (
            "inclination_deg",
            "longitude_ascending_node_deg",
            "argument_periapsis_deg",
            "mean_anomaly_deg",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> KeplerOrbit:
        return cls(
            semimajor_axis_km=float(payload["semimajor_axis_km"]),
            eccentricity=float(payload.get("eccentricity", 0.0)),
            inclination_deg=float(payload.get("inclination_deg", 0.0)),
            longitude_ascending_node_deg=float(payload.get("longitude_ascending_node_deg", 0.0)),
            argument_periapsis_deg=float(payload.get("argument_periapsis_deg", 0.0)),
            mean_anomaly_deg=float(payload.get("mean_anomaly_deg", 0.0)),
        )

    def to_elements(self, epoch_utc: str) -> OrbitalElements:
        return OrbitalElements(epoch_utc=epoch_utc, **asdict(self))


@dataclass(frozen=True, slots=True)
class BinaryMoonArchitecture:
    """Jacobi elements for a moon pair orbiting Earth."""

    epoch_utc: str
    outer_orbit: KeplerOrbit
    mutual_orbit: KeplerOrbit
    name: str = "hierarchical-binary-moons"

    def __post_init__(self) -> None:
        if not self.epoch_utc.strip():
            raise ValueError("epoch_utc must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def architecture_from_config(
    payload: Mapping[str, Any],
) -> BinaryMoonArchitecture | None:
    """Load an optional ``binary_moons`` hierarchy from a YAML payload."""

    section = payload.get("binary_moons")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("binary_moons must be a mapping")
    model = payload.get("model", {})
    epoch_utc = str(
        section.get(
            "epoch_utc",
            model.get("epoch_utc", payload.get("orbital_elements", {}).get("epoch_utc", "")),
        )
    )
    return BinaryMoonArchitecture(
        epoch_utc=epoch_utc,
        outer_orbit=KeplerOrbit.from_mapping(section["outer_orbit"]),
        mutual_orbit=KeplerOrbit.from_mapping(section["mutual_orbit"]),
        name=str(section.get("name", "hierarchical-binary-moons")),
    )


def elements_from_config(payload: Mapping[str, Any]) -> OrbitalElements:
    """Load the second moon's physical properties from either config style."""

    if "orbital_elements" in payload:
        return OrbitalElements(**payload["orbital_elements"])
    moon = payload["second_moon"]
    model = payload["model"]
    radius_km = float(moon["radius_km"])
    density_kg_m3 = float(moon["density_kg_m3"])
    mass = moon.get("mass_kg")
    return OrbitalElements(
        semimajor_axis_km=float(moon["semimajor_axis_km"]),
        eccentricity=float(moon.get("eccentricity", 0.0)),
        inclination_deg=float(moon.get("inclination_deg", 0.0)),
        longitude_ascending_node_deg=float(moon.get("longitude_ascending_node_deg") or 0.0),
        argument_periapsis_deg=float(moon.get("argument_periapsis_deg") or 0.0),
        mean_anomaly_deg=float(moon.get("mean_anomaly_deg") or 0.0),
        epoch_utc=str(model["epoch_utc"]),
        radius_km=radius_km,
        density_kg_m3=density_kg_m3,
        mass_kg=(spherical_mass_kg(radius_km, density_kg_m3) if mass is None else float(mass)),
    )


def binary_moon_states_icrf(
    architecture: BinaryMoonArchitecture,
    *,
    gravitational_constant_km3_kg_s2: float,
    earth_mass_kg: float,
    real_moon_mass_kg: float,
    second_moon_mass_kg: float,
) -> dict[str, np.ndarray]:
    """Return Earth-relative ICRF states for a hierarchical moon pair.

    The outer state is the moon-pair barycenter relative to Earth.  The mutual
    state is the second moon relative to the real Moon.  Mass-weighting those
    two Jacobi vectors yields Cartesian states with the requested barycenter.
    """

    masses = (
        gravitational_constant_km3_kg_s2,
        earth_mass_kg,
        real_moon_mass_kg,
        second_moon_mass_kg,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in masses):
        raise ValueError("binary-moon state generation requires finite positive masses and G")
    if architecture.epoch_utc != architecture.epoch_utc.strip():
        raise ValueError("architecture epoch_utc must not contain surrounding whitespace")

    moon_pair_mass = real_moon_mass_kg + second_moon_mass_kg
    outer_mu = gravitational_constant_km3_kg_s2 * (earth_mass_kg + moon_pair_mass)
    mutual_mu = gravitational_constant_km3_kg_s2 * moon_pair_mass
    outer_state = elements_to_state(
        architecture.outer_orbit.to_elements(architecture.epoch_utc),
        outer_mu,
    )
    mutual_state = elements_to_state(
        architecture.mutual_orbit.to_elements(architecture.epoch_utc),
        mutual_mu,
    )
    real_fraction = second_moon_mass_kg / moon_pair_mass
    second_fraction = real_moon_mass_kg / moon_pair_mass
    return {
        "real_moon": outer_state - real_fraction * mutual_state,
        "second_moon": outer_state + second_fraction * mutual_state,
        "moon_pair_barycenter": outer_state,
        "second_relative_to_real": mutual_state,
    }


def _mutual_inclination_rad(first: KeplerOrbit, second: KeplerOrbit) -> float:
    first_i = math.radians(first.inclination_deg)
    second_i = math.radians(second.inclination_deg)
    node_delta = math.radians(
        first.longitude_ascending_node_deg - second.longitude_ascending_node_deg
    )
    cosine = math.cos(first_i) * math.cos(second_i) + math.sin(first_i) * math.sin(
        second_i
    ) * math.cos(node_delta)
    return math.acos(max(-1.0, min(1.0, cosine)))


def architecture_diagnostics(
    architecture: BinaryMoonArchitecture,
    second_moon: OrbitalElements,
    *,
    earth_mass_kg: float = EARTH_MASS_KG,
    real_moon_mass_kg: float = REAL_MOON_MASS_KG,
    real_moon_radius_km: float = REAL_MOON_RADIUS_KM,
) -> dict[str, Any]:
    """Return analytic hierarchy, Hill, collision, and viewing diagnostics."""

    second_mass = float(second_moon.mass_kg)
    pair_mass = real_moon_mass_kg + second_mass
    project_g = MU_EARTH_KM3_S2 / EARTH_MASS_KG
    outer = architecture.outer_orbit
    mutual = architecture.mutual_orbit
    outer_periapsis = outer.semimajor_axis_km * (1.0 - outer.eccentricity)
    outer_apoapsis = outer.semimajor_axis_km * (1.0 + outer.eccentricity)
    mutual_periapsis = mutual.semimajor_axis_km * (1.0 - mutual.eccentricity)
    mutual_apoapsis = mutual.semimajor_axis_km * (1.0 + mutual.eccentricity)
    binary_hill_at_periapsis = outer_periapsis * (pair_mass / (3.0 * earth_mass_kg)) ** (1.0 / 3.0)
    mutual_inclination = _mutual_inclination_rad(mutual, outer)

    # Mardling--Aarseth's empirical hierarchy screen, expressed as an outer
    # pericenter-to-inner-semimajor-axis ratio.
    outer_mass_ratio = earth_mass_kg / pair_mass
    hierarchy_threshold = (
        2.8
        * (
            (1.0 + outer_mass_ratio)
            * (1.0 + outer.eccentricity)
            / math.sqrt(1.0 - outer.eccentricity)
        )
        ** 0.4
        * (1.0 - 0.3 * mutual_inclination / math.pi)
    )
    hierarchy_ratio = outer_periapsis / mutual.semimajor_axis_km

    earth_system_mass = earth_mass_kg + pair_mass
    earth_hill_radius = AU_KM * (earth_system_mass / (3.0 * SUN_MASS_KG)) ** (1.0 / 3.0)
    mutual_hill_mass_factor = (pair_mass / (3.0 * earth_mass_kg)) ** (1.0 / 3.0)
    circular_hill_factor = math.sqrt(3.0) * mutual_hill_mass_factor
    independent_inner_limit = REAL_MOON_SEMIMAJOR_AXIS_KM * (
        (1.0 - circular_hill_factor) / (1.0 + circular_hill_factor)
    )
    independent_outer_limit = REAL_MOON_SEMIMAJOR_AXIS_KM * (
        (1.0 + circular_hill_factor) / (1.0 - circular_hill_factor)
    )
    prograde_solar_limit = 0.4895 * earth_hill_radius
    real_offset_fraction = second_mass / pair_mass
    second_offset_fraction = real_moon_mass_kg / pair_mass
    minimum_real_distance = outer_periapsis - real_offset_fraction * mutual_apoapsis
    minimum_second_distance = outer_periapsis - second_offset_fraction * mutual_apoapsis
    maximum_real_distance = outer_apoapsis + real_offset_fraction * mutual_apoapsis
    maximum_second_distance = outer_apoapsis + second_offset_fraction * mutual_apoapsis
    collision_radius_sum = real_moon_radius_km + second_moon.radius_km

    outer_period_days = (
        2.0
        * math.pi
        * math.sqrt(outer.semimajor_axis_km**3 / (project_g * (earth_mass_kg + pair_mass)))
        / 86_400.0
    )
    mutual_period_days = (
        2.0 * math.pi * math.sqrt(mutual.semimajor_axis_km**3 / (project_g * pair_mass)) / 86_400.0
    )
    giant_angular_min = 2.0 * math.degrees(
        math.asin(second_moon.radius_km / maximum_second_distance)
    )
    giant_angular_max = 2.0 * math.degrees(
        math.asin(second_moon.radius_km / minimum_second_distance)
    )
    real_angular_min = 2.0 * math.degrees(math.asin(real_moon_radius_km / maximum_real_distance))
    real_angular_max = 2.0 * math.degrees(math.asin(real_moon_radius_km / minimum_real_distance))

    return {
        "architecture": architecture.to_dict(),
        "mass": {
            "real_moon_mass_kg": real_moon_mass_kg,
            "second_moon_mass_kg": second_mass,
            "pair_mass_kg": pair_mass,
            "second_moon_mass_in_real_moons": second_mass / real_moon_mass_kg,
        },
        "periods": {
            "outer_barycenter_period_days": outer_period_days,
            "mutual_orbit_period_days": mutual_period_days,
        },
        "hierarchy": {
            "outer_periapsis_to_mutual_semimajor_axis": hierarchy_ratio,
            "mardling_aarseth_threshold": hierarchy_threshold,
            "mardling_aarseth_screen_pass": hierarchy_ratio > hierarchy_threshold,
            "binary_hill_radius_at_outer_periapsis_km": binary_hill_at_periapsis,
            "mutual_apoapsis_fraction_of_binary_hill": (mutual_apoapsis / binary_hill_at_periapsis),
            "mutual_orbit_inside_half_binary_hill": (
                mutual_apoapsis < 0.5 * binary_hill_at_periapsis
            ),
            "mutual_inclination_deg": math.degrees(mutual_inclination),
        },
        "earth_orbit": {
            "earth_system_hill_radius_km": earth_hill_radius,
            "outer_semimajor_axis_fraction_of_earth_hill": (
                outer.semimajor_axis_km / earth_hill_radius
            ),
            "outer_semimajor_axis_inside_half_earth_hill": (
                outer.semimajor_axis_km < 0.5 * earth_hill_radius
            ),
            "minimum_real_moon_distance_km": minimum_real_distance,
            "maximum_real_moon_distance_km": maximum_real_distance,
            "minimum_second_moon_distance_km": minimum_second_distance,
            "maximum_second_moon_distance_km": maximum_second_distance,
            "minimum_earth_surface_clearance_km": (
                min(minimum_real_distance, minimum_second_distance)
                - WGS84_A_KM
                - max(real_moon_radius_km, second_moon.radius_km)
            ),
        },
        "independent_orbit_screen": {
            "criterion": ("circular coplanar spacing greater than 2 sqrt(3) mutual Hill radii"),
            "real_moon_semimajor_axis_km": REAL_MOON_SEMIMAJOR_AXIS_KM,
            "second_moon_inner_axis_must_be_below_km": independent_inner_limit,
            "second_moon_outer_axis_must_be_above_km": independent_outer_limit,
            "approximate_prograde_solar_stability_limit_km": prograde_solar_limit,
            "outer_solution_inside_prograde_solar_limit": (
                independent_outer_limit < prograde_solar_limit
            ),
            "interpretation": (
                "A distinct outer prograde orbit cannot satisfy both screens; a distinct "
                "inner orbit sacrifices the distant, solar-sized appearance."
            ),
        },
        "collision": {
            "physical_radius_sum_km": collision_radius_sum,
            "mutual_periapsis_km": mutual_periapsis,
            "minimum_surface_gap_km": mutual_periapsis - collision_radius_sum,
            "mutual_periapsis_outside_contact": mutual_periapsis > collision_radius_sum,
        },
        "apparent_diameters_deg": {
            "real_moon_min": real_angular_min,
            "real_moon_max": real_angular_max,
            "second_moon_min": giant_angular_min,
            "second_moon_max": giant_angular_max,
        },
        "interpretation": (
            "These analytic screens reject obvious failures; the coupled N-body "
            "integration remains the authoritative boundedness test."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Print the analytic screens for a binary-moon YAML configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/bound_binary_giant.yaml")
    args = parser.parse_args(argv)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    architecture = architecture_from_config(payload)
    if architecture is None:
        parser.error("the selected configuration has no binary_moons section")
    print(
        json.dumps(
            architecture_diagnostics(architecture, elements_from_config(payload)),
            indent=2,
        )
    )
    return 0


__all__ = [
    "BinaryMoonArchitecture",
    "KeplerOrbit",
    "architecture_diagnostics",
    "architecture_from_config",
    "binary_moon_states_icrf",
    "elements_from_config",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
