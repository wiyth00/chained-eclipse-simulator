"""Typed data models shared by the search, geometry, and reporting layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .constants import (
    SECOND_MOON_DENSITY_KG_M3,
    SECOND_MOON_MASS_KG,
    SECOND_MOON_RADIUS_KM,
)


def spherical_mass_kg(radius_km: float, density_kg_m3: float) -> float:
    """Return the mass of a spherical body from its radius and bulk density."""

    if not math.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("radius_km must be a finite positive value")
    if not math.isfinite(density_kg_m3) or density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be a finite positive value")
    return 4.0 / 3.0 * math.pi * (radius_km * 1_000.0) ** 3 * density_kg_m3


@dataclass(slots=True)
class OrbitalElements:
    """Osculating Earth-centred elements at the configured epoch.

    Angular fields are degrees in the mean J2000 ecliptic frame. When callers
    customize the radius or density but leave ``mass_kg`` at the package
    default, the mass is recalculated to keep the physical parameters
    internally consistent. An explicitly supplied non-default mass is retained
    for callers modelling a body whose mass is independently constrained.
    """

    semimajor_axis_km: float = 180_000.0
    eccentricity: float = 0.04
    inclination_deg: float = 5.0
    longitude_ascending_node_deg: float = 0.0
    argument_periapsis_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    epoch_utc: str = "2026-07-10T00:00:00Z"
    radius_km: float = SECOND_MOON_RADIUS_KM
    density_kg_m3: float = SECOND_MOON_DENSITY_KG_M3
    mass_kg: float = SECOND_MOON_MASS_KG
    frame: str = "mean_ecliptic_J2000"

    def __post_init__(self) -> None:
        derived_mass = spherical_mass_kg(self.radius_km, self.density_kg_m3)
        uses_custom_bulk_properties = (
            self.radius_km != SECOND_MOON_RADIUS_KM
            or self.density_kg_m3 != SECOND_MOON_DENSITY_KG_M3
        )
        if uses_custom_bulk_properties and self.mass_kg == SECOND_MOON_MASS_KG:
            self.mass_kg = derived_mass
        elif not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be a finite positive value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ShadowGeometry:
    time_utc: str
    axis_distance_km: float
    axis_parameter_km: float
    penumbra_radius_km: float
    core_radius_km: float
    umbra_apex_distance_km: float
    central: bool
    partial_intersection: bool
    core_intersection: bool
    core_kind: str | None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    solar_altitude_deg: float | None = None


@dataclass(slots=True)
class LocalCircumstances:
    body: str
    eclipse_type: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    c1_utc: str | None
    c2_utc: str | None
    maximum_utc: str
    c3_utc: str | None
    c4_utc: str | None
    magnitude: float
    obscuration: float
    central_duration_s: float
    solar_altitude_deg: float
    center_separation_deg: float
    solar_angular_diameter_deg: float
    moon_angular_diameter_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RealEclipse:
    event_id: str
    maximum_utc: str
    eclipse_type: str
    latitude_deg: float | None
    longitude_deg: float | None
    axis_distance_km: float
    penumbra_margin_km: float
    core_margin_km: float
    solar_altitude_deg: float | None
    source: str = "DE440s numerical conjunction/shadow verification"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChainedEvent:
    rank: int
    event_id: str
    definition_a: bool
    definition_b: bool
    real_eclipse: LocalCircumstances
    second_eclipse: LocalCircumstances
    midpoint_separation_s: float
    track_distance_km: float
    best_latitude_deg: float
    best_longitude_deg: float
    both_total: bool
    two_totality_components: bool
    external_contact_intervals_disjoint: bool
    thresholds: dict[str, bool] = field(default_factory=dict)
    numerical_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Trajectory:
    """Callable Earth-centred trajectory in ICRF coordinates."""

    epoch_jd: float
    start_jd: float
    end_jd: float
    evaluator: Callable[[np.ndarray | float], np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def state(self, jd: np.ndarray | float) -> np.ndarray:
        result = np.asarray(self.evaluator(jd), dtype=float)
        return result

    def position(self, jd: np.ndarray | float) -> np.ndarray:
        state = self.state(jd)
        if state.ndim == 1:
            return state[:3]
        return state[..., :3]
