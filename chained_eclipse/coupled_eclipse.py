"""Eclipse detection inside the self-consistent coupled four-body model.

DE440s is used only at the saved 2026-07-10 epoch.  From that instant onward
the Sun, Earth, real Moon, and second moon are propagated together by REBOUND
IAS15.  Eclipse geometry is then evaluated from the coupled Cartesian states
on a rotating WGS84 Earth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize, minimize_scalar
from skyfield.framelib import itrs
import yaml

from .constants import (
    REAL_MOON_RADIUS_KM,
    SECOND_MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SPEED_OF_LIGHT_KM_S,
    SUN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_E2,
)
from .eclipse_geometry import (
    ApparentDiskGeometry,
    GenericShadowState,
    _closest_axis_surface_guess,
    angular_separation,
    disk_overlap_fraction,
    minimum_track_distance_km,
)
from .ephemeris import (
    EphemerisContext,
    axis_ellipsoid_intersections,
    itrs_to_geodetic,
    load_ephemeris,
    time_iso_utc,
)
from .models import OrbitalElements
from .moon_architecture import (
    BinaryMoonArchitecture,
    architecture_from_config,
    elements_from_config,
)
from .stability import build_coupled_simulation


BodyName = Literal["real_moon", "second_moon"]
BODY_LABEL = {"real_moon": "real Moon", "second_moon": "second moon"}
PARTICLE_NAMES = ("sun", "earth", "real_moon", "second_moon")


def body_radius_km(ephemeris: CoupledEphemeris, body: BodyName) -> float:
    """Return the configured physical radius for an eclipse-causing body."""

    if body == "real_moon":
        return REAL_MOON_RADIUS_KM
    elements = getattr(ephemeris, "elements", None)
    return float(getattr(elements, "radius_km", SECOND_MOON_RADIUS_KM))


@dataclass(slots=True)
class CoupledEvent:
    body: str
    eclipse_type: str
    global_start_utc: str
    axis_maximum_utc: str
    global_end_utc: str
    latitude_deg: float
    longitude_deg: float
    c1_utc: str | None
    c2_utc: str | None
    local_maximum_utc: str
    c3_utc: str | None
    c4_utc: str | None
    magnitude: float
    obscuration: float
    central_duration_s: float
    solar_altitude_deg: float
    solar_angular_diameter_deg: float
    moon_angular_diameter_deg: float
    core_radius_km: float
    penumbra_margin_km: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CoupledLocalCircumstances:
    eclipse_type: str
    c1_utc: str | None
    c2_utc: str | None
    maximum_utc: str
    c3_utc: str | None
    c4_utc: str | None
    magnitude: float
    obscuration: float
    central_duration_s: float
    solar_altitude_deg: float
    solar_angular_diameter_deg: float
    moon_angular_diameter_deg: float


class CoupledEphemeris:
    """Cubic interpolation of a sampled REBOUND four-body trajectory."""

    def __init__(
        self,
        context: EphemerisContext,
        elements: OrbitalElements,
        end_utc: str,
        *,
        sample_step_seconds: float = 600.0,
        ias15_epsilon: float = 1.0e-10,
        binary_architecture: BinaryMoonArchitecture | None = None,
    ) -> None:
        if sample_step_seconds <= 0.0:
            raise ValueError("sample_step_seconds must be positive")
        self.context = context
        self.elements = elements
        self.binary_architecture = binary_architecture
        self.epoch_tt_jd = float(context.time_utc(elements.epoch_utc).tt)
        self.end_tt_jd = float(context.time_utc(end_utc).tt)
        duration_seconds = (self.end_tt_jd - self.epoch_tt_jd) * SECONDS_PER_DAY
        self.seconds = np.arange(
            0.0,
            duration_seconds + sample_step_seconds,
            sample_step_seconds,
        )
        simulation, metadata = build_coupled_simulation(
            context,
            elements,
            binary_architecture=binary_architecture,
            ias15_epsilon=ias15_epsilon,
        )
        initial_energy = float(simulation.energy())
        positions = np.empty((len(self.seconds), len(PARTICLE_NAMES), 3), dtype=float)
        for index, seconds in enumerate(self.seconds):
            simulation.integrate(float(seconds), exact_finish_time=1)
            for body_index, name in enumerate(PARTICLE_NAMES):
                particle = simulation.particles[name]
                positions[index, body_index] = (particle.x, particle.y, particle.z)
        final_energy = float(simulation.energy())
        self._splines = {
            name: CubicSpline(self.seconds, positions[:, index, :], axis=0)
            for index, name in enumerate(PARTICLE_NAMES)
        }
        self.metadata = {
            **metadata,
            "integrator": "REBOUND IAS15",
            "sample_step_seconds": sample_step_seconds,
            "sample_count": len(self.seconds),
            "start_utc": elements.epoch_utc,
            "end_utc": end_utc,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "relative_energy_error": (final_energy - initial_energy) / abs(initial_energy),
            "force_model": "Newtonian point-mass Sun/Earth/real Moon/second moon",
            "omissions": [
                "Earth J2",
                "tides",
                "relativity",
                "major planets",
                "lunar figure terms",
            ],
        }

    def _seconds(self, tt_jd: float | np.ndarray) -> np.ndarray:
        return (np.asarray(tt_jd, dtype=float) - self.epoch_tt_jd) * SECONDS_PER_DAY

    def position(self, name: str, tt_jd: float | np.ndarray) -> np.ndarray:
        return np.asarray(self._splines[name](self._seconds(tt_jd)), dtype=float)

    def velocity(self, name: str, tt_jd: float | np.ndarray) -> np.ndarray:
        """Return interpolated inertial velocity in kilometres per second."""

        return np.asarray(
            self._splines[name](self._seconds(tt_jd), 1),
            dtype=float,
        )

    def relative(self, name: str, tt_jd: float | np.ndarray) -> np.ndarray:
        return self.position(name, tt_jd) - self.position("earth", tt_jd)

    def longitude_offset_deg(self, tt_jd: float | np.ndarray) -> np.ndarray:
        """Return the modeled Earth-rotation offset from Skyfield ITRS.

        The baseline coupled model deliberately retains the real-Earth
        orientation history, so its offset is identically zero.  Enhanced
        ephemerides can override this method with an integrated spin model
        while reusing the same eclipse-geometry functions.
        """

        return np.zeros_like(np.asarray(tt_jd, dtype=float))

    def time(self, tt_jd: float | np.ndarray):
        return self.context.tt_jd(tt_jd)


def _earth_longitude_offset_deg(
    ephemeris: CoupledEphemeris,
    tt_jd: float | np.ndarray,
) -> np.ndarray:
    provider = getattr(ephemeris, "longitude_offset_deg", None)
    if provider is None:
        return np.zeros_like(np.asarray(tt_jd, dtype=float))
    return np.asarray(provider(tt_jd), dtype=float)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _earth_rotation_matrix(
    ephemeris: CoupledEphemeris,
    tt_jd: float,
) -> np.ndarray:
    """Return the model's ICRF-to-Earth-fixed rotation at one epoch."""

    provider = getattr(ephemeris, "rotation_matrix_icrf_to_itrs", None)
    if provider is not None:
        rotation = np.asarray(provider(tt_jd), dtype=float)
    else:
        time = ephemeris.time(tt_jd)
        offset = float(_earth_longitude_offset_deg(ephemeris, tt_jd))
        rotation = _rotation_z(np.radians(offset)) @ np.asarray(itrs.rotation_at(time), dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Earth rotation provider must return a finite 3x3 matrix")
    return rotation


def _geodetic_observer_icrf_km(
    rotation_icrf_to_itrs: np.ndarray,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
) -> np.ndarray:
    """Convert a WGS84 observer directly through the modeled Earth attitude."""

    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    height_km = altitude_m / 1_000.0
    sin_latitude = np.sin(latitude)
    prime_vertical = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_latitude * sin_latitude)
    ecef = np.asarray(
        (
            (prime_vertical + height_km) * np.cos(latitude) * np.cos(longitude),
            (prime_vertical + height_km) * np.cos(latitude) * np.sin(longitude),
            (prime_vertical * (1.0 - WGS84_E2) + height_km) * sin_latitude,
        ),
        dtype=float,
    )
    return np.asarray(rotation_icrf_to_itrs, dtype=float).T @ ecef


def _apparent_sun_from_moon(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(tt_jd, dtype=float)
    moon = ephemeris.position(body, times)
    sun_now = ephemeris.position("sun", times)
    initial_distance = np.linalg.norm(sun_now - moon, axis=-1)
    retarded = times - initial_distance / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
    sun_emit = ephemeris.position("sun", retarded)
    vector = sun_emit - moon
    return vector, np.linalg.norm(vector, axis=-1)


def coupled_shadow_state(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
) -> GenericShadowState:
    moon = np.asarray(ephemeris.relative(body, tt_jd), dtype=float)
    moon_to_sun, distance = _apparent_sun_from_moon(ephemeris, body, tt_jd)
    moon_to_sun = np.asarray(moon_to_sun, dtype=float)
    distance = float(distance)
    axis = -moon_to_sun / distance
    earth_from_moon = -moon
    axial = float(np.dot(earth_from_moon, axis))
    miss = float(np.linalg.norm(earth_from_moon - axial * axis))
    radius = body_radius_km(ephemeris, body)
    penumbra_angle = np.arcsin((SUN_RADIUS_KM + radius) / distance)
    core_angle = np.arcsin((SUN_RADIUS_KM - radius) / distance)
    penumbra = radius / np.cos(penumbra_angle) + axial * np.tan(penumbra_angle)
    core = radius / np.cos(core_angle) - axial * np.tan(core_angle)
    return GenericShadowState(
        moon_from_earth_km=moon,
        axis_icrf=axis,
        sun_moon_distance_km=distance,
        axial_distance_km=axial,
        axis_miss_km=miss,
        penumbra_radius_km=float(penumbra),
        signed_core_radius_km=float(core),
        apex_distance_km=float(radius / np.sin(core_angle)),
    )


def coupled_central_point(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
) -> tuple[float, float, float, float] | None:
    state = coupled_shadow_state(ephemeris, body, tt_jd)
    time = ephemeris.time(tt_jd)
    rotation = _earth_rotation_matrix(ephemeris, tt_jd)
    intersections = axis_ellipsoid_intersections(
        time,
        state.moon_from_earth_km,
        state.axis_icrf,
        rotation_icrf_to_itrs=rotation,
    )
    if intersections is None:
        return None
    latitude, longitude, height = itrs_to_geodetic(intersections[0])
    radius = body_radius_km(ephemeris, body)
    angle = np.arcsin((SUN_RADIUS_KM - radius) / state.sun_moon_distance_km)
    core = radius / np.cos(angle) - intersections[2][0] * np.tan(angle)
    return latitude, longitude, height, float(core)


def _retarded_topocentric_vector(
    ephemeris: CoupledEphemeris,
    body: str,
    tt_jd: float,
    observer_icrf_km: np.ndarray,
) -> np.ndarray:
    earth_reception = ephemeris.position("earth", tt_jd)
    body_now = ephemeris.position(body, tt_jd)
    initial = body_now - earth_reception - observer_icrf_km
    retarded = tt_jd - np.linalg.norm(initial) / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
    return ephemeris.position(body, retarded) - earth_reception - observer_icrf_km


def coupled_apparent_geometry(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0,
) -> ApparentDiskGeometry:
    rotation = _earth_rotation_matrix(ephemeris, tt_jd)
    observer = _geodetic_observer_icrf_km(
        rotation,
        latitude_deg,
        longitude_deg,
        altitude_m,
    )
    sun_vector = _retarded_topocentric_vector(ephemeris, "sun", tt_jd, observer)
    moon_vector = _retarded_topocentric_vector(ephemeris, body, tt_jd, observer)
    sun_distance = float(np.linalg.norm(sun_vector))
    moon_distance = float(np.linalg.norm(moon_vector))
    sun_radius = float(np.arcsin(SUN_RADIUS_KM / sun_distance))
    moon_radius = float(np.arcsin(body_radius_km(ephemeris, body) / moon_distance))
    separation = angular_separation(sun_vector, moon_vector)
    magnitude = max(0.0, (sun_radius + moon_radius - separation) / (2.0 * sun_radius))
    obscuration = disk_overlap_fraction(separation, sun_radius, moon_radius)
    sun_ecef = rotation @ sun_vector
    sun_ecef /= np.linalg.norm(sun_ecef)
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    normal = np.asarray(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    altitude = float(np.degrees(np.arcsin(np.clip(np.dot(normal, sun_ecef), -1.0, 1.0))))
    if separation >= sun_radius + moon_radius:
        kind = "none"
    elif separation <= abs(moon_radius - sun_radius):
        kind = "total" if moon_radius >= sun_radius else "annular"
    else:
        kind = "partial"
    return ApparentDiskGeometry(
        separation_rad=separation,
        sun_radius_rad=sun_radius,
        moon_radius_rad=moon_radius,
        solar_altitude_deg=altitude,
        magnitude=float(magnitude),
        obscuration=float(obscuration),
        eclipse_type=kind,
    )


def coupled_solar_altaz(
    ephemeris: CoupledEphemeris,
    tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0,
) -> tuple[float, float]:
    """Return the apparent solar altitude and azimuth for one observer.

    Azimuth is measured eastward from true north.  The calculation uses the
    same rotating WGS84 observer and retarded solar vector as the coupled
    eclipse geometry.
    """

    rotation = _earth_rotation_matrix(ephemeris, tt_jd)
    observer = _geodetic_observer_icrf_km(
        rotation,
        latitude_deg,
        longitude_deg,
        altitude_m,
    )
    sun_vector = _retarded_topocentric_vector(ephemeris, "sun", tt_jd, observer)
    sun_ecef = rotation @ sun_vector
    sun_ecef /= np.linalg.norm(sun_ecef)
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    east = np.asarray((-np.sin(longitude), np.cos(longitude), 0.0))
    north = np.asarray(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    up = np.asarray(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    altitude = float(np.degrees(np.arcsin(np.clip(np.dot(up, sun_ecef), -1.0, 1.0))))
    azimuth = float(
        np.degrees(np.arctan2(np.dot(east, sun_ecef), np.dot(north, sun_ecef))) % 360.0
    )
    return altitude, azimuth


def coupled_sky_plane_disks(
    ephemeris: CoupledEphemeris,
    tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Return apparent disk centers and radii on the local celestial sky.

    Centers use a gnomonic projection about the apparent Sun. ``east_deg`` is
    positive toward celestial east and ``north_deg`` toward celestial north.
    Light-time and topocentric parallax match :func:`coupled_apparent_geometry`.
    """

    rotation = _earth_rotation_matrix(ephemeris, tt_jd)
    observer = _geodetic_observer_icrf_km(
        rotation,
        latitude_deg,
        longitude_deg,
        altitude_m,
    )
    vectors = {
        name: _retarded_topocentric_vector(ephemeris, name, tt_jd, observer)
        for name in ("sun", "real_moon", "second_moon")
    }
    directions = {name: vector / np.linalg.norm(vector) for name, vector in vectors.items()}
    sun_direction = directions["sun"]
    celestial_pole = np.asarray((0.0, 0.0, 1.0), dtype=float)
    north = celestial_pole - np.dot(celestial_pole, sun_direction) * sun_direction
    if np.linalg.norm(north) < 1.0e-12:
        celestial_pole = np.asarray((0.0, 1.0, 0.0), dtype=float)
        north = celestial_pole - np.dot(celestial_pole, sun_direction) * sun_direction
    north /= np.linalg.norm(north)
    east = np.cross(north, sun_direction)
    east /= np.linalg.norm(east)
    radii = {
        "sun": SUN_RADIUS_KM,
        "real_moon": REAL_MOON_RADIUS_KM,
        "second_moon": body_radius_km(ephemeris, "second_moon"),
    }
    result: dict[str, dict[str, float]] = {}
    for name in ("sun", "real_moon", "second_moon"):
        direction = directions[name]
        denominator = float(np.dot(direction, sun_direction))
        result[name] = {
            "east_deg": float(np.degrees(np.arctan2(np.dot(direction, east), denominator))),
            "north_deg": float(np.degrees(np.arctan2(np.dot(direction, north), denominator))),
            "angular_radius_deg": float(
                np.degrees(np.arcsin(radii[name] / np.linalg.norm(vectors[name])))
            ),
            "distance_km": float(np.linalg.norm(vectors[name])),
        }
    return result


def _maximum_surface_point(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
) -> tuple[float, float, float] | None:
    state = coupled_shadow_state(ephemeris, body, tt_jd)
    if state.axis_miss_km > WGS84_A_KM + state.penumbra_radius_km:
        return None
    central = coupled_central_point(ephemeris, body, tt_jd)
    if central is not None:
        geometry = coupled_apparent_geometry(ephemeris, body, tt_jd, central[0], central[1])
        return central[0], central[1], geometry.magnitude
    initial = np.asarray(
        _closest_axis_surface_guess(
            ephemeris.time(tt_jd),
            state.moon_from_earth_km,
            state.axis_icrf,
            rotation_icrf_to_itrs=_earth_rotation_matrix(ephemeris, tt_jd),
        ),
        dtype=float,
    )

    def objective(coordinates: np.ndarray) -> float:
        latitude = float(np.clip(coordinates[0], -90.0, 90.0))
        longitude = float((coordinates[1] + 180.0) % 360.0 - 180.0)
        geometry = coupled_apparent_geometry(ephemeris, body, tt_jd, latitude, longitude)
        horizon_penalty = max(0.0, -geometry.solar_altitude_deg) / 10.0
        return (
            geometry.separation_rad / (geometry.sun_radius_rad + geometry.moon_radius_rad)
            + horizon_penalty
        )

    result = minimize(
        objective,
        np.asarray(initial),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 120},
    )
    latitude = float(np.clip(result.x[0], -90.0, 90.0))
    longitude = float((result.x[1] + 180.0) % 360.0 - 180.0)
    geometry = coupled_apparent_geometry(ephemeris, body, tt_jd, latitude, longitude)
    if geometry.magnitude <= 0.0 or geometry.solar_altitude_deg <= 0.0:
        return None
    return latitude, longitude, geometry.magnitude


def solve_coupled_local(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    approximate_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    *,
    half_window_hours: float = 4.0,
    step_seconds: float = 15.0,
) -> CoupledLocalCircumstances:
    limit = half_window_hours * 3_600.0

    def geometry(offset_seconds: float) -> ApparentDiskGeometry:
        return coupled_apparent_geometry(
            ephemeris,
            body,
            approximate_tt_jd + offset_seconds / SECONDS_PER_DAY,
            latitude_deg,
            longitude_deg,
        )

    def normalized_separation(offset_seconds: float) -> float:
        value = geometry(offset_seconds)
        return value.separation_rad / (value.sun_radius_rad + value.moon_radius_rad)

    maximum = minimize_scalar(
        normalized_separation,
        bounds=(-limit, limit),
        method="bounded",
        options={"xatol": 0.02},
    )
    maximum_offset = float(maximum.x)
    maximum_geometry = geometry(maximum_offset)
    grid = np.arange(-limit, limit + step_seconds, step_seconds)

    def roots(function) -> list[float]:
        values = np.asarray([function(float(offset)) for offset in grid])
        result: list[float] = []
        for left, right, y_left, y_right in zip(
            grid[:-1], grid[1:], values[:-1], values[1:], strict=True
        ):
            if y_left * y_right < 0.0:
                result.append(float(brentq(function, left, right, xtol=1e-5)))
        return result

    def external(seconds: float) -> float:
        value = geometry(seconds)
        return value.separation_rad - (value.sun_radius_rad + value.moon_radius_rad)

    def internal(seconds: float) -> float:
        value = geometry(seconds)
        return value.separation_rad - abs(value.moon_radius_rad - value.sun_radius_rad)

    external_roots = roots(external)
    internal_roots = roots(internal)

    def formatted(offset: float | None) -> str | None:
        if offset is None:
            return None
        return time_iso_utc(ephemeris.time(approximate_tt_jd + offset / SECONDS_PER_DAY))

    c1 = external_roots[0] if len(external_roots) >= 2 else None
    c4 = external_roots[-1] if len(external_roots) >= 2 else None
    c2 = internal_roots[0] if len(internal_roots) >= 2 else None
    c3 = internal_roots[-1] if len(internal_roots) >= 2 else None
    kind = maximum_geometry.eclipse_type
    if maximum_geometry.solar_altitude_deg <= 0.0:
        kind = f"{kind}-below-horizon"
    return CoupledLocalCircumstances(
        eclipse_type=kind,
        c1_utc=formatted(c1),
        c2_utc=formatted(c2),
        maximum_utc=formatted(maximum_offset) or "",
        c3_utc=formatted(c3),
        c4_utc=formatted(c4),
        magnitude=maximum_geometry.magnitude,
        obscuration=maximum_geometry.obscuration,
        central_duration_s=0.0 if c2 is None or c3 is None else float(c3 - c2),
        solar_altitude_deg=maximum_geometry.solar_altitude_deg,
        solar_angular_diameter_deg=float(np.degrees(2.0 * maximum_geometry.sun_radius_rad)),
        moon_angular_diameter_deg=float(np.degrees(2.0 * maximum_geometry.moon_radius_rad)),
    )


def generate_coupled_track(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    center_tt_jd: float,
    *,
    half_window_hours: float = 4.0,
    step_seconds: float = 30.0,
    partial_step_seconds: float = 60.0,
) -> dict[str, np.ndarray]:
    """Sample a central line, or the maximum-eclipse path for a partial event.

    A central eclipse keeps the original axis/ellipsoid-intersection algorithm
    and requested cadence.  When the shadow axis misses Earth at the event
    maximum, there is no central line; in that case the sampled path is the
    locus of surface points returned by :func:`_maximum_surface_point`.  The
    coarser default cadence keeps that numerical surface optimization practical
    while still resolving the path on a roughly minute scale.

    ``signed_core_radius_km`` is NaN for a partial maximum-eclipse path because
    those points are not intersections of the umbral or antumbral axis.
    """

    if step_seconds <= 0.0 or partial_step_seconds <= 0.0:
        raise ValueError("track sample steps must be positive")
    central_mode = coupled_central_point(ephemeris, body, center_tt_jd) is not None
    sample_step = step_seconds if central_mode else max(step_seconds, partial_step_seconds)
    offsets = np.arange(
        -half_window_hours * 3_600.0,
        half_window_hours * 3_600.0 + sample_step,
        sample_step,
    )
    times: list[float] = []
    latitudes: list[float] = []
    longitudes: list[float] = []
    cores: list[float] = []
    for offset in offsets:
        tt_jd = center_tt_jd + offset / SECONDS_PER_DAY
        if central_mode:
            central = coupled_central_point(ephemeris, body, tt_jd)
            if central is None:
                continue
            latitude, longitude, _, core = central
            geometry = coupled_apparent_geometry(ephemeris, body, tt_jd, latitude, longitude)
            if geometry.solar_altitude_deg <= 0.0:
                continue
        else:
            maximum = _maximum_surface_point(ephemeris, body, tt_jd)
            if maximum is None:
                continue
            latitude, longitude, _ = maximum
            core = float("nan")
        times.append(tt_jd)
        latitudes.append(latitude)
        longitudes.append(longitude)
        cores.append(core)

    # An event maximum known to be partial should normally be present in the
    # regular grid, but explicitly seed it if floating-point grid placement or
    # a grazing horizon leaves the sampled path empty.
    if not times and not central_mode:
        maximum = _maximum_surface_point(ephemeris, body, center_tt_jd)
        if maximum is not None:
            times.append(center_tt_jd)
            latitudes.append(maximum[0])
            longitudes.append(maximum[1])
            cores.append(float("nan"))
    return {
        "tt_jd": np.asarray(times),
        "latitude_deg": np.asarray(latitudes),
        "longitude_deg": np.asarray(longitudes),
        "signed_core_radius_km": np.asarray(cores),
    }


def _vectorized_clearance(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(tt_jd, dtype=float)
    moon = ephemeris.position(body, times)
    earth = ephemeris.position("earth", times)
    moon_relative = moon - earth
    moon_to_sun, distance = _apparent_sun_from_moon(ephemeris, body, times)
    axis = -moon_to_sun / distance[:, None]
    earth_from_moon = -moon_relative
    axial = np.sum(earth_from_moon * axis, axis=1)
    miss = np.linalg.norm(earth_from_moon - axial[:, None] * axis, axis=1)
    radius = body_radius_km(ephemeris, body)
    angle = np.arcsin((SUN_RADIUS_KM + radius) / distance)
    penumbra = radius / np.cos(angle) + axial * np.tan(angle)
    return miss - (WGS84_A_KM + penumbra), axial > 0.0, miss


def scan_coupled_eclipses(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    start_utc: str,
    end_utc: str,
    *,
    step_seconds: float = 300.0,
) -> list[CoupledEvent]:
    start_tt = float(ephemeris.context.time_utc(start_utc).tt)
    end_tt = float(ephemeris.context.time_utc(end_utc).tt)
    samples = np.arange(
        start_tt,
        end_tt + step_seconds / SECONDS_PER_DAY,
        step_seconds / SECONDS_PER_DAY,
    )
    clearance, in_front, sampled_miss = _vectorized_clearance(ephemeris, body, samples)
    inside = (clearance <= 0.0) & in_front
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])
    events: list[CoupledEvent] = []

    def scalar_clearance(tt_jd: float) -> float:
        state = coupled_shadow_state(ephemeris, body, tt_jd)
        return state.axis_miss_km - (WGS84_A_KM + state.penumbra_radius_km)

    for start_index, end_index in zip(starts, ends, strict=True):
        if start_index == 0 or end_index == len(samples) - 1:
            continue
        global_start = brentq(
            scalar_clearance,
            float(samples[start_index - 1]),
            float(samples[start_index]),
            xtol=1e-11,
        )
        global_end = brentq(
            scalar_clearance,
            float(samples[end_index]),
            float(samples[end_index + 1]),
            xtol=1e-11,
        )
        sampled_index = start_index + int(np.argmin(sampled_miss[start_index : end_index + 1]))
        reference = float(samples[sampled_index])

        def miss_seconds(seconds: float) -> float:
            return coupled_shadow_state(
                ephemeris, body, reference + seconds / SECONDS_PER_DAY
            ).axis_miss_km

        maximum = minimize_scalar(
            miss_seconds,
            bounds=(-step_seconds, step_seconds),
            method="bounded",
            options={"xatol": 0.02},
        )
        maximum_tt = reference + float(maximum.x) / SECONDS_PER_DAY
        central = coupled_central_point(ephemeris, body, maximum_tt)
        if central is None:
            point = _maximum_surface_point(ephemeris, body, maximum_tt)
            if point is None:
                continue
            latitude, longitude, _ = point
            kind = "partial"
            core = 0.0
        else:
            latitude, longitude, _, core = central
            kind = "total" if core > 0.0 else "annular"
            track = generate_coupled_track(ephemeris, body, maximum_tt, step_seconds=30.0)
            track_cores = track["signed_core_radius_km"]
            if len(track_cores) and np.min(track_cores) < 0.0 < np.max(track_cores):
                kind = "hybrid"
        local = solve_coupled_local(ephemeris, body, maximum_tt, latitude, longitude)
        events.append(
            CoupledEvent(
                body=BODY_LABEL[body],
                eclipse_type=kind,
                global_start_utc=time_iso_utc(ephemeris.time(global_start)),
                axis_maximum_utc=time_iso_utc(ephemeris.time(maximum_tt), places=6),
                global_end_utc=time_iso_utc(ephemeris.time(global_end)),
                latitude_deg=float(latitude),
                longitude_deg=float(longitude),
                c1_utc=local.c1_utc,
                c2_utc=local.c2_utc,
                local_maximum_utc=local.maximum_utc,
                c3_utc=local.c3_utc,
                c4_utc=local.c4_utc,
                magnitude=local.magnitude,
                obscuration=local.obscuration,
                central_duration_s=local.central_duration_s,
                solar_altitude_deg=local.solar_altitude_deg,
                solar_angular_diameter_deg=local.solar_angular_diameter_deg,
                moon_angular_diameter_deg=local.moon_angular_diameter_deg,
                core_radius_km=float(core),
                penumbra_margin_km=float(-scalar_clearance(maximum_tt)),
            )
        )
    return events


def _flatten_events(events: list[CoupledEvent]) -> list[dict[str, object]]:
    return [event.to_dict() for event in events]


def run_coupled_search(
    *,
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    start_utc: str = "2026-07-10T00:00:00Z",
    end_utc: str = "2027-08-13T00:00:00Z",
    sample_step_seconds: float = 300.0,
    output_dir: str | Path = "outputs/coupled",
) -> dict[str, object]:
    context = load_ephemeris(ephemeris_cache)
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = elements_from_config(payload)
    binary_architecture = architecture_from_config(payload)
    ephemeris = CoupledEphemeris(
        context,
        elements,
        end_utc,
        sample_step_seconds=sample_step_seconds,
        binary_architecture=binary_architecture,
    )
    real_events = scan_coupled_eclipses(ephemeris, "real_moon", start_utc, end_utc)
    second_events = scan_coupled_eclipses(ephemeris, "second_moon", start_utc, end_utc)
    pairs: list[dict[str, object]] = []
    geod = Geod(ellps="WGS84")
    for real in real_events:
        real_tt = float(context.time_utc(real.axis_maximum_utc).tt)
        for second in second_events:
            second_tt = float(context.time_utc(second.axis_maximum_utc).tt)
            separation_hours = abs(second_tt - real_tt) * 24.0
            if separation_hours <= 12.0:
                real_track = generate_coupled_track(
                    ephemeris, "real_moon", real_tt, step_seconds=20.0
                )
                second_track = generate_coupled_track(
                    ephemeris, "second_moon", second_tt, step_seconds=20.0
                )
                track_distance, real_index, second_index = minimum_track_distance_km(
                    real_track, second_track
                )
                real_lat = float(real_track["latitude_deg"][real_index])
                real_lon = float(real_track["longitude_deg"][real_index])
                second_lat = float(second_track["latitude_deg"][second_index])
                second_lon = float(second_track["longitude_deg"][second_index])
                real_closest_tt = float(real_track["tt_jd"][real_index])
                second_closest_tt = float(second_track["tt_jd"][second_index])
                azimuth, _, distance_m = geod.inv(real_lon, real_lat, second_lon, second_lat)
                common_lon, common_lat, _ = geod.fwd(real_lon, real_lat, azimuth, distance_m / 2.0)
                # A common surface site can encounter a shadow several hours
                # before or after the corresponding geocentric axis maximum.
                # Leave enough room to recover all four contacts for pairs
                # whose global maxima are already separated by up to 12 h.
                local_half_window_hours = max(6.0, separation_hours + 6.0)
                real_local = solve_coupled_local(
                    ephemeris,
                    "real_moon",
                    real_tt,
                    common_lat,
                    common_lon,
                    half_window_hours=local_half_window_hours,
                )
                second_local = solve_coupled_local(
                    ephemeris,
                    "second_moon",
                    second_tt,
                    common_lat,
                    common_lon,
                    half_window_hours=local_half_window_hours,
                )
                local_real_tt = float(context.time_utc(real_local.maximum_utc).tt)
                local_second_tt = float(context.time_utc(second_local.maximum_utc).tt)
                local_separation_hours = abs(local_second_tt - local_real_tt) * 24.0
                both_visible_same_location = (
                    real_local.magnitude > 0.0
                    and second_local.magnitude > 0.0
                    and real_local.solar_altitude_deg > 0.0
                    and second_local.solar_altitude_deg > 0.0
                    and local_separation_hours <= 12.0
                )
                at_least_one_local_total = (
                    real_local.eclipse_type == "total" or second_local.eclipse_type == "total"
                )
                at_least_one_global_total = real.eclipse_type in {
                    "total",
                    "hybrid",
                } or second.eclipse_type in {"total", "hybrid"}
                same_location = both_visible_same_location and at_least_one_local_total
                regional_500km = track_distance <= 500.0 and at_least_one_global_total
                pairs.append(
                    {
                        "real_maximum_utc": real.axis_maximum_utc,
                        "second_maximum_utc": second.axis_maximum_utc,
                        "maximum_separation_hours": separation_hours,
                        "real_type": real.eclipse_type,
                        "second_type": second.eclipse_type,
                        "track_distance_km": track_distance,
                        "real_track_kind": (
                            "maximum-eclipse path"
                            if real.eclipse_type == "partial"
                            else "central line"
                        ),
                        "second_track_kind": (
                            "maximum-eclipse path"
                            if second.eclipse_type == "partial"
                            else "central line"
                        ),
                        "real_closest_track_point": {
                            "utc": time_iso_utc(ephemeris.time(real_closest_tt), places=3),
                            "latitude_deg": real_lat,
                            "longitude_deg": real_lon,
                        },
                        "second_closest_track_point": {
                            "utc": time_iso_utc(ephemeris.time(second_closest_tt), places=3),
                            "latitude_deg": second_lat,
                            "longitude_deg": second_lon,
                        },
                        "best_common_latitude_deg": common_lat,
                        "best_common_longitude_deg": common_lon,
                        "local_maximum_separation_hours": local_separation_hours,
                        "both_visible_same_location": both_visible_same_location,
                        "at_least_one_total_at_common_location": at_least_one_local_total,
                        "definition_a_same_location": same_location,
                        "definition_b_500km": regional_500km,
                        "both_total_at_common_location": (
                            real_local.eclipse_type == "total"
                            and second_local.eclipse_type == "total"
                        ),
                        "thresholds": {
                            "same_location_12h": same_location,
                            "same_location_6h": same_location and local_separation_hours <= 6.0,
                            "same_location_3h": same_location and local_separation_hours <= 3.0,
                            "same_location_1h": same_location and local_separation_hours <= 1.0,
                            "track_1000km_12h": (
                                track_distance <= 1_000.0 and at_least_one_global_total
                            ),
                            "track_500km_12h": regional_500km,
                            "track_100km_12h": (
                                track_distance <= 100.0 and at_least_one_global_total
                            ),
                        },
                        "real_local": asdict(real_local),
                        "second_local": asdict(second_local),
                    }
                )
    result = {
        "schema_version": "1.0",
        "mode": "fully coupled four-body eclipse detection",
        "start_utc": start_utc,
        "end_utc": end_utc,
        "initial_elements": elements.to_dict(),
        "initial_architecture": (
            None if binary_architecture is None else binary_architecture.to_dict()
        ),
        "trajectory": ephemeris.metadata,
        "real_moon_event_count": len(real_events),
        "second_moon_event_count": len(second_events),
        "within_12h_pair_count": len(pairs),
        "real_moon_events": _flatten_events(real_events),
        "second_moon_events": _flatten_events(second_events),
        "within_12h_pairs": pairs,
        "interpretation_boundary": (
            "More dynamically self-consistent than the DE440s-forced search, but not a real "
            "alternate-Earth ephemeris because major planets, J2, tides, relativity, and figure "
            "terms are omitted; Earth rotation remains prescribed by Skyfield."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "coupled_eclipses.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(_flatten_events(real_events) + _flatten_events(second_events)).to_csv(
        output / "coupled_eclipses.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(output / "within_12h_pairs.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument("--start", default="2026-07-10T00:00:00Z")
    parser.add_argument("--end", default="2027-08-13T00:00:00Z")
    parser.add_argument("--sample-step-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", default="outputs/coupled")
    args = parser.parse_args(argv)
    result = run_coupled_search(
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        start_utc=args.start,
        end_utc=args.end,
        sample_step_seconds=args.sample_step_seconds,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "real_moon_event_count": result["real_moon_event_count"],
                "second_moon_event_count": result["second_moon_event_count"],
                "within_12h_pair_count": result["within_12h_pair_count"],
                "output": str((Path(args.output_dir) / "coupled_eclipses.json").resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
