"""JPL ephemeris loading and real-Moon eclipse enumeration.

Skyfield is used to enumerate new moons only.  Each retained lunation is
independently accepted or rejected using a three-dimensional shadow cone and a
rotating WGS84 ellipsoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from skyfield import almanac
from skyfield.api import Loader, Time
from skyfield.framelib import itrs

from .constants import (
    REAL_MOON_RADIUS_KM,
    SUN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_B_KM,
    WGS84_E2,
)
from .models import RealEclipse


@dataclass(slots=True)
class EphemerisContext:
    """Loaded DE440s kernel and Skyfield objects."""

    loader: Loader
    timescale: object
    ephemeris: object
    earth: object
    moon: object
    sun: object
    cache_dir: Path
    kernel_path: Path

    def time_utc(self, value: str) -> Time:
        return self.timescale.from_datetime(_parse_utc(value))

    def tt_jd(self, jd: float | np.ndarray) -> Time:
        return self.timescale.tt_jd(jd)


@dataclass(frozen=True, slots=True)
class RealShadowState:
    moon_from_earth_km: np.ndarray
    axis_icrf: np.ndarray
    sun_moon_distance_km: float
    axial_distance_km: float
    axis_miss_km: float
    penumbra_radius_km: float
    signed_umbra_radius_km: float


def _parse_utc(value: str):
    from datetime import datetime, timezone

    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_ephemeris(
    cache_dir: str | Path = "data/ephemeris", kernel: str = "de440s.bsp"
) -> EphemerisContext:
    """Load (and automatically download) the JPL kernel into a local cache."""

    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(cache), verbose=True)
    timescale = loader.timescale()
    ephemeris = loader(kernel)
    return EphemerisContext(
        loader=loader,
        timescale=timescale,
        ephemeris=ephemeris,
        earth=ephemeris["earth"],
        moon=ephemeris["moon"],
        sun=ephemeris["sun"],
        cache_dir=cache,
        kernel_path=cache / kernel,
    )


def time_iso_utc(time: Time, places: int = 3) -> str:
    """Return a stable ISO UTC string using Skyfield's loaded leap-second table."""

    text = time.utc_strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if places >= 6:
        return text
    head, fraction = text[:-1].split(".")
    return f"{head}.{fraction[:places]}Z" if places else f"{head}Z"


def real_shadow_state(context: EphemerisContext, time: Time) -> RealShadowState:
    """Evaluate the real lunar shadow using an apparent Moon-to-Sun direction.

    The negative apparent Sun direction as observed from the Moon gives
    sub-second agreement with published Besselian greatest-eclipse times.  A
    same-epoch geometric Sun-to-Moon vector is measurably worse.
    """

    earth_position = context.earth.at(time).position.km
    moon_position = context.moon.at(time).position.km
    moon_to_apparent_sun = (
        context.moon.at(time).observe(context.sun).apparent().position.km
    )
    distance = float(np.linalg.norm(moon_to_apparent_sun))
    axis = -moon_to_apparent_sun / distance
    moon_from_earth = moon_position - earth_position
    earth_from_moon = -moon_from_earth
    axial_distance = float(np.dot(earth_from_moon, axis))
    off_axis = earth_from_moon - axial_distance * axis
    axis_miss = float(np.linalg.norm(off_axis))

    penumbra_angle = np.arcsin(
        (SUN_RADIUS_KM + REAL_MOON_RADIUS_KM) / distance
    )
    umbra_angle = np.arcsin((SUN_RADIUS_KM - REAL_MOON_RADIUS_KM) / distance)
    penumbra_radius = (
        REAL_MOON_RADIUS_KM / np.cos(penumbra_angle)
        + axial_distance * np.tan(penumbra_angle)
    )
    signed_umbra_radius = (
        REAL_MOON_RADIUS_KM / np.cos(umbra_angle)
        - axial_distance * np.tan(umbra_angle)
    )
    return RealShadowState(
        moon_from_earth,
        axis,
        distance,
        axial_distance,
        axis_miss,
        float(penumbra_radius),
        float(signed_umbra_radius),
    )


def axis_ellipsoid_intersections(
    time: Time,
    point_icrf_km: np.ndarray,
    direction_icrf: np.ndarray,
    *,
    rotation_icrf_to_itrs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Intersect a forward ICRF line with the rotating WGS84 ellipsoid.

    Returns the two ITRS points and their line parameters.  The smaller
    positive root is the Sun-facing central-line point for a solar eclipse.
    """

    rotation = (
        itrs.rotation_at(time)
        if rotation_icrf_to_itrs is None
        else np.asarray(rotation_icrf_to_itrs, dtype=float)
    )
    if np.shape(rotation) != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_icrf_to_itrs must be a finite 3x3 matrix")
    point = rotation @ np.asarray(point_icrf_km, dtype=float)
    direction = rotation @ np.asarray(direction_icrf, dtype=float)
    direction /= np.linalg.norm(direction)
    qa = (
        (direction[0] ** 2 + direction[1] ** 2) / WGS84_A_KM**2
        + direction[2] ** 2 / WGS84_B_KM**2
    )
    qb = 2.0 * (
        (point[0] * direction[0] + point[1] * direction[1]) / WGS84_A_KM**2
        + point[2] * direction[2] / WGS84_B_KM**2
    )
    qc = (
        (point[0] ** 2 + point[1] ** 2) / WGS84_A_KM**2
        + point[2] ** 2 / WGS84_B_KM**2
        - 1.0
    )
    discriminant = qb * qb - 4.0 * qa * qc
    if discriminant < 0.0:
        return None
    root_disc = np.sqrt(discriminant)
    roots = np.sort(np.asarray(((-qb - root_disc) / (2.0 * qa), (-qb + root_disc) / (2.0 * qa))))
    roots = roots[roots > 0.0]
    if len(roots) != 2:
        return None
    return point + roots[0] * direction, point + roots[1] * direction, roots


def itrs_to_geodetic(point_itrs_km: np.ndarray) -> tuple[float, float, float]:
    """Convert an ITRS Cartesian point to WGS84 geodetic coordinates."""

    x, y, z = np.asarray(point_itrs_km, dtype=float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / max(np.cos(lat), 1e-15) - n
        next_lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(next_lat - lat) < 1e-14:
            lat = next_lat
            break
        lat = next_lat
    sin_lat = np.sin(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / max(np.cos(lat), 1e-15) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(height)


def central_point_real(
    context: EphemerisContext, time: Time
) -> tuple[float, float, float, float] | None:
    """Return latitude, longitude, altitude, and near-side core radius."""

    state = real_shadow_state(context, time)
    intersections = axis_ellipsoid_intersections(
        time, state.moon_from_earth_km, state.axis_icrf
    )
    if intersections is None:
        return None
    near, _, roots = intersections
    lat, lon, height = itrs_to_geodetic(near)
    umbra_angle = np.arcsin(
        (SUN_RADIUS_KM - REAL_MOON_RADIUS_KM) / state.sun_moon_distance_km
    )
    signed_radius = (
        REAL_MOON_RADIUS_KM / np.cos(umbra_angle)
        - roots[0] * np.tan(umbra_angle)
    )
    return lat, lon, height, float(signed_radius)


def enumerate_real_solar_eclipses(
    context: EphemerisContext,
    start: Time,
    end: Time,
    *,
    screen_margin_km: float = 3_000.0,
) -> list[RealEclipse]:
    """Enumerate and independently verify every real-Moon solar eclipse."""

    phase_times, phases = almanac.find_discrete(
        start, end, almanac.moon_phases(context.ephemeris)
    )
    new_moons = phase_times[phases == 0]

    earth_positions = context.earth.at(new_moons).position.km
    moon_positions = context.moon.at(new_moons).position.km
    apparent_sun = (
        context.moon.at(new_moons).observe(context.sun).apparent().position.km
    )
    distances = np.linalg.norm(apparent_sun, axis=0)
    axes = -apparent_sun / distances
    earth_from_moon = earth_positions - moon_positions
    axial = np.sum(earth_from_moon * axes, axis=0)
    radial = np.linalg.norm(earth_from_moon - axial * axes, axis=0)
    penumbra = REAL_MOON_RADIUS_KM + axial * (
        SUN_RADIUS_KM + REAL_MOON_RADIUS_KM
    ) / distances
    coarse_clearance = radial - (WGS84_A_KM + penumbra)
    retained = np.flatnonzero(coarse_clearance < screen_margin_km)

    events: list[RealEclipse] = []
    for index in retained:
        epoch_tt = float(new_moons[index].tt)

        def clearance(offset_days: float) -> float:
            state = real_shadow_state(context, context.tt_jd(epoch_tt + offset_days))
            return state.axis_miss_km - (WGS84_A_KM + state.penumbra_radius_km)

        minimum_clearance = minimize_scalar(
            clearance,
            bounds=(-0.25, 0.25),
            method="bounded",
            options={"xatol": 0.02 / 86_400.0},
        )
        if minimum_clearance.fun > 0.0:
            continue

        def axis_miss(offset_days: float) -> float:
            return real_shadow_state(
                context, context.tt_jd(epoch_tt + offset_days)
            ).axis_miss_km

        greatest = minimize_scalar(
            axis_miss,
            bounds=(-0.25, 0.25),
            method="bounded",
            options={"xatol": 0.02 / 86_400.0},
        )
        maximum = context.tt_jd(epoch_tt + greatest.x)
        state = real_shadow_state(context, maximum)
        central = central_point_real(context, maximum)
        if central is None:
            kind = "partial"
            lat = lon = altitude = None
            core_margin = abs(state.signed_umbra_radius_km) - max(
                0.0, state.axis_miss_km - WGS84_A_KM
            )
        else:
            lat, lon, altitude, near_core_radius = central
            kind = "total" if near_core_radius > 0.0 else "annular"
            core_margin = abs(near_core_radius)
        event_id = maximum.utc_strftime("%Y%m%d") + "_real"
        events.append(
            RealEclipse(
                event_id=event_id,
                maximum_utc=time_iso_utc(maximum),
                eclipse_type=kind,
                latitude_deg=lat,
                longitude_deg=lon,
                axis_distance_km=state.axis_miss_km,
                penumbra_margin_km=float(-minimum_clearance.fun),
                core_margin_km=float(core_margin),
                solar_altitude_deg=None,
            )
        )
    return events


def iter_real_track(
    context: EphemerisContext,
    center_tt_jd: float,
    offsets_seconds: Iterable[float],
) -> Iterable[tuple[float, float, float, float]]:
    """Yield time offset, latitude, longitude, and signed core radius."""

    for seconds in offsets_seconds:
        time = context.tt_jd(center_tt_jd + seconds / 86_400.0)
        point = central_point_real(context, time)
        if point is not None:
            lat, lon, _, core = point
            yield float(seconds), lat, lon, core
