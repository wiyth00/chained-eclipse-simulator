"""Three-dimensional shadow geometry and exact topocentric circumstances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pyproj import Geod
from scipy.optimize import brentq, minimize, minimize_scalar
from skyfield.api import Time, wgs84
from skyfield.framelib import itrs

from .constants import (
    REAL_MOON_RADIUS_KM,
    SECOND_MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SPEED_OF_LIGHT_KM_S,
    SUN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_E2,
)
from .ephemeris import (
    EphemerisContext,
    axis_ellipsoid_intersections,
    itrs_to_geodetic,
    real_shadow_state,
    time_iso_utc,
)
from .models import LocalCircumstances

BodyKind = Literal["real_moon", "second_moon"]
PositionProvider = Callable[[float | np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ApparentDiskGeometry:
    separation_rad: float
    sun_radius_rad: float
    moon_radius_rad: float
    solar_altitude_deg: float
    magnitude: float
    obscuration: float
    eclipse_type: str


@dataclass(frozen=True, slots=True)
class GenericShadowState:
    moon_from_earth_km: np.ndarray
    axis_icrf: np.ndarray
    sun_moon_distance_km: float
    axial_distance_km: float
    axis_miss_km: float
    penumbra_radius_km: float
    signed_core_radius_km: float
    apex_distance_km: float


def angular_separation(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable angle between two Cartesian vectors."""

    ua = np.asarray(a, dtype=float)
    ub = np.asarray(b, dtype=float)
    ua /= np.linalg.norm(ua)
    ub /= np.linalg.norm(ub)
    return float(np.arctan2(np.linalg.norm(np.cross(ua, ub)), np.dot(ua, ub)))


def disk_overlap_fraction(
    separation: float, sun_radius: float, moon_radius: float
) -> float:
    """Fraction of the apparent solar disk covered by a circular moon."""

    d, r_s, r_m = float(separation), float(sun_radius), float(moon_radius)
    if d >= r_s + r_m:
        return 0.0
    if d <= abs(r_m - r_s):
        return 1.0 if r_m >= r_s else (r_m / r_s) ** 2
    arg_s = np.clip((d * d + r_s * r_s - r_m * r_m) / (2.0 * d * r_s), -1.0, 1.0)
    arg_m = np.clip((d * d + r_m * r_m - r_s * r_s) / (2.0 * d * r_m), -1.0, 1.0)
    radicand = max(
        0.0,
        (-d + r_s + r_m)
        * (d + r_s - r_m)
        * (d - r_s + r_m)
        * (d + r_s + r_m),
    )
    area = (
        r_s * r_s * np.arccos(arg_s)
        + r_m * r_m * np.arccos(arg_m)
        - 0.5 * np.sqrt(radicand)
    )
    return float(np.clip(area / (np.pi * r_s * r_s), 0.0, 1.0))


def _classify_disks(separation: float, sun_radius: float, moon_radius: float) -> str:
    if separation >= sun_radius + moon_radius:
        return "none"
    if separation <= abs(moon_radius - sun_radius):
        return "total" if moon_radius >= sun_radius else "annular"
    return "partial"


def _position_at(provider: PositionProvider, jd_tt: float) -> np.ndarray:
    value = np.asarray(provider(float(jd_tt)), dtype=float)
    if value.shape == (6,):
        return value[:3]
    if value.shape == (3,):
        return value
    if value.ndim == 2 and value.shape[-1] >= 3:
        return value[..., :3]
    raise ValueError(f"position provider returned unexpected shape {value.shape}")


def apparent_disk_geometry(
    context: EphemerisContext,
    time: Time,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    body: BodyKind,
    *,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
) -> ApparentDiskGeometry:
    """Compute direct apparent disk geometry for one fixed observer.

    Skyfield supplies fully apparent directions for the real bodies.  The
    fictional body uses a one-iteration retarded position (roughly 0.6 s at
    180,000 km); stellar aberration of the synthetic body is not modeled and is
    included in the reported uncertainty budget.
    """

    location = wgs84.latlon(latitude_deg, longitude_deg, elevation_m=altitude_m)
    observer = context.earth + location
    sun_apparent = observer.at(time).observe(context.sun).apparent()
    sun_vector = np.asarray(sun_apparent.position.km, dtype=float)
    sun_distance = float(np.linalg.norm(sun_vector))
    sun_radius = float(np.arcsin(SUN_RADIUS_KM / sun_distance))
    solar_altitude = float(sun_apparent.altaz()[0].degrees)

    if body == "real_moon":
        moon_apparent = observer.at(time).observe(context.moon).apparent()
        moon_vector = np.asarray(moon_apparent.position.km, dtype=float)
        radius = REAL_MOON_RADIUS_KM if body_radius_km is None else body_radius_km
    else:
        if position_provider is None:
            raise ValueError("a position provider is required for the second moon")
        observer_geocentric = np.asarray(location.at(time).position.km, dtype=float)
        first_position = _position_at(position_provider, float(time.tt))
        first_distance = float(np.linalg.norm(first_position - observer_geocentric))
        retarded_tt = float(time.tt) - first_distance / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
        retarded_position = _position_at(position_provider, retarded_tt)
        moon_vector = retarded_position - observer_geocentric
        radius = SECOND_MOON_RADIUS_KM if body_radius_km is None else body_radius_km
    moon_distance = float(np.linalg.norm(moon_vector))
    moon_radius = float(np.arcsin(radius / moon_distance))
    separation = angular_separation(sun_vector, moon_vector)
    magnitude = max(0.0, (sun_radius + moon_radius - separation) / (2.0 * sun_radius))
    obscuration = disk_overlap_fraction(separation, sun_radius, moon_radius)
    return ApparentDiskGeometry(
        separation_rad=separation,
        sun_radius_rad=sun_radius,
        moon_radius_rad=moon_radius,
        solar_altitude_deg=solar_altitude,
        magnitude=float(magnitude),
        obscuration=obscuration,
        eclipse_type=_classify_disks(separation, sun_radius, moon_radius),
    )


def hypothetical_shadow_state(
    context: EphemerisContext,
    time: Time,
    moon_from_earth_km: np.ndarray,
    *,
    moon_radius_km: float = SECOND_MOON_RADIUS_KM,
) -> GenericShadowState:
    """Evaluate the second moon's cone relative to Earth.

    The Sun direction uses Skyfield's apparent geocentric Sun vector.  A final
    observer calculation applies the moon's finite light time directly.
    """

    moon_position = np.asarray(moon_from_earth_km, dtype=float)
    apparent_sun_from_earth = (
        context.earth.at(time).observe(context.sun).apparent().position.km
    )
    moon_to_sun = np.asarray(apparent_sun_from_earth, dtype=float) - moon_position
    distance = float(np.linalg.norm(moon_to_sun))
    axis = -moon_to_sun / distance
    earth_from_moon = -moon_position
    axial_distance = float(np.dot(earth_from_moon, axis))
    off_axis = earth_from_moon - axial_distance * axis
    miss = float(np.linalg.norm(off_axis))
    pen_angle = np.arcsin((SUN_RADIUS_KM + moon_radius_km) / distance)
    core_angle = np.arcsin((SUN_RADIUS_KM - moon_radius_km) / distance)
    penumbra = moon_radius_km / np.cos(pen_angle) + axial_distance * np.tan(pen_angle)
    signed_core = moon_radius_km / np.cos(core_angle) - axial_distance * np.tan(core_angle)
    apex = moon_radius_km / np.sin(core_angle)
    return GenericShadowState(
        moon_from_earth_km=moon_position,
        axis_icrf=axis,
        sun_moon_distance_km=distance,
        axial_distance_km=axial_distance,
        axis_miss_km=miss,
        penumbra_radius_km=float(penumbra),
        signed_core_radius_km=float(signed_core),
        apex_distance_km=float(apex),
    )


def central_point_hypothetical(
    context: EphemerisContext,
    time: Time,
    moon_from_earth_km: np.ndarray,
    *,
    moon_radius_km: float = SECOND_MOON_RADIUS_KM,
) -> tuple[float, float, float, float] | None:
    state = hypothetical_shadow_state(
        context, time, moon_from_earth_km, moon_radius_km=moon_radius_km
    )
    intersections = axis_ellipsoid_intersections(
        time, state.moon_from_earth_km, state.axis_icrf
    )
    if intersections is None:
        return None
    near, _, roots = intersections
    lat, lon, height = itrs_to_geodetic(near)
    angle = np.arcsin((SUN_RADIUS_KM - moon_radius_km) / state.sun_moon_distance_km)
    signed_core = moon_radius_km / np.cos(angle) - roots[0] * np.tan(angle)
    return lat, lon, height, float(signed_core)


def _closest_axis_surface_guess(
    time: Time,
    point_icrf_km: np.ndarray,
    direction_icrf: np.ndarray,
    *,
    rotation_icrf_to_itrs: np.ndarray | None = None,
) -> tuple[float, float]:
    """Approximate the closest ellipsoid point to a non-intersecting axis."""

    rotation = (
        itrs.rotation_at(time)
        if rotation_icrf_to_itrs is None
        else np.asarray(rotation_icrf_to_itrs, dtype=float)
    )
    point = rotation @ np.asarray(point_icrf_km, dtype=float)
    direction = rotation @ np.asarray(direction_icrf, dtype=float)
    direction /= np.linalg.norm(direction)
    parameter = max(0.0, float(-np.dot(point, direction)))
    closest = point + parameter * direction
    denominator = np.sqrt(
        (closest[0] ** 2 + closest[1] ** 2) / WGS84_A_KM**2
        + closest[2] ** 2 / (WGS84_A_KM * (1.0 - 1.0 / 298.257_223_563)) ** 2
    )
    if denominator <= 0.0:
        closest = -direction * WGS84_A_KM
    else:
        closest /= denominator
    latitude, longitude, _ = itrs_to_geodetic(closest)
    return latitude, longitude


def maximum_surface_point(
    context: EphemerisContext,
    time: Time,
    body: BodyKind,
    *,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
) -> tuple[float, float, float] | None:
    """Find the maximum-eclipse point when no central axis intersects Earth."""

    if body == "real_moon":
        state = real_shadow_state(context, time)
        if state.axis_miss_km > WGS84_A_KM + state.penumbra_radius_km:
            return None
        central = axis_ellipsoid_intersections(
            time, state.moon_from_earth_km, state.axis_icrf
        )
        if central is not None:
            latitude, longitude, _ = itrs_to_geodetic(central[0])
            geometry = apparent_disk_geometry(
                context, time, latitude, longitude, 0.0, body
            )
            return latitude, longitude, geometry.magnitude
        initial = _closest_axis_surface_guess(
            time, state.moon_from_earth_km, state.axis_icrf
        )
    else:
        if position_provider is None:
            raise ValueError("position provider required")
        position = _position_at(position_provider, float(time.tt))
        state = hypothetical_shadow_state(
            context,
            time,
            position,
            moon_radius_km=SECOND_MOON_RADIUS_KM if body_radius_km is None else body_radius_km,
        )
        if state.axis_miss_km > WGS84_A_KM + state.penumbra_radius_km:
            return None
        central_point = central_point_hypothetical(
            context,
            time,
            position,
            moon_radius_km=SECOND_MOON_RADIUS_KM if body_radius_km is None else body_radius_km,
        )
        if central_point is not None:
            geometry = apparent_disk_geometry(
                context,
                time,
                central_point[0],
                central_point[1],
                0.0,
                body,
                position_provider=position_provider,
                body_radius_km=body_radius_km,
            )
            return central_point[0], central_point[1], geometry.magnitude
        initial = _closest_axis_surface_guess(time, position, state.axis_icrf)

    def objective(coordinates: np.ndarray) -> float:
        latitude = float(np.clip(coordinates[0], -90.0, 90.0))
        longitude = float((coordinates[1] + 180.0) % 360.0 - 180.0)
        geometry = apparent_disk_geometry(
            context,
            time,
            latitude,
            longitude,
            0.0,
            body,
            position_provider=position_provider,
            body_radius_km=body_radius_km,
        )
        horizon_penalty = max(0.0, -geometry.solar_altitude_deg) / 10.0
        return geometry.separation_rad / (
            geometry.sun_radius_rad + geometry.moon_radius_rad
        ) + horizon_penalty

    refined = minimize(
        objective,
        np.asarray(initial),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 120},
    )
    latitude = float(np.clip(refined.x[0], -90.0, 90.0))
    longitude = float((refined.x[1] + 180.0) % 360.0 - 180.0)
    geometry = apparent_disk_geometry(
        context,
        time,
        latitude,
        longitude,
        0.0,
        body,
        position_provider=position_provider,
        body_radius_km=body_radius_km,
    )
    if geometry.magnitude <= 0.0 or geometry.solar_altitude_deg <= 0.0:
        return None
    return latitude, longitude, geometry.magnitude


def _root_crossings(x: np.ndarray, y: np.ndarray, fn: Callable[[float], float]) -> list[float]:
    roots: list[float] = []
    for left, right, yl, yr in zip(x[:-1], x[1:], y[:-1], y[1:], strict=True):
        if yl == 0.0:
            roots.append(float(left))
        if yl * yr < 0.0:
            roots.append(float(brentq(fn, float(left), float(right), xtol=1e-5)))
    if y[-1] == 0.0:
        roots.append(float(x[-1]))
    deduped: list[float] = []
    for root in roots:
        if not deduped or abs(root - deduped[-1]) > 0.01:
            deduped.append(root)
    return deduped


def solve_local_circumstances(
    context: EphemerisContext,
    approximate_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    body: BodyKind,
    *,
    altitude_m: float = 0.0,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
    search_half_window_hours: float = 4.0,
    bracket_step_seconds: float = 30.0,
) -> LocalCircumstances:
    """Refine maximum and C1–C4 with topocentric disk geometry and Brent roots."""

    def evaluate(offset_seconds: float) -> ApparentDiskGeometry:
        time = context.tt_jd(approximate_tt_jd + offset_seconds / SECONDS_PER_DAY)
        return apparent_disk_geometry(
            context,
            time,
            latitude_deg,
            longitude_deg,
            altitude_m,
            body,
            position_provider=position_provider,
            body_radius_km=body_radius_km,
        )

    limit = search_half_window_hours * 3_600.0
    # Obscuration is flat at 100% throughout totality and flat at zero far
    # outside an eclipse, so it is a poor scalar objective for a wide bounded
    # search.  The normalized center separation is smooth and selects the
    # conventional instant of deepest alignment; obscuration is then evaluated
    # at that refined time.
    def normalized_separation(seconds: float) -> float:
        geometry = evaluate(float(seconds))
        return geometry.separation_rad / (
            geometry.sun_radius_rad + geometry.moon_radius_rad
        )

    maximum_search = minimize_scalar(
        normalized_separation,
        bounds=(-limit, limit),
        method="bounded",
        options={"xatol": 0.02},
    )
    max_offset = float(maximum_search.x)
    max_geometry = evaluate(max_offset)
    max_time = context.tt_jd(approximate_tt_jd + max_offset / SECONDS_PER_DAY)

    grid = np.arange(-limit, limit + bracket_step_seconds, bracket_step_seconds)

    def external(seconds: float) -> float:
        g = evaluate(seconds)
        return g.separation_rad - (g.sun_radius_rad + g.moon_radius_rad)

    external_values = np.asarray([external(float(value)) for value in grid])
    external_roots = _root_crossings(grid, external_values, external)
    c1 = external_roots[0] if len(external_roots) >= 2 else None
    c4 = external_roots[-1] if len(external_roots) >= 2 else None

    def internal(seconds: float) -> float:
        g = evaluate(seconds)
        return g.separation_rad - abs(g.moon_radius_rad - g.sun_radius_rad)

    internal_values = np.asarray([internal(float(value)) for value in grid])
    internal_roots = _root_crossings(grid, internal_values, internal)
    c2 = internal_roots[0] if len(internal_roots) >= 2 else None
    c3 = internal_roots[-1] if len(internal_roots) >= 2 else None

    def formatted(offset: float | None) -> str | None:
        if offset is None:
            return None
        return time_iso_utc(context.tt_jd(approximate_tt_jd + offset / SECONDS_PER_DAY))

    central_duration = 0.0 if c2 is None or c3 is None else float(c3 - c2)
    label = "real Moon" if body == "real_moon" else "second moon"
    eclipse_type = max_geometry.eclipse_type
    if max_geometry.solar_altitude_deg <= 0.0:
        eclipse_type = f"{eclipse_type}-below-horizon"
    return LocalCircumstances(
        body=label,
        eclipse_type=eclipse_type,
        latitude_deg=float(latitude_deg),
        longitude_deg=float(longitude_deg),
        altitude_m=float(altitude_m),
        c1_utc=formatted(c1),
        c2_utc=formatted(c2),
        maximum_utc=time_iso_utc(max_time),
        c3_utc=formatted(c3),
        c4_utc=formatted(c4),
        magnitude=max_geometry.magnitude,
        obscuration=max_geometry.obscuration,
        central_duration_s=central_duration,
        solar_altitude_deg=max_geometry.solar_altitude_deg,
        center_separation_deg=float(np.degrees(max_geometry.separation_rad)),
        solar_angular_diameter_deg=float(np.degrees(2.0 * max_geometry.sun_radius_rad)),
        moon_angular_diameter_deg=float(np.degrees(2.0 * max_geometry.moon_radius_rad)),
    )


def generate_central_track(
    context: EphemerisContext,
    center_tt_jd: float,
    body: BodyKind,
    *,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
    half_window_hours: float = 3.0,
    step_seconds: float = 60.0,
    include_partial_maximum: bool = False,
) -> dict[str, np.ndarray]:
    offsets = np.arange(
        -half_window_hours * 3_600.0,
        half_window_hours * 3_600.0 + step_seconds,
        step_seconds,
    )
    times: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    cores: list[float] = []
    for offset in offsets:
        time = context.tt_jd(center_tt_jd + offset / SECONDS_PER_DAY)
        if body == "real_moon":
            state = real_shadow_state(context, time)
            intersections = axis_ellipsoid_intersections(
                time, state.moon_from_earth_km, state.axis_icrf
            )
            if intersections is None:
                if not include_partial_maximum:
                    continue
                maximum_point = maximum_surface_point(context, time, body)
                if maximum_point is None:
                    continue
                point = (maximum_point[0], maximum_point[1], 0.0)
                core = 0.0
            else:
                point = itrs_to_geodetic(intersections[0])
                angle = np.arcsin(
                    (SUN_RADIUS_KM - REAL_MOON_RADIUS_KM) / state.sun_moon_distance_km
                )
                core = REAL_MOON_RADIUS_KM / np.cos(angle) - intersections[2][0] * np.tan(angle)
        else:
            if position_provider is None:
                raise ValueError("position provider required")
            position = _position_at(position_provider, float(time.tt))
            central = central_point_hypothetical(
                context,
                time,
                position,
                moon_radius_km=SECOND_MOON_RADIUS_KM if body_radius_km is None else body_radius_km,
            )
            if central is None:
                if not include_partial_maximum:
                    continue
                maximum_point = maximum_surface_point(
                    context,
                    time,
                    body,
                    position_provider=position_provider,
                    body_radius_km=body_radius_km,
                )
                if maximum_point is None:
                    continue
                point = (maximum_point[0], maximum_point[1], 0.0)
                core = 0.0
            else:
                point = central[:3]
                core = central[3]
        lat, lon, _ = point
        # Axis intersections on the night side can occur mathematically; the
        # local Sun-altitude check removes them from an eclipse ground track.
        geometry = apparent_disk_geometry(
            context,
            time,
            lat,
            lon,
            0.0,
            body,
            position_provider=position_provider,
            body_radius_km=body_radius_km,
        )
        if geometry.solar_altitude_deg <= 0.0:
            continue
        times.append(float(time.tt))
        lats.append(float(lat))
        lons.append(float(lon))
        cores.append(float(core))
    return {
        "tt_jd": np.asarray(times),
        "latitude_deg": np.asarray(lats),
        "longitude_deg": np.asarray(lons),
        "signed_core_radius_km": np.asarray(cores),
    }


def minimum_track_distance_km(
    track_a: dict[str, np.ndarray], track_b: dict[str, np.ndarray]
) -> tuple[float, int, int]:
    """Minimum WGS84 geodesic distance between sampled centerlines."""

    if not len(track_a["latitude_deg"]) or not len(track_b["latitude_deg"]):
        return float("inf"), -1, -1
    geod = Geod(ellps="WGS84")
    best = (float("inf"), -1, -1)
    b_lat = np.asarray(track_b["latitude_deg"])
    b_lon = np.asarray(track_b["longitude_deg"])
    for i, (lat, lon) in enumerate(
        zip(track_a["latitude_deg"], track_a["longitude_deg"], strict=True)
    ):
        _, _, distances_m = geod.inv(
            np.full_like(b_lon, lon, dtype=float),
            np.full_like(b_lat, lat, dtype=float),
            b_lon,
            b_lat,
        )
        j = int(np.argmin(distances_m))
        distance = float(distances_m[j] / 1_000.0)
        if distance < best[0]:
            best = (distance, i, j)
    return best


def midpoint_on_sphere(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> tuple[float, float]:
    vectors = []
    for latitude, longitude in ((latitude_a, longitude_a), (latitude_b, longitude_b)):
        lat = np.radians(latitude)
        lon = np.radians(longitude)
        vectors.append(np.asarray((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))))
    vector = vectors[0] + vectors[1]
    vector /= np.linalg.norm(vector)
    return float(np.degrees(np.arcsin(vector[2]))), float(np.degrees(np.arctan2(vector[1], vector[0])))


def angular_series(
    context: EphemerisContext,
    center_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    body: BodyKind,
    *,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
    half_window_minutes: float = 120.0,
    step_seconds: float = 20.0,
) -> dict[str, np.ndarray]:
    offsets = np.arange(
        -half_window_minutes * 60.0,
        half_window_minutes * 60.0 + step_seconds,
        step_seconds,
    )
    values = [
        apparent_disk_geometry(
            context,
            context.tt_jd(center_tt_jd + value / SECONDS_PER_DAY),
            latitude_deg,
            longitude_deg,
            0.0,
            body,
            position_provider=position_provider,
            body_radius_km=body_radius_km,
        )
        for value in offsets
    ]
    return {
        "offset_seconds": offsets,
        "separation_deg": np.degrees([value.separation_rad for value in values]),
        "sun_radius_deg": np.degrees([value.sun_radius_rad for value in values]),
        "moon_radius_deg": np.degrees([value.moon_radius_rad for value in values]),
        "magnitude": np.asarray([value.magnitude for value in values]),
        "solar_altitude_deg": np.asarray([value.solar_altitude_deg for value in values]),
    }


def maximum_visibility_grid(
    context: EphemerisContext,
    center_tt_jd: float,
    body: BodyKind,
    *,
    position_provider: PositionProvider | None = None,
    body_radius_km: float | None = None,
    half_window_hours: float = 3.0,
    time_step_minutes: float = 4.0,
    grid_step_deg: float = 2.0,
) -> dict[str, np.ndarray]:
    """Grid the maximum topocentric eclipse magnitude over an event window.

    This is a vectorized disk-overlap screen on a rotating WGS84 ellipsoid.
    Its binary visibility contour is used only for plotted partial-eclipse
    envelopes; exact candidate circumstances use the scalar apparent solver.
    """

    longitudes = np.arange(-180.0, 180.0, grid_step_deg)
    latitudes = np.arange(-90.0, 90.0 + 0.5 * grid_step_deg, grid_step_deg)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lat_rad = np.radians(lat_grid.ravel())
    lon_rad = np.radians(lon_grid.ravel())
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    observer_ecef = np.vstack(
        (
            n * cos_lat * np.cos(lon_rad),
            n * cos_lat * np.sin(lon_rad),
            n * (1.0 - WGS84_E2) * sin_lat,
        )
    )
    normal_ecef = np.vstack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), sin_lat)
    )
    maximum = np.zeros(observer_ecef.shape[1], dtype=float)
    offsets = np.arange(
        -half_window_hours * 3_600.0,
        half_window_hours * 3_600.0 + time_step_minutes * 60.0,
        time_step_minutes * 60.0,
    )
    radius = REAL_MOON_RADIUS_KM if body == "real_moon" else (
        SECOND_MOON_RADIUS_KM if body_radius_km is None else body_radius_km
    )
    for offset in offsets:
        tt_jd = center_tt_jd + offset / SECONDS_PER_DAY
        time = context.tt_jd(tt_jd)
        rotation = itrs.rotation_at(time)
        sun_icrf = np.asarray(
            context.earth.at(time).observe(context.sun).apparent().position.km,
            dtype=float,
        )
        if body == "real_moon":
            moon_icrf = np.asarray(
                context.earth.at(time).observe(context.moon).apparent().position.km,
                dtype=float,
            )
        else:
            if position_provider is None:
                raise ValueError("position provider required")
            moon_icrf = _position_at(position_provider, tt_jd)
        sun_ecef = rotation @ sun_icrf
        moon_ecef = rotation @ moon_icrf
        sun_topo = sun_ecef[:, None] - observer_ecef
        moon_topo = moon_ecef[:, None] - observer_ecef
        sun_distance = np.linalg.norm(sun_topo, axis=0)
        moon_distance = np.linalg.norm(moon_topo, axis=0)
        sun_unit = sun_topo / sun_distance
        moon_unit = moon_topo / moon_distance
        dot = np.clip(np.sum(sun_unit * moon_unit, axis=0), -1.0, 1.0)
        cross = np.linalg.norm(np.cross(sun_unit.T, moon_unit.T), axis=1)
        separation = np.arctan2(cross, dot)
        sun_radius = np.arcsin(SUN_RADIUS_KM / sun_distance)
        moon_radius = np.arcsin(radius / moon_distance)
        magnitude = np.maximum(
            0.0, (sun_radius + moon_radius - separation) / (2.0 * sun_radius)
        )
        sun_above = np.sum(normal_ecef * sun_unit, axis=0) > 0.0
        maximum = np.maximum(maximum, np.where(sun_above, magnitude, 0.0))
    shape = lat_grid.shape
    return {
        "longitude_deg": longitudes,
        "latitude_deg": latitudes,
        "maximum_magnitude": maximum.reshape(shape),
        "visibility_mask": (maximum.reshape(shape) > 0.0).astype(float),
    }
