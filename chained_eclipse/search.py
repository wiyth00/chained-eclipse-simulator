"""Staged design-mode and fixed-system chained-eclipse searches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
from pyproj import Geod
from scipy.optimize import minimize_scalar

from .constants import SECONDS_PER_DAY, WGS84_A_KM
from .eclipse_geometry import (
    apparent_disk_geometry,
    central_point_hypothetical,
    generate_central_track,
    hypothetical_shadow_state,
    minimum_track_distance_km,
    solve_local_circumstances,
)
from .ephemeris import EphemerisContext
from .models import ChainedEvent, LocalCircumstances, RealEclipse, Trajectory
from .optimize import DesignResult


@dataclass(slots=True)
class HypotheticalEclipse:
    maximum_tt_jd: float
    eclipse_type: str
    latitude_deg: float | None
    longitude_deg: float | None
    penumbra_margin_km: float
    core_radius_km: float


def _iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _interval(c2: str | None, c3: str | None) -> tuple[datetime, datetime] | None:
    if c2 is None or c3 is None:
        return None
    return _iso_datetime(c2), _iso_datetime(c3)


def _external_interval(c1: str | None, c4: str | None) -> tuple[datetime, datetime] | None:
    if c1 is None or c4 is None:
        return None
    return _iso_datetime(c1), _iso_datetime(c4)


def _disjoint(a: tuple[datetime, datetime] | None, b: tuple[datetime, datetime] | None) -> bool:
    return bool(a is not None and b is not None and (a[1] < b[0] or b[1] < a[0]))


def _visible_type(value: str) -> str:
    return value.removesuffix("-below-horizon")


def build_event(
    event_id: str,
    real: LocalCircumstances,
    second: LocalCircumstances,
    *,
    track_distance_km: float,
    latitude_deg: float,
    longitude_deg: float,
    numerical_notes: list[str] | None = None,
) -> ChainedEvent:
    gap = abs((_iso_datetime(second.maximum_utc) - _iso_datetime(real.maximum_utc)).total_seconds())
    real_type = _visible_type(real.eclipse_type)
    second_type = _visible_type(second.eclipse_type)
    visible = real.solar_altitude_deg > 0.0 and second.solar_altitude_deg > 0.0
    at_least_one_total = real_type == "total" or second_type == "total"
    definition_a = (
        visible
        and real.magnitude > 0.0
        and second.magnitude > 0.0
        and gap <= 12.0 * 3_600.0
        and at_least_one_total
    )
    definition_b = track_distance_km <= 500.0 and gap <= 12.0 * 3_600.0 and at_least_one_total
    total_intervals_disjoint = _disjoint(
        _interval(real.c2_utc, real.c3_utc), _interval(second.c2_utc, second.c3_utc)
    )
    return ChainedEvent(
        rank=0,
        event_id=event_id,
        definition_a=definition_a,
        definition_b=definition_b,
        real_eclipse=real,
        second_eclipse=second,
        midpoint_separation_s=float(gap),
        track_distance_km=float(track_distance_km),
        best_latitude_deg=float(latitude_deg),
        best_longitude_deg=float(longitude_deg),
        both_total=real_type == "total" and second_type == "total",
        two_totality_components=(
            real_type == "total" and second_type == "total" and total_intervals_disjoint
        ),
        external_contact_intervals_disjoint=_disjoint(
            _external_interval(real.c1_utc, real.c4_utc),
            _external_interval(second.c1_utc, second.c4_utc),
        ),
        thresholds={
            "same_location_12h": definition_a,
            "same_location_6h": definition_a and gap <= 6.0 * 3_600.0,
            "same_location_3h": definition_a and gap <= 3.0 * 3_600.0,
            "same_location_1h": definition_a and gap <= 1.0 * 3_600.0,
            "track_1000km_12h": (
                track_distance_km <= 1_000.0
                and gap <= 12.0 * 3_600.0
                and at_least_one_total
            ),
            "track_500km_12h": definition_b,
            "track_100km_12h": (
                track_distance_km <= 100.0
                and gap <= 12.0 * 3_600.0
                and at_least_one_total
            ),
        },
        numerical_notes=[] if numerical_notes is None else numerical_notes,
    )


def design_mode_event(
    context: EphemerisContext,
    design: DesignResult,
    *,
    bracket_step_seconds: float = 20.0,
) -> tuple[ChainedEvent, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calculate exact local circumstances for the optimized first chain."""

    latitude, longitude = design.target_latitude_deg, design.target_longitude_deg
    real = solve_local_circumstances(
        context,
        design.real_maximum_tt_jd,
        latitude,
        longitude,
        "real_moon",
        bracket_step_seconds=bracket_step_seconds,
    )
    second = solve_local_circumstances(
        context,
        design.target_second_maximum_tt_jd,
        latitude,
        longitude,
        "second_moon",
        position_provider=design.trajectory.position,
        bracket_step_seconds=bracket_step_seconds,
    )
    real_track = generate_central_track(
        context,
        design.real_maximum_tt_jd,
        "real_moon",
        step_seconds=30.0,
    )
    second_track = generate_central_track(
        context,
        design.target_second_maximum_tt_jd,
        "second_moon",
        position_provider=design.trajectory.position,
        step_seconds=30.0,
    )
    distance, _, _ = minimum_track_distance_km(real_track, second_track)
    event = build_event(
        design.real_eclipse.event_id.replace("_real", "_design_chain"),
        real,
        second,
        track_distance_km=distance,
        latitude_deg=latitude,
        longitude_deg=longitude,
        numerical_notes=[
            "Second-moon light time iterated once; synthetic aberration omitted.",
            "Partial phases may overlap; totality intervals are tested separately.",
        ],
    )
    event.rank = 1
    return event, real_track, second_track


def find_hypothetical_eclipses_near(
    context: EphemerisContext,
    trajectory: Trajectory,
    center_tt_jd: float,
    *,
    half_window_hours: float = 12.0,
    step_minutes: float = 5.0,
) -> list[HypotheticalEclipse]:
    """Find second-moon eclipse intervals around one real eclipse."""

    offsets = np.arange(
        -half_window_hours * 3_600.0,
        half_window_hours * 3_600.0 + step_minutes * 60.0,
        step_minutes * 60.0,
    )

    def clearance(offset_seconds: float) -> float:
        tt_jd = center_tt_jd + offset_seconds / SECONDS_PER_DAY
        time = context.tt_jd(tt_jd)
        state = hypothetical_shadow_state(context, time, trajectory.position(tt_jd))
        return state.axis_miss_km - (WGS84_A_KM + state.penumbra_radius_km)

    values = np.asarray([clearance(float(offset)) for offset in offsets])
    inside = values <= 0.0
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])
    events: list[HypotheticalEclipse] = []
    for start_index, end_index in zip(starts, ends, strict=True):
        left = offsets[max(0, start_index - 1)]
        right = offsets[min(len(offsets) - 1, end_index + 1)]

        def axis_miss(offset_seconds: float) -> float:
            tt_jd = center_tt_jd + offset_seconds / SECONDS_PER_DAY
            state = hypothetical_shadow_state(
                context, context.tt_jd(tt_jd), trajectory.position(tt_jd)
            )
            return state.axis_miss_km

        result = minimize_scalar(
            axis_miss,
            bounds=(float(left), float(right)),
            method="bounded",
            options={"xatol": 0.05},
        )
        maximum_tt = center_tt_jd + float(result.x) / SECONDS_PER_DAY
        time = context.tt_jd(maximum_tt)
        point = central_point_hypothetical(context, time, trajectory.position(maximum_tt))
        if point is None:
            kind = "partial"
            lat = lon = None
            core = 0.0
        else:
            lat, lon, _, core = point
            kind = "total" if core > 0.0 else "annular"
        events.append(
            HypotheticalEclipse(
                maximum_tt_jd=maximum_tt,
                eclipse_type=kind,
                latitude_deg=lat,
                longitude_deg=lon,
                penumbra_margin_km=float(-clearance(float(result.x))),
                core_radius_km=float(core),
            )
        )
    return events


def _best_common_location(
    context: EphemerisContext,
    real_track: dict[str, np.ndarray],
    second_track: dict[str, np.ndarray],
    real_index: int,
    second_index: int,
    trajectory: Trajectory,
) -> tuple[float, float]:
    lat_a = float(real_track["latitude_deg"][real_index])
    lon_a = float(real_track["longitude_deg"][real_index])
    lat_b = float(second_track["latitude_deg"][second_index])
    lon_b = float(second_track["longitude_deg"][second_index])
    time_a = context.tt_jd(float(real_track["tt_jd"][real_index]))
    time_b = context.tt_jd(float(second_track["tt_jd"][second_index]))
    geod = Geod(ellps="WGS84")
    azimuth, _, distance_m = geod.inv(lon_a, lat_a, lon_b, lat_b)
    candidates = [
        geod.fwd(lon_a, lat_a, azimuth, distance_m * fraction)[:2][::-1]
        for fraction in np.linspace(0.0, 1.0, 9)
    ]
    scored: list[tuple[float, float, float]] = []
    for lat, lon in candidates:
        real_geometry = apparent_disk_geometry(
            context, time_a, lat, lon, 0.0, "real_moon"
        )
        second_geometry = apparent_disk_geometry(
            context,
            time_b,
            lat,
            lon,
            0.0,
            "second_moon",
            position_provider=trajectory.position,
        )
        horizon_penalty = 0.0 if min(real_geometry.solar_altitude_deg, second_geometry.solar_altitude_deg) > 0 else 10.0
        total_bonus = 2.0 if (
            real_geometry.eclipse_type == "total"
            or second_geometry.eclipse_type == "total"
        ) else 0.0
        score = (
            min(real_geometry.obscuration, second_geometry.obscuration)
            + total_bonus
            - horizon_penalty
        )
        scored.append((score, lat, lon))
    _, latitude, longitude = max(scored)
    return float(latitude), float(longitude)


def fixed_system_search(
    context: EphemerisContext,
    real_eclipses: Iterable[RealEclipse],
    trajectory: Trajectory,
    *,
    maximum_events: int | None = None,
) -> tuple[list[ChainedEvent], dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]]:
    """Search a single saved orbit through all later real eclipses."""

    candidates: list[ChainedEvent] = []
    tracks: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for real_event in real_eclipses:
        real_tt = float(context.time_utc(real_event.maximum_utc).tt)
        second_events = find_hypothetical_eclipses_near(context, trajectory, real_tt)
        for second_event in second_events:
            real_track = generate_central_track(
                context,
                real_tt,
                "real_moon",
                step_seconds=120.0,
                include_partial_maximum=True,
            )
            second_track = generate_central_track(
                context,
                second_event.maximum_tt_jd,
                "second_moon",
                position_provider=trajectory.position,
                step_seconds=120.0,
                include_partial_maximum=True,
            )
            distance, real_index, second_index = minimum_track_distance_km(real_track, second_track)
            if real_index < 0 or distance > 4_000.0:
                continue
            latitude, longitude = _best_common_location(
                context,
                real_track,
                second_track,
                real_index,
                second_index,
                trajectory,
            )
            real_local = solve_local_circumstances(
                context,
                real_tt,
                latitude,
                longitude,
                "real_moon",
                bracket_step_seconds=60.0,
            )
            second_local = solve_local_circumstances(
                context,
                second_event.maximum_tt_jd,
                latitude,
                longitude,
                "second_moon",
                position_provider=trajectory.position,
                bracket_step_seconds=60.0,
            )
            event_id = real_event.event_id.replace("_real", "_fixed_chain")
            event = build_event(
                event_id,
                real_local,
                second_local,
                track_distance_km=distance,
                latitude_deg=latitude,
                longitude_deg=longitude,
                numerical_notes=[
                    "Fixed-system candidate found without changing the saved epoch orbit.",
                    "Track distance is a sampled WGS84 centerline minimum (120 s sampling).",
                ],
            )
            if event.definition_a or event.thresholds["track_1000km_12h"]:
                candidates.append(event)
                tracks[event.event_id] = (real_track, second_track)
                if maximum_events is not None and len(candidates) >= maximum_events:
                    break
        if maximum_events is not None and len(candidates) >= maximum_events:
            break
    candidates.sort(
        key=lambda event: (
            _iso_datetime(event.real_eclipse.maximum_utc),
            event.midpoint_separation_s,
            event.track_distance_km,
            -min(event.real_eclipse.magnitude, event.second_eclipse.magnitude),
        )
    )
    for rank, event in enumerate(candidates, 1):
        event.rank = rank
    return candidates, tracks
