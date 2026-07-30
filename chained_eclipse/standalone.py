"""Search for standalone solar eclipses caused by the saved second moon."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import yaml

from .constants import (
    SECONDS_PER_DAY,
    SUN_RADIUS_KM,
    WGS84_A_KM,
)
from .eclipse_geometry import (
    central_point_hypothetical,
    generate_central_track,
    hypothetical_shadow_state,
    maximum_surface_point,
    solve_local_circumstances,
)
from .ephemeris import EphemerisContext, load_ephemeris, time_iso_utc
from .models import OrbitalElements, Trajectory
from .orbital_dynamics import integrate_restricted


@dataclass(slots=True)
class StandaloneEclipse:
    eclipse_type: str
    global_start_utc: str
    maximum_utc: str
    global_end_utc: str
    latitude_deg: float
    longitude_deg: float
    magnitude: float
    obscuration: float
    solar_altitude_deg: float
    central_duration_s: float
    core_radius_km: float
    penumbra_margin_km: float
    c1_utc: str | None
    c2_utc: str | None
    c3_utc: str | None
    c4_utc: str | None
    solar_angular_diameter_deg: float
    moon_angular_diameter_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _vectorized_clearance(
    context: EphemerisContext,
    trajectory: Trajectory,
    tt_jd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return penumbra clearance and Moon-behind-Earth rejection mask."""

    times = context.tt_jd(tt_jd)
    moon = np.asarray(trajectory.position(tt_jd), dtype=float)[..., :3]
    sun = np.asarray(
        context.earth.at(times).observe(context.sun).apparent().position.km,
        dtype=float,
    ).T
    moon_to_sun = sun - moon
    distance = np.linalg.norm(moon_to_sun, axis=1)
    axis = -moon_to_sun / distance[:, None]
    earth_from_moon = -moon
    axial = np.sum(earth_from_moon * axis, axis=1)
    off_axis = earth_from_moon - axial[:, None] * axis
    miss = np.linalg.norm(off_axis, axis=1)
    moon_radius_km = trajectory.second_moon_radius_km
    angle = np.arcsin((SUN_RADIUS_KM + moon_radius_km) / distance)
    penumbra = moon_radius_km / np.cos(angle) + axial * np.tan(angle)
    clearance = miss - (WGS84_A_KM + penumbra)
    return clearance, axial > 0.0, miss


def scan_standalone_eclipses(
    context: EphemerisContext,
    trajectory: Trajectory,
    start_tt_jd: float,
    end_tt_jd: float,
    *,
    step_seconds: float = 300.0,
) -> list[StandaloneEclipse]:
    """Enumerate every second-moon eclipse in a fixed propagated system."""

    samples = np.arange(
        start_tt_jd,
        end_tt_jd + step_seconds / SECONDS_PER_DAY,
        step_seconds / SECONDS_PER_DAY,
    )
    clearance, in_front, sampled_miss = _vectorized_clearance(
        context, trajectory, samples
    )
    inside = (clearance <= 0.0) & in_front
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])
    results: list[StandaloneEclipse] = []

    def scalar_clearance(tt_jd: float) -> float:
        state = hypothetical_shadow_state(
            context,
            context.tt_jd(tt_jd),
            trajectory.position(tt_jd),
            moon_radius_km=trajectory.second_moon_radius_km,
        )
        return state.axis_miss_km - (WGS84_A_KM + state.penumbra_radius_km)

    for start_index, end_index in zip(starts, ends, strict=True):
        # An interval already active at the scan boundary belongs to the prior
        # event and cannot be assigned a reliable global start here.
        if start_index == 0 or end_index == len(samples) - 1:
            continue
        start_tt = brentq(
            scalar_clearance,
            float(samples[start_index - 1]),
            float(samples[start_index]),
            xtol=1e-11,
        )
        end_tt = brentq(
            scalar_clearance,
            float(samples[end_index]),
            float(samples[end_index + 1]),
            xtol=1e-11,
        )

        sampled_index = start_index + int(
            np.argmin(sampled_miss[start_index : end_index + 1])
        )
        reference_tt = float(samples[sampled_index])

        def axis_miss_seconds(offset_seconds: float) -> float:
            tt_jd = reference_tt + offset_seconds / SECONDS_PER_DAY
            state = hypothetical_shadow_state(
                context,
                context.tt_jd(tt_jd),
                trajectory.position(tt_jd),
                moon_radius_km=trajectory.second_moon_radius_km,
            )
            return state.axis_miss_km

        maximum = minimize_scalar(
            axis_miss_seconds,
            bounds=(
                (
                    float(samples[max(start_index, sampled_index - 1)])
                    - reference_tt
                )
                * SECONDS_PER_DAY,
                (
                    float(samples[min(end_index, sampled_index + 1)])
                    - reference_tt
                )
                * SECONDS_PER_DAY,
            ),
            method="bounded",
            options={"xatol": 0.02},
        )
        maximum_tt = reference_tt + float(maximum.x) / SECONDS_PER_DAY
        maximum_time = context.tt_jd(maximum_tt)
        position = trajectory.position(maximum_tt)
        central = central_point_hypothetical(
            context,
            maximum_time,
            position,
            moon_radius_km=trajectory.second_moon_radius_km,
        )
        if central is None:
            surface = maximum_surface_point(
                context,
                maximum_time,
                "second_moon",
                position_provider=trajectory.position,
                body_radius_km=trajectory.second_moon_radius_km,
            )
            if surface is None:
                continue
            latitude, longitude, _ = surface
            eclipse_type = "partial"
            core_radius = 0.0
        else:
            latitude, longitude, _, core_radius = central
            eclipse_type = "total" if core_radius > 0.0 else "annular"
            track = generate_central_track(
                context,
                maximum_tt,
                "second_moon",
                position_provider=trajectory.position,
                body_radius_km=trajectory.second_moon_radius_km,
                half_window_hours=3.0,
                step_seconds=20.0,
            )
            cores = track["signed_core_radius_km"]
            if len(cores) and np.min(cores) < 0.0 < np.max(cores):
                eclipse_type = "hybrid"

        local = solve_local_circumstances(
            context,
            maximum_tt,
            latitude,
            longitude,
            "second_moon",
            position_provider=trajectory.position,
            body_radius_km=trajectory.second_moon_radius_km,
            search_half_window_hours=3.0,
            bracket_step_seconds=15.0,
        )
        results.append(
            StandaloneEclipse(
                eclipse_type=eclipse_type,
                global_start_utc=time_iso_utc(context.tt_jd(start_tt)),
                maximum_utc=time_iso_utc(maximum_time, places=6),
                global_end_utc=time_iso_utc(context.tt_jd(end_tt)),
                latitude_deg=float(latitude),
                longitude_deg=float(longitude),
                magnitude=local.magnitude,
                obscuration=local.obscuration,
                solar_altitude_deg=local.solar_altitude_deg,
                central_duration_s=local.central_duration_s,
                core_radius_km=float(core_radius),
                penumbra_margin_km=float(-scalar_clearance(maximum_tt)),
                c1_utc=local.c1_utc,
                c2_utc=local.c2_utc,
                c3_utc=local.c3_utc,
                c4_utc=local.c4_utc,
                solar_angular_diameter_deg=local.solar_angular_diameter_deg,
                moon_angular_diameter_deg=local.moon_angular_diameter_deg,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument("--start", default="2026-08-13T00:00:00Z")
    parser.add_argument("--end", default="2027-08-13T00:00:00Z")
    parser.add_argument("--step-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output", default="outputs/next_second_moon_eclipses.json"
    )
    args = parser.parse_args(argv)
    context = load_ephemeris(args.ephemeris_cache)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    elements = OrbitalElements(**payload["orbital_elements"])
    start = context.time_utc(args.start)
    end = context.time_utc(args.end)
    trajectory = integrate_restricted(
        context,
        elements,
        float(end.tt),
        rtol=1e-10,
        max_step_seconds=10_800.0,
    )
    events = scan_standalone_eclipses(
        context,
        trajectory,
        float(start.tt),
        float(end.tt),
        step_seconds=args.step_seconds,
    )
    data = {
        "schema_version": "1.0",
        "mode": "fixed-system",
        "start_utc": args.start,
        "end_utc": args.end,
        "event_count": len(events),
        "first_eclipse": events[0].to_dict() if events else None,
        "first_central_eclipse": next(
            (event.to_dict() for event in events if event.eclipse_type != "partial"),
            None,
        ),
        "events": [event.to_dict() for event in events],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    if events:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(events[0].to_dict()))
            writer.writeheader()
            writer.writerows(event.to_dict() for event in events)
    print(
        json.dumps(
            {
                "event_count": len(events),
                "output": str(output.resolve()),
                "csv": str(csv_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
