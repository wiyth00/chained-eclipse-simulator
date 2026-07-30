"""Lunar-eclipse detection in the coupled Sun-Earth-two-moon trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
import yaml

from .constants import (
    SECONDS_PER_DAY,
    SUN_RADIUS_KM,
    WGS84_A_KM,
)
from .coupled_eclipse import BodyName, CoupledEphemeris, body_radius_km
from .eclipse_geometry import angular_separation
from .ephemeris import load_ephemeris, time_iso_utc
from .models import OrbitalElements


ContactKind = Literal["penumbral", "umbral", "total"]


@dataclass(slots=True)
class LunarEclipseEvent:
    body: str
    eclipse_type: str
    p1_utc: str
    u1_utc: str | None
    u2_utc: str | None
    maximum_utc: str
    u3_utc: str | None
    u4_utc: str | None
    p4_utc: str
    penumbral_duration_s: float
    umbral_duration_s: float
    totality_duration_s: float
    center_offset_km: float
    umbra_radius_km: float
    penumbra_radius_km: float
    moon_distance_km: float
    moon_angular_diameter_deg: float
    other_moon_angular_separation_deg: float
    other_moon_shadow_offset_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def earth_shadow_geometry(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return axial distance, center offset, penumbra, umbra, and Moon distance."""

    times = np.asarray(tt_jd, dtype=float)
    earth = ephemeris.position("earth", times)
    sun = ephemeris.position("sun", times)
    moon = ephemeris.position(body, times)
    earth_to_sun = sun - earth
    sun_distance = np.linalg.norm(earth_to_sun, axis=-1)
    axis = -earth_to_sun / sun_distance[..., None]
    earth_to_moon = moon - earth
    axial = np.sum(earth_to_moon * axis, axis=-1)
    offset_vector = earth_to_moon - axial[..., None] * axis
    offset = np.linalg.norm(offset_vector, axis=-1)
    moon_distance = np.linalg.norm(earth_to_moon, axis=-1)
    penumbra_angle = np.arcsin((SUN_RADIUS_KM + WGS84_A_KM) / sun_distance)
    umbra_angle = np.arcsin((SUN_RADIUS_KM - WGS84_A_KM) / sun_distance)
    penumbra = WGS84_A_KM / np.cos(penumbra_angle) + axial * np.tan(penumbra_angle)
    umbra = WGS84_A_KM / np.cos(umbra_angle) - axial * np.tan(umbra_angle)
    return axial, offset, penumbra, umbra, moon_distance


def _margin(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
    contact: ContactKind,
) -> float:
    axial, offset, penumbra, umbra, _ = earth_shadow_geometry(ephemeris, body, tt_jd)
    if float(axial) <= 0.0:
        return 1.0e9
    radius = body_radius_km(ephemeris, body)
    if contact == "penumbral":
        boundary = float(penumbra) + radius
    elif contact == "umbral":
        boundary = float(umbra) + radius
    else:
        boundary = float(umbra) - radius
    return float(offset) - boundary


def _contact_root(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    contact: ContactKind,
    left_tt: float,
    right_tt: float,
) -> float | None:
    left = _margin(ephemeris, body, left_tt, contact)
    right = _margin(ephemeris, body, right_tt, contact)
    if left == 0.0:
        return left_tt
    if right == 0.0:
        return right_tt
    if left * right > 0.0:
        return None
    return float(
        brentq(
            lambda value: _margin(ephemeris, body, value, contact),
            left_tt,
            right_tt,
            xtol=1.0e-11,
        )
    )


def _other_moon_geometry(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    tt_jd: float,
) -> tuple[float, float]:
    other: BodyName = "second_moon" if body == "real_moon" else "real_moon"
    earth = ephemeris.position("earth", tt_jd)
    sun_vector = ephemeris.position("sun", tt_jd) - earth
    shadow_direction = -sun_vector / np.linalg.norm(sun_vector)
    moon_vector = ephemeris.position(body, tt_jd) - earth
    other_vector = ephemeris.position(other, tt_jd) - earth
    separation = angular_separation(moon_vector, other_vector)
    shadow_offset = angular_separation(shadow_direction, other_vector)
    return float(np.degrees(separation)), float(np.degrees(shadow_offset))


def find_lunar_eclipses(
    ephemeris: CoupledEphemeris,
    start_utc: str,
    end_utc: str,
    *,
    sample_step_seconds: float = 300.0,
) -> list[LunarEclipseEvent]:
    """Find lunar eclipses of both moons and solve all applicable contacts."""

    context = ephemeris.context
    start_tt = float(context.time_utc(start_utc).tt)
    end_tt = float(context.time_utc(end_utc).tt)
    step_days = sample_step_seconds / SECONDS_PER_DAY
    times = np.arange(start_tt, end_tt + step_days, step_days)
    events: list[LunarEclipseEvent] = []
    for body in ("real_moon", "second_moon"):
        axial, offset, penumbra, _, _ = earth_shadow_geometry(ephemeris, body, times)
        radius = body_radius_km(ephemeris, body)
        penumbral_margin = offset - (penumbra + radius)
        inside = (axial > 0.0) & (penumbral_margin <= 0.0)
        transitions = np.diff(inside.astype(np.int8))
        starts = list(np.flatnonzero(transitions == 1))
        ends = list(np.flatnonzero(transitions == -1))
        if inside[0]:
            starts.insert(0, 0)
        if inside[-1]:
            ends.append(len(times) - 2)
        for start_index, end_index in zip(starts, ends, strict=True):
            p1 = _contact_root(
                ephemeris, body, "penumbral", times[start_index], times[start_index + 1]
            )
            p4 = _contact_root(
                ephemeris, body, "penumbral", times[end_index], times[end_index + 1]
            )
            if p1 is None:
                p1 = float(times[start_index])
            if p4 is None:
                p4 = float(times[end_index + 1])
            duration_seconds = (p4 - p1) * SECONDS_PER_DAY
            result = minimize_scalar(
                lambda seconds: _margin(
                    ephemeris,
                    body,
                    p1 + seconds / SECONDS_PER_DAY,
                    "umbral",
                ),
                bounds=(0.0, duration_seconds),
                method="bounded",
                options={"xatol": 1.0e-3},
            )
            maximum = p1 + float(result.x) / SECONDS_PER_DAY
            umbral_at_max = _margin(ephemeris, body, maximum, "umbral")
            total_at_max = _margin(ephemeris, body, maximum, "total")
            u1 = u4 = u2 = u3 = None
            eclipse_type = "penumbral"
            if umbral_at_max <= 0.0:
                eclipse_type = "partial"
                u1 = _contact_root(ephemeris, body, "umbral", p1, maximum)
                u4 = _contact_root(ephemeris, body, "umbral", maximum, p4)
            if total_at_max <= 0.0:
                eclipse_type = "total"
                u2 = _contact_root(ephemeris, body, "total", u1 or p1, maximum)
                u3 = _contact_root(ephemeris, body, "total", maximum, u4 or p4)
            _, center_offset, pen_radius, umb_radius, moon_distance = earth_shadow_geometry(
                ephemeris, body, maximum
            )
            other_separation, other_shadow_offset = _other_moon_geometry(
                ephemeris, body, maximum
            )
            events.append(
                LunarEclipseEvent(
                    body=body,
                    eclipse_type=eclipse_type,
                    p1_utc=time_iso_utc(context.tt_jd(p1)),
                    u1_utc=None if u1 is None else time_iso_utc(context.tt_jd(u1)),
                    u2_utc=None if u2 is None else time_iso_utc(context.tt_jd(u2)),
                    maximum_utc=time_iso_utc(context.tt_jd(maximum)),
                    u3_utc=None if u3 is None else time_iso_utc(context.tt_jd(u3)),
                    u4_utc=None if u4 is None else time_iso_utc(context.tt_jd(u4)),
                    p4_utc=time_iso_utc(context.tt_jd(p4)),
                    penumbral_duration_s=(p4 - p1) * SECONDS_PER_DAY,
                    umbral_duration_s=0.0 if u1 is None or u4 is None else (u4 - u1) * SECONDS_PER_DAY,
                    totality_duration_s=0.0 if u2 is None or u3 is None else (u3 - u2) * SECONDS_PER_DAY,
                    center_offset_km=float(center_offset),
                    umbra_radius_km=float(umb_radius),
                    penumbra_radius_km=float(pen_radius),
                    moon_distance_km=float(moon_distance),
                    moon_angular_diameter_deg=float(
                        2.0 * np.degrees(np.arcsin(radius / float(moon_distance)))
                    ),
                    other_moon_angular_separation_deg=other_separation,
                    other_moon_shadow_offset_deg=other_shadow_offset,
                )
            )
    return sorted(events, key=lambda item: item.maximum_utc)


def run_catalog(
    *,
    start_utc: str = "2026-07-10T00:00:00Z",
    end_utc: str = "2027-08-13T00:00:00Z",
    config_path: str | Path = "config/optimized_system.yaml",
    output_dir: str | Path = "outputs/coupled/lunar_eclipses",
) -> dict[str, object]:
    context = load_ephemeris("data/ephemeris")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    ephemeris = CoupledEphemeris(context, elements, end_utc, sample_step_seconds=300.0)
    events = find_lunar_eclipses(ephemeris, start_utc, end_utc)
    payload: dict[str, object] = {
        "model": ephemeris.metadata,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "events": [event.to_dict() for event in events],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "lunar_eclipses.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame([event.to_dict() for event in events]).to_csv(
        output / "lunar_eclipses.csv", index=False
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-07-10T00:00:00Z")
    parser.add_argument("--end", default="2027-08-13T00:00:00Z")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--output-dir", default="outputs/coupled/lunar_eclipses")
    args = parser.parse_args(argv)
    payload = run_catalog(
        start_utc=args.start,
        end_utc=args.end,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["events"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
