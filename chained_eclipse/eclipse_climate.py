"""Thirty-year eclipse-climate catalog and visualization for the coupled system."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import spearmanr
import yaml

from .constants import OBLIQUITY_J2000_DEG, SECONDS_PER_DAY
from .coupled_eclipse import (
    BODY_LABEL,
    BodyName,
    CoupledEphemeris,
    _maximum_surface_point,
    _vectorized_clearance,
    coupled_apparent_geometry,
    coupled_central_point,
    coupled_shadow_state,
    generate_coupled_track,
    solve_coupled_local,
)
from .ephemeris import load_ephemeris, time_iso_utc
from .eclipse_geometry import minimum_track_distance_km
from .lunar_eclipse import find_lunar_eclipses
from .models import OrbitalElements


REAL_BLUE = "#24649A"
SECOND_ORANGE = "#D47424"
INK = "#102039"
MUTED = "#68778F"
GRID = "#DCE4EC"


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def _iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scan_solar_chunk(
    ephemeris: CoupledEphemeris,
    body: BodyName,
    start_tt: float,
    end_tt: float,
    *,
    step_seconds: float,
) -> list[dict[str, Any]]:
    """Find every refined local minimum of global penumbral clearance."""

    step_days = step_seconds / SECONDS_PER_DAY
    samples = np.arange(start_tt, end_tt + step_days, step_days)
    clearance, in_front, _ = _vectorized_clearance(ephemeris, body, samples)
    minima = np.flatnonzero(
        in_front[1:-1]
        & (clearance[1:-1] <= clearance[:-2])
        & (clearance[1:-1] <= clearance[2:])
    ) + 1

    # Skyfield's Earth ephemeris object has no shape model; use the same WGS84
    # equatorial radius as the production solar-eclipse solver.
    from .constants import WGS84_A_KM

    def physical_clearance(tt_jd: float) -> float:
        state = coupled_shadow_state(ephemeris, body, tt_jd)
        return state.axis_miss_km - (WGS84_A_KM + state.penumbra_radius_km)

    events: list[dict[str, Any]] = []
    for sample_index in minima:
        reference = float(samples[sample_index])

        def miss_seconds(seconds: float) -> float:
            return coupled_shadow_state(
                ephemeris,
                body,
                reference + seconds / SECONDS_PER_DAY,
            ).axis_miss_km

        refined = minimize_scalar(
            miss_seconds,
            bounds=(-step_seconds, step_seconds),
            method="bounded",
            options={"xatol": 0.02},
        )
        maximum_tt = reference + float(refined.x) / SECONDS_PER_DAY
        if physical_clearance(maximum_tt) > 0.0:
            continue

        left = maximum_tt
        right = maximum_tt
        for _ in range(48):
            if physical_clearance(left) > 0.0:
                break
            left -= step_days
        for _ in range(48):
            if physical_clearance(right) > 0.0:
                break
            right += step_days
        if physical_clearance(left) <= 0.0 or physical_clearance(right) <= 0.0:
            continue
        global_start = brentq(physical_clearance, left, maximum_tt, xtol=1.0e-11)
        global_end = brentq(physical_clearance, maximum_tt, right, xtol=1.0e-11)

        state = coupled_shadow_state(ephemeris, body, maximum_tt)
        central = coupled_central_point(ephemeris, body, maximum_tt)
        if central is None:
            point = _maximum_surface_point(ephemeris, body, maximum_tt)
            if point is None:
                continue
            latitude, longitude, magnitude = point
            eclipse_type = "partial"
            core_radius = 0.0
            solar_altitude = coupled_apparent_geometry(
                ephemeris,
                body,
                maximum_tt,
                latitude,
                longitude,
            ).solar_altitude_deg
        else:
            latitude, longitude, _, core_radius = central
            geometry = coupled_apparent_geometry(
                ephemeris,
                body,
                maximum_tt,
                latitude,
                longitude,
            )
            magnitude = geometry.magnitude
            solar_altitude = geometry.solar_altitude_deg
            eclipse_type = "total" if core_radius > 0.0 else "annular"
            if abs(core_radius) < 150.0:
                track = generate_coupled_track(
                    ephemeris,
                    body,
                    maximum_tt,
                    step_seconds=180.0,
                )
                cores = track["signed_core_radius_km"]
                if len(cores) and np.min(cores) < 0.0 < np.max(cores):
                    eclipse_type = "hybrid"
        events.append(
            {
                "domain": "solar",
                "body": body,
                "body_label": BODY_LABEL[body],
                "eclipse_type": eclipse_type,
                "global_start_utc": time_iso_utc(ephemeris.time(global_start)),
                "maximum_utc": time_iso_utc(ephemeris.time(maximum_tt), places=6),
                "global_end_utc": time_iso_utc(ephemeris.time(global_end)),
                "latitude_deg": float(latitude),
                "longitude_deg": float(longitude),
                "magnitude": float(magnitude),
                "solar_altitude_deg": float(solar_altitude),
                "axis_miss_km": float(state.axis_miss_km),
                "core_radius_km": float(core_radius),
                "penumbra_margin_km": float(-physical_clearance(maximum_tt)),
                "central_duration_s": None,
            }
        )
    return events


def _deduplicate(events: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for body in ("real_moon", "second_moon"):
        ordered = sorted(
            (event for event in events if event["body"] == body),
            key=lambda item: str(item[key]),
        )
        body_result: list[dict[str, Any]] = []
        for event in ordered:
            if not body_result:
                body_result.append(event)
                continue
            difference = abs(
                (
                    _iso_datetime(str(event[key]))
                    - _iso_datetime(str(body_result[-1][key]))
                ).total_seconds()
            )
            if difference < 21_600.0:
                if float(event.get("penumbra_margin_km", 0.0)) > float(
                    body_result[-1].get("penumbra_margin_km", 0.0)
                ):
                    body_result[-1] = event
                continue
            body_result.append(event)
        result.extend(body_result)
    return sorted(result, key=lambda item: str(item[key]))


def scan_solar_climate(
    ephemeris: CoupledEphemeris,
    start_utc: str,
    end_utc: str,
    *,
    step_seconds: float = 600.0,
    chunk_days: float = 366.0,
    overlap_days: float = 2.0,
) -> list[dict[str, Any]]:
    start_tt = float(ephemeris.context.time_utc(start_utc).tt)
    end_tt = float(ephemeris.context.time_utc(end_utc).tt)
    events: list[dict[str, Any]] = []
    chunk_start = start_tt
    while chunk_start < end_tt:
        chunk_end = min(end_tt, chunk_start + chunk_days)
        search_start = max(start_tt, chunk_start - overlap_days)
        search_end = min(end_tt, chunk_end + overlap_days)
        for body in ("real_moon", "second_moon"):
            events.extend(
                _scan_solar_chunk(
                    ephemeris,
                    body,
                    search_start,
                    search_end,
                    step_seconds=step_seconds,
                )
            )
        chunk_start = chunk_end
    return _deduplicate(events, "maximum_utc")


def scan_lunar_climate(
    ephemeris: CoupledEphemeris,
    start_utc: str,
    end_utc: str,
    *,
    step_seconds: float = 600.0,
    chunk_days: float = 366.0,
    overlap_days: float = 2.0,
) -> list[dict[str, Any]]:
    context = ephemeris.context
    start_tt = float(context.time_utc(start_utc).tt)
    end_tt = float(context.time_utc(end_utc).tt)
    events: list[dict[str, Any]] = []
    chunk_start = start_tt
    while chunk_start < end_tt:
        chunk_end = min(end_tt, chunk_start + chunk_days)
        search_start = max(start_tt, chunk_start - overlap_days)
        search_end = min(end_tt, chunk_end + overlap_days)
        found = find_lunar_eclipses(
            ephemeris,
            time_iso_utc(context.tt_jd(search_start)),
            time_iso_utc(context.tt_jd(search_end)),
            sample_step_seconds=step_seconds,
        )
        for event in found:
            row = asdict(event)
            row["domain"] = "lunar"
            row["body_label"] = BODY_LABEL[str(row["body"])]
            events.append(row)
        chunk_start = chunk_end
    return _deduplicate(events, "maximum_utc")


def inclination_series(
    ephemeris: CoupledEphemeris,
    start_utc: str,
    end_utc: str,
    *,
    step_days: float = 15.0,
) -> pd.DataFrame:
    start_tt = float(ephemeris.context.time_utc(start_utc).tt)
    end_tt = float(ephemeris.context.time_utc(end_utc).tt)
    times = np.arange(start_tt, end_tt + step_days, step_days)
    rotation = _rotation_x(-np.radians(OBLIQUITY_J2000_DEG))
    rows: list[dict[str, Any]] = []
    for body in ("real_moon", "second_moon"):
        position = ephemeris.relative(body, times)
        velocity = ephemeris.velocity(body, times) - ephemeris.velocity("earth", times)
        position = (rotation @ position.T).T
        velocity = (rotation @ velocity.T).T
        momentum = np.cross(position, velocity)
        inclination = np.degrees(
            np.arccos(
                np.clip(
                    momentum[:, 2] / np.linalg.norm(momentum, axis=1),
                    -1.0,
                    1.0,
                )
            )
        )
        for tt_jd, value in zip(times, inclination, strict=True):
            rows.append(
                {
                    "time_utc": time_iso_utc(ephemeris.time(float(tt_jd))),
                    "body": body,
                    "inclination_deg": float(value),
                }
            )
    return pd.DataFrame(rows)


def _refine_notable_solar(
    ephemeris: CoupledEphemeris,
    events: list[dict[str, Any]],
    *,
    per_body: int = 6,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for body in ("real_moon", "second_moon"):
        central = [
            event
            for event in events
            if event["body"] == body and event["eclipse_type"] in {"total", "hybrid"}
        ]
        central.sort(key=lambda event: float(event["core_radius_km"]), reverse=True)
        selected.extend(central[:per_body])
    for event in selected:
        maximum_tt = float(ephemeris.context.time_utc(str(event["maximum_utc"])).tt)
        local = solve_coupled_local(
            ephemeris,
            str(event["body"]),
            maximum_tt,
            float(event["latitude_deg"]),
            float(event["longitude_deg"]),
        )
        event["central_duration_s"] = local.central_duration_s
        for key, value in asdict(local).items():
            event[f"local_{key}"] = value
    return sorted(
        selected,
        key=lambda event: float(event.get("central_duration_s") or 0.0),
        reverse=True,
    )


def _annual_counts(
    solar_events: list[dict[str, Any]],
    lunar_events: list[dict[str, Any]],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for body in ("real_moon", "second_moon"):
            solar = [
                event
                for event in solar_events
                if event["body"] == body and _iso_datetime(str(event["maximum_utc"])).year == year
            ]
            lunar = [
                event
                for event in lunar_events
                if event["body"] == body and _iso_datetime(str(event["maximum_utc"])).year == year
            ]
            rows.append(
                {
                    "year": year,
                    "body": body,
                    "solar_all": len(solar),
                    "solar_total": sum(
                        event["eclipse_type"] in {"total", "hybrid"} for event in solar
                    ),
                    "lunar_all": len(lunar),
                    "lunar_total": sum(event["eclipse_type"] == "total" for event in lunar),
                }
            )
    return pd.DataFrame(rows)


def _correlations(counts: pd.DataFrame, inclinations: pd.DataFrame) -> dict[str, Any]:
    inclinations = inclinations.copy()
    inclinations["year"] = inclinations["time_utc"].map(lambda value: _iso_datetime(value).year)
    annual_i = inclinations.groupby(["year", "body"], as_index=False)["inclination_deg"].mean()
    merged = counts.merge(annual_i, on=["year", "body"], how="left")
    results: dict[str, Any] = {}
    for body in ("real_moon", "second_moon"):
        subset = merged[merged["body"] == body].copy()
        if len(subset) > 2:
            subset = subset.iloc[1:-1]
        for column in ("solar_all", "lunar_all"):
            smooth = subset[column].rolling(3, center=True, min_periods=1).mean()
            if len(subset) < 3:
                value = None
            else:
                statistic = float(
                    spearmanr(subset["inclination_deg"], smooth).statistic
                )
                value = statistic if np.isfinite(statistic) else None
            results[f"{body}_{column}_vs_inclination_spearman"] = value
    return results


def _refine_temporal_pairs(
    ephemeris: CoupledEphemeris,
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure track proximity and a shared-site seed for temporal solar pairs."""

    geod = Geod(ellps="WGS84")
    refined: list[dict[str, Any]] = []
    for pair in pairs:
        real_tt = float(
            ephemeris.context.time_utc(str(pair["real_maximum_utc"])).tt
        )
        second_tt = float(
            ephemeris.context.time_utc(str(pair["second_maximum_utc"])).tt
        )
        real_track = generate_coupled_track(
            ephemeris, "real_moon", real_tt, step_seconds=20.0
        )
        second_track = generate_coupled_track(
            ephemeris, "second_moon", second_tt, step_seconds=20.0
        )
        distance, real_index, second_index = minimum_track_distance_km(
            real_track, second_track
        )
        real_latitude = float(real_track["latitude_deg"][real_index])
        real_longitude = float(real_track["longitude_deg"][real_index])
        second_latitude = float(second_track["latitude_deg"][second_index])
        second_longitude = float(second_track["longitude_deg"][second_index])
        azimuth, _, distance_m = geod.inv(
            real_longitude,
            real_latitude,
            second_longitude,
            second_latitude,
        )
        common_longitude, common_latitude, _ = geod.fwd(
            real_longitude,
            real_latitude,
            azimuth,
            distance_m / 2.0,
        )
        real_local = solve_coupled_local(
            ephemeris,
            "real_moon",
            real_tt,
            common_latitude,
            common_longitude,
        )
        second_local = solve_coupled_local(
            ephemeris,
            "second_moon",
            second_tt,
            common_latitude,
            common_longitude,
        )
        local_separation = abs(
            float(ephemeris.context.time_utc(real_local.maximum_utc).tt)
            - float(ephemeris.context.time_utc(second_local.maximum_utc).tt)
        ) * 24.0
        both_visible_same_location = (
            real_local.magnitude > 0.0
            and second_local.magnitude > 0.0
            and real_local.solar_altitude_deg > 0.0
            and second_local.solar_altitude_deg > 0.0
            and local_separation <= 12.0
        )
        at_least_one_total = (
            real_local.eclipse_type == "total"
            or second_local.eclipse_type == "total"
        )
        same_location = both_visible_same_location and at_least_one_total
        at_least_one_global_total = (
            pair["real_type"] in {"total", "hybrid"}
            or pair["second_type"] in {"total", "hybrid"}
        )
        regional_500km = distance <= 500.0 and at_least_one_global_total
        result = {
            **pair,
            "track_distance_km": float(distance),
            "real_track_kind": (
                "maximum-eclipse path"
                if pair["real_type"] == "partial"
                else "central line"
            ),
            "second_track_kind": (
                "maximum-eclipse path"
                if pair["second_type"] == "partial"
                else "central line"
            ),
            "real_closest_track_point": {
                "utc": time_iso_utc(
                    ephemeris.time(float(real_track["tt_jd"][real_index])), places=3
                ),
                "latitude_deg": real_latitude,
                "longitude_deg": real_longitude,
            },
            "second_closest_track_point": {
                "utc": time_iso_utc(
                    ephemeris.time(float(second_track["tt_jd"][second_index])), places=3
                ),
                "latitude_deg": second_latitude,
                "longitude_deg": second_longitude,
            },
            "best_common_latitude_deg": float(common_latitude),
            "best_common_longitude_deg": float(common_longitude),
            "local_maximum_separation_hours": float(local_separation),
            "both_visible_same_location": bool(both_visible_same_location),
            "at_least_one_total_at_common_location": bool(at_least_one_total),
            "definition_a_same_location": bool(same_location),
            "definition_b_500km": bool(regional_500km),
            "both_total_at_common_location": (
                real_local.eclipse_type == "total"
                and second_local.eclipse_type == "total"
            ),
            "thresholds": {
                "same_location_12h": bool(same_location),
                "same_location_6h": bool(same_location and local_separation <= 6.0),
                "same_location_3h": bool(same_location and local_separation <= 3.0),
                "same_location_1h": bool(same_location and local_separation <= 1.0),
                "track_1000km_12h": bool(
                    distance <= 1_000.0 and at_least_one_global_total
                ),
                "track_500km_12h": bool(regional_500km),
                "track_100km_12h": bool(
                    distance <= 100.0 and at_least_one_global_total
                ),
            },
            "real_local": asdict(real_local),
            "second_local": asdict(second_local),
        }
        if same_location:
            result["status"] = "same-location chained eclipse"
        elif both_visible_same_location:
            result["status"] = (
                "same-location partial-only pair; fails the at-least-one-total requirement"
            )
        else:
            result["status"] = (
                "temporal-only pair; no eclipse from both moons at the shared-site seed"
            )
        refined.append(result)
    return refined


def _plot_climate(
    inclinations: pd.DataFrame,
    counts: pd.DataFrame,
    solar_events: list[dict[str, Any]],
    lunar_events: list[dict[str, Any]],
    notable_solar: list[dict[str, Any]],
    notable_lunar: list[dict[str, Any]],
    output_path: Path,
    *,
    model_description: str,
    model_limitations: str,
) -> None:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titlecolor": INK,
        }
    ):
        figure = plt.figure(figsize=(15.2, 10.8), facecolor="white")
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=(1.15, 0.9, 1.0),
            left=0.075,
            right=0.98,
            bottom=0.085,
            top=0.86,
            hspace=0.34,
            wspace=0.15,
        )
        inclination_ax = figure.add_subplot(grid[0, :])
        raster_ax = figure.add_subplot(grid[1, :], sharex=inclination_ax)
        real_ax = figure.add_subplot(grid[2, 0])
        second_ax = figure.add_subplot(grid[2, 1], sharey=real_ax)

        for body, color, label, linestyle in (
            ("real_moon", REAL_BLUE, "Real Moon", "-"),
            ("second_moon", SECOND_ORANGE, "Second moon", (0, (6, 3))),
        ):
            subset = inclinations[inclinations["body"] == body].copy()
            dates = subset["time_utc"].map(_iso_datetime)
            values = subset["inclination_deg"].to_numpy()
            smooth = pd.Series(values).rolling(13, center=True, min_periods=1).median()
            inclination_ax.plot(dates, values, color=color, alpha=0.18, linewidth=0.8)
            inclination_ax.plot(
                dates,
                smooth,
                color=color,
                linewidth=2.4,
                linestyle=linestyle,
                label=label,
            )
        inclination_ax.set_ylabel("Inclination to ecliptic (°)")
        inclination_ax.set_title(
            "The orbital planes exchange tilt on a roughly 30-year cycle",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=10,
        )
        inclination_ax.legend(frameon=False, ncol=2, loc="upper left")

        lanes = {
            ("solar", "real_moon"): 3.0,
            ("solar", "second_moon"): 2.0,
            ("lunar", "real_moon"): 1.0,
            ("lunar", "second_moon"): 0.0,
        }
        lane_labels = [
            "Real Moon · solar",
            "Second moon · solar",
            "Real Moon · lunar",
            "Second moon · lunar",
        ]
        all_events = solar_events + lunar_events
        notable_keys = {
            (event["domain"], event["body"], event["maximum_utc"])
            for event in notable_solar + notable_lunar
        }
        for event in all_events:
            lane = lanes[(str(event["domain"]), str(event["body"]))]
            date = _iso_datetime(str(event["maximum_utc"]))
            eclipse_type = str(event["eclipse_type"])
            color = REAL_BLUE if event["body"] == "real_moon" else SECOND_ORANGE
            if eclipse_type in {"total", "hybrid"}:
                marker, size, alpha, face = "o", 18.0, 0.82, color
            elif eclipse_type == "annular":
                marker, size, alpha, face = "o", 17.0, 0.75, "white"
            else:
                marker, size, alpha, face = "|", 22.0, 0.28, color
            scatter_style: dict[str, Any]
            if marker == "|":
                scatter_style = {"color": color, "linewidth": 0.75}
            else:
                scatter_style = {
                    "facecolor": face,
                    "edgecolor": color,
                    "linewidth": 0.75,
                }
            raster_ax.scatter(
                [date],
                [lane],
                marker=marker,
                s=size,
                alpha=alpha,
                zorder=3,
                **scatter_style,
            )
            key = (event["domain"], event["body"], event["maximum_utc"])
            if key in notable_keys:
                raster_ax.scatter(
                    [date], [lane], marker="*", s=82, color=color, edgecolor="white", linewidth=0.8, zorder=5
                )
        raster_ax.set_yticks((3.0, 2.0, 1.0, 0.0), labels=lane_labels)
        raster_ax.set_ylim(-0.55, 3.55)
        raster_ax.set_title(
            "Every eclipse · stars mark the standout total events",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=10,
        )

        for axis, body, color, title in (
            (real_ax, "real_moon", REAL_BLUE, "Real-Moon eclipse years"),
            (second_ax, "second_moon", SECOND_ORANGE, "Second-moon eclipse years"),
        ):
            subset = counts[counts["body"] == body]
            years = subset["year"].to_numpy()
            axis.plot(years, subset["solar_all"], color=color, linewidth=2.1, label="Solar eclipses")
            axis.plot(
                years,
                subset["lunar_all"],
                color=color,
                linewidth=1.8,
                linestyle=(0, (5, 3)),
                alpha=0.78,
                label="Lunar eclipses",
            )
            axis.fill_between(years, 0, subset["solar_total"], color=color, alpha=0.12, label="Solar totals")
            axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=9)
            axis.set_xlabel("Calendar year")
            axis.set_ylabel("Events per year" if body == "real_moon" else "")
            axis.legend(frameon=False, fontsize=8.7, ncol=2, loc="upper left")
            axis.set_xlim(years.min(), years.max())

        for axis in (inclination_ax, raster_ax, real_ax, second_ax):
            axis.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["bottom", "left"]].set_color(GRID)
        inclination_ax.xaxis.set_major_locator(mdates.YearLocator(5))
        inclination_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        raster_ax.xaxis.set_major_locator(mdates.YearLocator(5))
        raster_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(inclination_ax.get_xticklabels(), visible=False)

        figure.text(
            0.075,
            0.965,
            "Thirty years of eclipses in the two-moon system",
            fontsize=26,
            fontweight="bold",
            ha="left",
            va="top",
            color=INK,
        )
        figure.text(
            0.075,
            0.923,
            model_description,
            fontsize=11.8,
            ha="left",
            va="top",
            color=MUTED,
        )
        figure.text(
            0.075,
            0.026,
            "Faint inclination curves are 15-day osculating samples; heavy curves are 180-day medians. "
            "Counts include partial and penumbral events. "
            + model_limitations,
            fontsize=8.9,
            color=MUTED,
            ha="left",
            va="bottom",
        )
        figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
        figure.savefig(output_path.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
        plt.close(figure)


def run_eclipse_climate(
    *,
    start_utc: str = "2026-07-10T00:00:00Z",
    end_utc: str = "2056-07-10T00:00:00Z",
    config_path: str | Path = "config/optimized_system.yaml",
    output_dir: str | Path | None = None,
    trajectory_step_seconds: float = 3_600.0,
    detector_step_seconds: float = 600.0,
    dynamics_model: str = "baseline",
) -> dict[str, Any]:
    context = load_ephemeris("data/ephemeris")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    if dynamics_model == "baseline":
        ephemeris = CoupledEphemeris(
            context,
            elements,
            end_utc,
            sample_step_seconds=trajectory_step_seconds,
        )
        default_output_dir = "outputs/coupled/eclipse_climate_30y"
        model_description = (
            "Fully coupled Newtonian Sun–Earth–Moon–second-moon trajectory · "
            "10 Jul 2026–10 Jul 2056"
        )
        model_limitations = (
            "Long-range geography is conditional because tides, major planets, "
            "J2, and alternate-Earth rotation are omitted."
        )
    elif dynamics_model == "enhanced":
        from .enhanced_ephemeris import EnhancedEphemeris

        ephemeris = EnhancedEphemeris(
            context,
            elements,
            end_utc,
            sample_step_seconds=trajectory_step_seconds,
        )
        default_output_dir = "outputs/coupled/eclipse_climate_30y_enhanced"
        model_description = (
            "Enhanced coupled planets + Earth J2 + solar 1PN + tidal-spin trajectory · "
            "10 Jul 2026–10 Jul 2056"
        )
        model_limitations = (
            "Late ground tracks remain conditional because tides and terrestrial "
            "orientation use bounded approximations rather than a fitted alternate ephemeris."
        )
    else:
        raise ValueError("dynamics_model must be 'baseline' or 'enhanced'")
    solar_events = scan_solar_climate(
        ephemeris,
        start_utc,
        end_utc,
        step_seconds=detector_step_seconds,
    )
    lunar_events = scan_lunar_climate(
        ephemeris,
        start_utc,
        end_utc,
        step_seconds=detector_step_seconds,
    )
    inclinations = inclination_series(ephemeris, start_utc, end_utc)
    start_year = _iso_datetime(start_utc).year
    end_year = _iso_datetime(end_utc).year
    counts = _annual_counts(solar_events, lunar_events, start_year, end_year)
    notable_solar = _refine_notable_solar(ephemeris, solar_events)
    notable_lunar = sorted(
        [event for event in lunar_events if event["eclipse_type"] == "total"],
        key=lambda event: float(event["totality_duration_s"]),
        reverse=True,
    )
    lunar_by_body: list[dict[str, Any]] = []
    for body in ("real_moon", "second_moon"):
        lunar_by_body.extend(
            [event for event in notable_lunar if event["body"] == body][:6]
        )
    notable_lunar = sorted(
        lunar_by_body,
        key=lambda event: float(event["totality_duration_s"]),
        reverse=True,
    )
    temporal_pairs: list[dict[str, Any]] = []
    real_solar = [event for event in solar_events if event["body"] == "real_moon"]
    second_solar = [event for event in solar_events if event["body"] == "second_moon"]
    for real in real_solar:
        real_time = _iso_datetime(str(real["maximum_utc"]))
        for second in second_solar:
            delta_hours = abs(
                (_iso_datetime(str(second["maximum_utc"])) - real_time).total_seconds()
            ) / 3_600.0
            if delta_hours <= 12.0:
                temporal_pairs.append(
                    {
                        "real_maximum_utc": real["maximum_utc"],
                        "second_maximum_utc": second["maximum_utc"],
                        "separation_hours": delta_hours,
                        "real_type": real["eclipse_type"],
                        "second_type": second["eclipse_type"],
                    }
                )
    temporal_pairs = _refine_temporal_pairs(ephemeris, temporal_pairs)
    summary = {
        "solar_event_counts": {
            body: {
                kind: sum(
                    event["body"] == body and event["eclipse_type"] == kind
                    for event in solar_events
                )
                for kind in ("total", "hybrid", "annular", "partial")
            }
            for body in ("real_moon", "second_moon")
        },
        "lunar_event_counts": {
            body: {
                kind: sum(
                    event["body"] == body and event["eclipse_type"] == kind
                    for event in lunar_events
                )
                for kind in ("total", "partial", "penumbral")
            }
            for body in ("real_moon", "second_moon")
        },
        "inclination_ranges_deg": {
            body: {
                "minimum": float(
                    inclinations.loc[inclinations["body"] == body, "inclination_deg"].min()
                ),
                "maximum": float(
                    inclinations.loc[inclinations["body"] == body, "inclination_deg"].max()
                ),
            }
            for body in ("real_moon", "second_moon")
        },
        "correlations": _correlations(counts, inclinations),
        "temporal_solar_pair_count_within_12h": len(temporal_pairs),
        "definition_a_same_location_pair_count": sum(
            pair["definition_a_same_location"] for pair in temporal_pairs
        ),
        "definition_b_500km_pair_count": sum(
            pair["definition_b_500km"] for pair in temporal_pairs
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": f"30-year coupled {dynamics_model} eclipse climate",
        "dynamics_model": dynamics_model,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "trajectory": ephemeris.metadata,
        "summary": summary,
        "solar_events": solar_events,
        "lunar_events": lunar_events,
        "notable_solar_events": notable_solar,
        "notable_lunar_events": notable_lunar,
        "temporal_solar_pairs_within_12h": temporal_pairs,
        "limitations": ephemeris.metadata.get(
            "limitations",
            [
                "DE440s supplies epoch states only.",
                "Newtonian point masses omit tides, major planets, Earth J2, relativity, and figure terms.",
                "Earth rotation remains prescribed from the real-Earth Skyfield timescale.",
                "Lunar shadows use geometric simultaneous Sun positions and no atmospheric umbra enlargement.",
                "Grazing classifications are more cadence-sensitive than central and total eclipses.",
            ],
        ),
    }
    output = Path(default_output_dir if output_dir is None else output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "climate.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(solar_events + lunar_events).to_csv(output / "events.csv", index=False)
    pd.DataFrame(notable_solar + notable_lunar).to_csv(
        output / "notable_events.csv", index=False
    )
    pd.DataFrame(temporal_pairs).to_csv(output / "temporal_solar_pairs.csv", index=False)
    counts.to_csv(output / "annual_counts.csv", index=False)
    inclinations.to_csv(output / "inclinations.csv", index=False)
    _plot_climate(
        inclinations,
        counts,
        solar_events,
        lunar_events,
        notable_solar,
        notable_lunar,
        output / "eclipse_climate.png",
        model_description=model_description,
        model_limitations=model_limitations,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-07-10T00:00:00Z")
    parser.add_argument("--end", default="2056-07-10T00:00:00Z")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--trajectory-step-seconds", type=float, default=3_600.0)
    parser.add_argument("--detector-step-seconds", type=float, default=600.0)
    parser.add_argument(
        "--dynamics-model",
        choices=("baseline", "enhanced"),
        default="baseline",
    )
    args = parser.parse_args(argv)
    payload = run_eclipse_climate(
        start_utc=args.start,
        end_utc=args.end,
        config_path=args.config,
        output_dir=args.output_dir,
        trajectory_step_seconds=args.trajectory_step_seconds,
        detector_step_seconds=args.detector_step_seconds,
        dynamics_model=args.dynamics_model,
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
