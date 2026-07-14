"""Compare baseline and enhanced thirty-year eclipse-climate catalogs.

Events are matched independently inside each ``(domain, body)`` group by a
globally minimum sum of absolute maximum-time differences, subject to a hard
time window.  This makes the assignment one-to-one: a newly shifted event can
never be claimed by two baseline events.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
from pyproj import Geod
from scipy.optimize import linear_sum_assignment


REAL_BLUE = "#24649A"
SECOND_ORANGE = "#D47424"
INK = "#102039"
MUTED = "#68778F"
GRID = "#DCE4EC"
TYPE_CHANGE_RED = "#B5483A"
WGS84_GEOD = Geod(ellps="WGS84")
EVENT_DOMAINS = ("solar", "lunar")
BODY_ORDER = ("real_moon", "second_moon")
CHAIN_FLAGS = (
    "definition_a_same_location",
    "definition_b_500km",
    "both_total_at_common_location",
)


@dataclass(frozen=True, slots=True)
class EventReference:
    """One catalog event plus its normalized matching fields."""

    domain: str
    body: str
    catalog_index: int
    maximum: datetime
    event: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PairReference:
    """One two-moon temporal solar pair."""

    catalog_index: int
    real_maximum: datetime
    second_maximum: datetime
    pair: Mapping[str, Any]


def load_climate(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one climate JSON catalog."""

    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Climate catalog must contain a JSON object: {source}")
    for key in ("solar_events", "lunar_events"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Climate catalog is missing list field {key!r}: {source}")
    return payload


def compare_catalogs(
    baseline: Mapping[str, Any],
    enhanced: Mapping[str, Any],
    *,
    max_match_days: float = 10.0,
    pair_match_days: float | None = None,
) -> dict[str, Any]:
    """Return a strict-JSON-ready comparison of two climate catalogs."""

    if not math.isfinite(max_match_days) or max_match_days <= 0.0:
        raise ValueError("max_match_days must be finite and positive")
    pair_days = max_match_days if pair_match_days is None else pair_match_days
    if not math.isfinite(pair_days) or pair_days <= 0.0:
        raise ValueError("pair_match_days must be finite and positive")

    baseline_events = _event_references(baseline)
    enhanced_events = _event_references(enhanced)
    matched, removed, added = match_events(
        baseline_events,
        enhanced_events,
        max_match_days=max_match_days,
    )
    pair_changes = compare_temporal_pairs(
        baseline.get("temporal_solar_pairs_within_12h", []),
        enhanced.get("temporal_solar_pairs_within_12h", []),
        max_match_days=pair_days,
    )
    type_changes = [row for row in matched if row["type_changed"]]
    coordinate_shifts = [
        float(row["global_maximum_point_shift_km"])
        for row in matched
        if row["global_maximum_point_shift_km"] is not None
    ]
    absolute_time_shifts = [float(row["absolute_time_shift_seconds"]) for row in matched]
    baseline_period = {
        "start_utc": baseline.get("start_utc"),
        "end_utc": baseline.get("end_utc"),
    }
    enhanced_period = {
        "start_utc": enhanced.get("start_utc"),
        "end_utc": enhanced.get("end_utc"),
    }
    limitations = [
        (
            "Event identity is inferred from a globally optimal one-to-one nearest-time "
            f"assignment within {max_match_days:g} days, separately by domain and moon."
        ),
        (
            "Solar geographic displacement is the WGS84 geodesic between the two global-maximum "
            "points; it is not a full central-track or eclipse-boundary distance."
        ),
        "Lunar eclipses have no terrestrial ground-track coordinate, so their displacement is null.",
        (
            "Pair changes compare only the catalogs' saved solar-pair lists; a removed pair may "
            "still have both component eclipses outside the twelve-hour pair threshold."
        ),
    ]
    if baseline_period != enhanced_period:
        limitations.append(
            "The two catalogs cover different start or end times, so edge event-count deltas are confounded."
        )

    return {
        "schema_version": "1.0",
        "comparison_mode": "baseline versus enhanced eclipse climate",
        "matching": {
            "method": "minimum-total-absolute-time one-to-one assignment by domain and body",
            "max_match_days": max_match_days,
            "pair_match_days": pair_days,
        },
        "baseline": _catalog_descriptor(baseline),
        "enhanced": _catalog_descriptor(enhanced),
        "periods": {"baseline": baseline_period, "enhanced": enhanced_period},
        "summary": {
            "baseline_event_count": len(baseline_events),
            "enhanced_event_count": len(enhanced_events),
            "event_count_delta": len(enhanced_events) - len(baseline_events),
            "matched_event_count": len(matched),
            "added_event_count": len(added),
            "removed_event_count": len(removed),
            "type_change_count": len(type_changes),
            "median_absolute_time_shift_seconds": _percentile(absolute_time_shifts, 50.0),
            "maximum_absolute_time_shift_seconds": max(absolute_time_shifts, default=None),
            "median_global_maximum_point_shift_km": _percentile(coordinate_shifts, 50.0),
            "maximum_global_maximum_point_shift_km": max(coordinate_shifts, default=None),
        },
        "event_counts": _event_count_comparison(
            baseline_events,
            enhanced_events,
            matched,
            removed,
            added,
        ),
        "matched_events": matched,
        "added_events": [_event_summary(reference) for reference in added],
        "removed_events": [_event_summary(reference) for reference in removed],
        "type_changes": type_changes,
        "pair_and_chain_changes": pair_changes,
        "limitations": limitations,
    }


def match_events(
    baseline: Sequence[EventReference],
    enhanced: Sequence[EventReference],
    *,
    max_match_days: float,
) -> tuple[list[dict[str, Any]], list[EventReference], list[EventReference]]:
    """Match events one-to-one within domain/body groups."""

    max_seconds = max_match_days * 86_400.0
    matched: list[dict[str, Any]] = []
    removed: list[EventReference] = []
    added: list[EventReference] = []
    groups = sorted(
        {(event.domain, event.body) for event in (*baseline, *enhanced)},
        key=_group_sort_key,
    )
    for domain, body in groups:
        baseline_group = sorted(
            (event for event in baseline if (event.domain, event.body) == (domain, body)),
            key=lambda event: event.maximum,
        )
        enhanced_group = sorted(
            (event for event in enhanced if (event.domain, event.body) == (domain, body)),
            key=lambda event: event.maximum,
        )
        assignments, unmatched_baseline, unmatched_enhanced = _time_assignment(
            [event.maximum for event in baseline_group],
            [event.maximum for event in enhanced_group],
            max_seconds=max_seconds,
        )
        matched.extend(
            _matched_event_row(baseline_group[left], enhanced_group[right])
            for left, right in assignments
        )
        removed.extend(baseline_group[index] for index in unmatched_baseline)
        added.extend(enhanced_group[index] for index in unmatched_enhanced)

    matched.sort(key=lambda row: (str(row["baseline_maximum_utc"]), str(row["domain"])))
    removed.sort(key=lambda event: event.maximum)
    added.sort(key=lambda event: event.maximum)
    return matched, removed, added


def compare_temporal_pairs(
    baseline_pairs: Any,
    enhanced_pairs: Any,
    *,
    max_match_days: float,
) -> dict[str, Any]:
    """Compare saved rapid solar-pair and chained-eclipse classifications."""

    if not isinstance(baseline_pairs, list) or not isinstance(enhanced_pairs, list):
        raise ValueError("temporal_solar_pairs_within_12h must be a list in both catalogs")
    baseline = _pair_references(baseline_pairs)
    enhanced = _pair_references(enhanced_pairs)
    max_seconds = max_match_days * 86_400.0
    assignments, removed_indices, added_indices = _pair_assignment(
        baseline,
        enhanced,
        max_seconds=max_seconds,
    )
    matched = [_matched_pair_row(baseline[left], enhanced[right]) for left, right in assignments]
    removed = [_pair_summary(baseline[index]) for index in removed_indices]
    added = [_pair_summary(enhanced[index]) for index in added_indices]
    changed = [
        row
        for row in matched
        if row["separation_changed"]
        or row["status_changed"]
        or any(row[f"{flag}_changed"] for flag in CHAIN_FLAGS)
    ]
    baseline_flag_counts = {
        flag: sum(bool(pair.pair.get(flag, False)) for pair in baseline) for flag in CHAIN_FLAGS
    }
    enhanced_flag_counts = {
        flag: sum(bool(pair.pair.get(flag, False)) for pair in enhanced) for flag in CHAIN_FLAGS
    }
    return {
        "baseline_pair_count": len(baseline),
        "enhanced_pair_count": len(enhanced),
        "pair_count_delta": len(enhanced) - len(baseline),
        "matched_pair_count": len(matched),
        "added_pair_count": len(added),
        "removed_pair_count": len(removed),
        "changed_matched_pair_count": len(changed),
        "chain_flag_counts": {
            flag: {
                "baseline": baseline_flag_counts[flag],
                "enhanced": enhanced_flag_counts[flag],
                "delta": enhanced_flag_counts[flag] - baseline_flag_counts[flag],
            }
            for flag in CHAIN_FLAGS
        },
        "matched_pairs": matched,
        "changed_matched_pairs": changed,
        "added_pairs": added,
        "removed_pairs": removed,
    }


def write_comparison(
    baseline_path: str | Path,
    enhanced_path: str | Path,
    output_dir: str | Path,
    *,
    max_match_days: float = 10.0,
    pair_match_days: float | None = None,
) -> dict[str, Any]:
    """Compare two files and write JSON, matched CSV, and the static figure."""

    baseline_source = Path(baseline_path).resolve()
    enhanced_source = Path(enhanced_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison = compare_catalogs(
        load_climate(baseline_source),
        load_climate(enhanced_source),
        max_match_days=max_match_days,
        pair_match_days=pair_match_days,
    )
    comparison["sources"] = {
        "baseline_climate_json": str(baseline_source),
        "enhanced_climate_json": str(enhanced_source),
    }
    json_path = output / "comparison.json"
    csv_path = output / "matched_events.csv"
    figure_path = output / "comparison.png"
    json_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    _write_matched_csv(csv_path, comparison["matched_events"])
    plot_comparison(comparison, figure_path)
    return comparison


def plot_comparison(comparison: Mapping[str, Any], output_path: str | Path) -> Figure:
    """Plot event-time and solar maximum-point shifts against calendar year."""

    rows = comparison.get("matched_events", [])
    if not isinstance(rows, list):
        raise ValueError("comparison matched_events must be a list")
    figure = Figure(figsize=(13.2, 8.4), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.10})
    top, bottom = axes
    styles = {
        ("solar", "real_moon"): (REAL_BLUE, "o", "Solar · real Moon"),
        ("solar", "second_moon"): (SECOND_ORANGE, "o", "Solar · second moon"),
        ("lunar", "real_moon"): (REAL_BLUE, "^", "Lunar · real Moon"),
        ("lunar", "second_moon"): (SECOND_ORANGE, "^", "Lunar · second moon"),
    }
    max_abs_seconds = max(
        (abs(float(row["time_shift_seconds"])) for row in rows),
        default=0.0,
    )
    time_divisor, time_unit = _time_plot_scale(max_abs_seconds)
    for group, (color, marker, label) in styles.items():
        selected = [row for row in rows if (row.get("domain"), row.get("body")) == group]
        if not selected:
            continue
        dates = [_parse_utc(str(row["baseline_maximum_utc"])) for row in selected]
        shifts = [float(row["time_shift_seconds"]) / time_divisor for row in selected]
        changed = np.asarray([bool(row["type_changed"]) for row in selected])
        normal = ~changed
        if np.any(normal):
            top.scatter(
                np.asarray(dates, dtype=object)[normal],
                np.asarray(shifts)[normal],
                s=24,
                marker=marker,
                color=color,
                alpha=0.72,
                linewidths=0.0,
                label=label,
            )
        if np.any(changed):
            top.scatter(
                np.asarray(dates, dtype=object)[changed],
                np.asarray(shifts)[changed],
                s=45,
                marker=marker,
                facecolors="none",
                edgecolors=TYPE_CHANGE_RED,
                linewidths=1.4,
                label=f"{label} · type changed",
            )

    solar_rows = [
        row
        for row in rows
        if row.get("domain") == "solar"
        and row.get("global_maximum_point_shift_km") is not None
    ]
    for body, color, label in (
        ("real_moon", REAL_BLUE, "Real Moon"),
        ("second_moon", SECOND_ORANGE, "Second moon"),
    ):
        selected = [row for row in solar_rows if row.get("body") == body]
        if not selected:
            continue
        bottom.scatter(
            [_parse_utc(str(row["baseline_maximum_utc"])) for row in selected],
            [float(row["global_maximum_point_shift_km"]) for row in selected],
            s=28,
            color=color,
            alpha=0.76,
            linewidths=0.0,
            label=label,
        )

    top.axhline(0.0, color=INK, linewidth=0.9, alpha=0.72)
    top.set_ylabel(f"Enhanced − baseline time ({time_unit})", color=INK)
    bottom.set_ylabel("Global-maximum point shift (km)", color=INK)
    bottom.set_xlabel("Baseline eclipse date", color=INK)
    title = "How enhanced dynamics moves the thirty-year eclipse climate"
    baseline_label = str(comparison.get("baseline", {}).get("mode", "Baseline"))
    enhanced_label = str(comparison.get("enhanced", {}).get("mode", "Enhanced"))
    summary = comparison.get("summary", {})
    figure.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=0.975)
    figure.text(
        0.5,
        0.938,
        (
            f"{baseline_label} → {enhanced_label}  ·  "
            f"{summary.get('matched_event_count', 0)} matched, "
            f"{summary.get('added_event_count', 0)} added, "
            f"{summary.get('removed_event_count', 0)} removed"
        ),
        ha="center",
        va="top",
        color=MUTED,
        fontsize=10.2,
    )
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.78)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=MUTED)
    top.legend(loc="best", frameon=False, fontsize=8.5, ncols=2)
    bottom.legend(loc="best", frameon=False, fontsize=9)
    bottom.xaxis.set_major_locator(mdates.YearLocator(5))
    bottom.xaxis.set_minor_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if not rows:
        top.text(0.5, 0.5, "No matched events", transform=top.transAxes, ha="center", color=MUTED)
    if not solar_rows:
        bottom.text(
            0.5,
            0.5,
            "No matched solar maximum points",
            transform=bottom.transAxes,
            ha="center",
            color=MUTED,
        )
    figure.subplots_adjust(top=0.89, left=0.10, right=0.975, bottom=0.10)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190, facecolor="white")
    return figure


def _event_references(catalog: Mapping[str, Any]) -> list[EventReference]:
    references: list[EventReference] = []
    catalog_index = 0
    for domain in EVENT_DOMAINS:
        raw_events = catalog.get(f"{domain}_events", [])
        if not isinstance(raw_events, list):
            raise ValueError(f"{domain}_events must be a list")
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise ValueError(f"{domain}_events entries must be objects")
            body = str(event.get("body", ""))
            if not body:
                raise ValueError(f"{domain} event is missing body")
            references.append(
                EventReference(
                    domain=domain,
                    body=body,
                    catalog_index=catalog_index,
                    maximum=_parse_utc(str(event.get("maximum_utc", ""))),
                    event=event,
                )
            )
            catalog_index += 1
    return references


def _time_assignment(
    baseline_times: Sequence[datetime],
    enhanced_times: Sequence[datetime],
    *,
    max_seconds: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not baseline_times:
        return [], [], list(range(len(enhanced_times)))
    if not enhanced_times:
        return [], list(range(len(baseline_times))), []
    valid_cost = np.asarray(
        [
            [abs((enhanced - baseline).total_seconds()) for enhanced in enhanced_times]
            for baseline in baseline_times
        ],
        dtype=float,
    )
    assignment_size = min(len(baseline_times), len(enhanced_times))
    invalid_penalty = (max_seconds + 1.0) * (assignment_size + 1.0)
    optimization_cost = np.where(valid_cost <= max_seconds, valid_cost, invalid_penalty)
    # A microscopic deterministic tie breaker keeps results stable for exactly
    # symmetric synthetic or rounded catalogs without affecting physical costs.
    optimization_cost += (
        np.arange(len(baseline_times))[:, None] * len(enhanced_times)
        + np.arange(len(enhanced_times))[None, :]
    ) * np.finfo(float).eps
    row_indices, column_indices = linear_sum_assignment(optimization_cost)
    assignments = sorted(
        (
            (int(left), int(right))
            for left, right in zip(row_indices, column_indices, strict=True)
            if valid_cost[left, right] <= max_seconds
        ),
        key=lambda pair: pair[0],
    )
    used_left = {left for left, _ in assignments}
    used_right = {right for _, right in assignments}
    return (
        assignments,
        [index for index in range(len(baseline_times)) if index not in used_left],
        [index for index in range(len(enhanced_times)) if index not in used_right],
    )


def _matched_event_row(baseline: EventReference, enhanced: EventReference) -> dict[str, Any]:
    baseline_type = str(baseline.event.get("eclipse_type", "unknown"))
    enhanced_type = str(enhanced.event.get("eclipse_type", "unknown"))
    time_shift = (enhanced.maximum - baseline.maximum).total_seconds()
    baseline_latitude = _finite_number(baseline.event.get("latitude_deg"))
    baseline_longitude = _finite_number(baseline.event.get("longitude_deg"))
    enhanced_latitude = _finite_number(enhanced.event.get("latitude_deg"))
    enhanced_longitude = _finite_number(enhanced.event.get("longitude_deg"))
    coordinate_shift = _coordinate_shift_km(
        baseline_latitude,
        baseline_longitude,
        enhanced_latitude,
        enhanced_longitude,
    )
    return {
        "domain": baseline.domain,
        "body": baseline.body,
        "baseline_catalog_index": baseline.catalog_index,
        "enhanced_catalog_index": enhanced.catalog_index,
        "baseline_maximum_utc": _iso_utc(baseline.maximum),
        "enhanced_maximum_utc": _iso_utc(enhanced.maximum),
        "time_shift_seconds": time_shift,
        "absolute_time_shift_seconds": abs(time_shift),
        "baseline_type": baseline_type,
        "enhanced_type": enhanced_type,
        "type_changed": baseline_type != enhanced_type,
        "baseline_latitude_deg": baseline_latitude,
        "baseline_longitude_deg": baseline_longitude,
        "enhanced_latitude_deg": enhanced_latitude,
        "enhanced_longitude_deg": enhanced_longitude,
        "global_maximum_point_shift_km": coordinate_shift,
    }


def _event_summary(reference: EventReference) -> dict[str, Any]:
    return {
        "domain": reference.domain,
        "body": reference.body,
        "catalog_index": reference.catalog_index,
        "maximum_utc": _iso_utc(reference.maximum),
        "eclipse_type": str(reference.event.get("eclipse_type", "unknown")),
        "latitude_deg": _finite_number(reference.event.get("latitude_deg")),
        "longitude_deg": _finite_number(reference.event.get("longitude_deg")),
    }


def _event_count_comparison(
    baseline: Sequence[EventReference],
    enhanced: Sequence[EventReference],
    matched: Sequence[Mapping[str, Any]],
    removed: Sequence[EventReference],
    added: Sequence[EventReference],
) -> dict[str, Any]:
    baseline_group = Counter((event.domain, event.body) for event in baseline)
    enhanced_group = Counter((event.domain, event.body) for event in enhanced)
    matched_group = Counter((str(row["domain"]), str(row["body"])) for row in matched)
    removed_group = Counter((event.domain, event.body) for event in removed)
    added_group = Counter((event.domain, event.body) for event in added)
    group_keys = sorted(set(baseline_group) | set(enhanced_group), key=_group_sort_key)

    baseline_type = Counter(
        (event.domain, event.body, str(event.event.get("eclipse_type", "unknown")))
        for event in baseline
    )
    enhanced_type = Counter(
        (event.domain, event.body, str(event.event.get("eclipse_type", "unknown")))
        for event in enhanced
    )
    type_keys = sorted(set(baseline_type) | set(enhanced_type))
    return {
        "by_domain_and_body": [
            {
                "domain": domain,
                "body": body,
                "baseline": baseline_group[(domain, body)],
                "enhanced": enhanced_group[(domain, body)],
                "delta": enhanced_group[(domain, body)] - baseline_group[(domain, body)],
                "matched": matched_group[(domain, body)],
                "added": added_group[(domain, body)],
                "removed": removed_group[(domain, body)],
            }
            for domain, body in group_keys
        ],
        "by_domain_body_and_type": [
            {
                "domain": domain,
                "body": body,
                "eclipse_type": eclipse_type,
                "baseline": baseline_type[(domain, body, eclipse_type)],
                "enhanced": enhanced_type[(domain, body, eclipse_type)],
                "delta": (
                    enhanced_type[(domain, body, eclipse_type)]
                    - baseline_type[(domain, body, eclipse_type)]
                ),
            }
            for domain, body, eclipse_type in type_keys
        ],
    }


def _pair_references(pairs: Sequence[Mapping[str, Any]]) -> list[PairReference]:
    references = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError("temporal solar pair entries must be objects")
        references.append(
            PairReference(
                catalog_index=index,
                real_maximum=_parse_utc(str(pair.get("real_maximum_utc", ""))),
                second_maximum=_parse_utc(str(pair.get("second_maximum_utc", ""))),
                pair=pair,
            )
        )
    return references


def _pair_assignment(
    baseline: Sequence[PairReference],
    enhanced: Sequence[PairReference],
    *,
    max_seconds: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not baseline:
        return [], [], list(range(len(enhanced)))
    if not enhanced:
        return [], list(range(len(baseline))), []
    costs = np.full((len(baseline), len(enhanced)), np.inf, dtype=float)
    valid = np.zeros_like(costs, dtype=bool)
    for left, baseline_pair in enumerate(baseline):
        for right, enhanced_pair in enumerate(enhanced):
            real_shift = abs(
                (enhanced_pair.real_maximum - baseline_pair.real_maximum).total_seconds()
            )
            second_shift = abs(
                (enhanced_pair.second_maximum - baseline_pair.second_maximum).total_seconds()
            )
            valid[left, right] = real_shift <= max_seconds and second_shift <= max_seconds
            costs[left, right] = real_shift + second_shift
    assignment_size = min(len(baseline), len(enhanced))
    penalty = (2.0 * max_seconds + 1.0) * (assignment_size + 1.0)
    optimized = np.where(valid, costs, penalty)
    rows, columns = linear_sum_assignment(optimized)
    assignments = sorted(
        (
            (int(left), int(right))
            for left, right in zip(rows, columns, strict=True)
            if valid[left, right]
        ),
        key=lambda pair: pair[0],
    )
    used_left = {left for left, _ in assignments}
    used_right = {right for _, right in assignments}
    return (
        assignments,
        [index for index in range(len(baseline)) if index not in used_left],
        [index for index in range(len(enhanced)) if index not in used_right],
    )


def _matched_pair_row(baseline: PairReference, enhanced: PairReference) -> dict[str, Any]:
    baseline_separation = _finite_number(baseline.pair.get("separation_hours"))
    enhanced_separation = _finite_number(enhanced.pair.get("separation_hours"))
    row: dict[str, Any] = {
        "baseline_catalog_index": baseline.catalog_index,
        "enhanced_catalog_index": enhanced.catalog_index,
        "baseline_real_maximum_utc": _iso_utc(baseline.real_maximum),
        "enhanced_real_maximum_utc": _iso_utc(enhanced.real_maximum),
        "real_time_shift_seconds": (
            enhanced.real_maximum - baseline.real_maximum
        ).total_seconds(),
        "baseline_second_maximum_utc": _iso_utc(baseline.second_maximum),
        "enhanced_second_maximum_utc": _iso_utc(enhanced.second_maximum),
        "second_time_shift_seconds": (
            enhanced.second_maximum - baseline.second_maximum
        ).total_seconds(),
        "baseline_separation_hours": baseline_separation,
        "enhanced_separation_hours": enhanced_separation,
        "separation_shift_hours": (
            enhanced_separation - baseline_separation
            if baseline_separation is not None and enhanced_separation is not None
            else None
        ),
        "separation_changed": baseline_separation != enhanced_separation,
        "baseline_track_distance_km": _finite_number(baseline.pair.get("track_distance_km")),
        "enhanced_track_distance_km": _finite_number(enhanced.pair.get("track_distance_km")),
        "baseline_status": baseline.pair.get("status"),
        "enhanced_status": enhanced.pair.get("status"),
        "status_changed": baseline.pair.get("status") != enhanced.pair.get("status"),
    }
    for flag in CHAIN_FLAGS:
        baseline_value = bool(baseline.pair.get(flag, False))
        enhanced_value = bool(enhanced.pair.get(flag, False))
        row[f"baseline_{flag}"] = baseline_value
        row[f"enhanced_{flag}"] = enhanced_value
        row[f"{flag}_changed"] = baseline_value != enhanced_value
    return row


def _pair_summary(reference: PairReference) -> dict[str, Any]:
    return {
        "catalog_index": reference.catalog_index,
        "real_maximum_utc": _iso_utc(reference.real_maximum),
        "second_maximum_utc": _iso_utc(reference.second_maximum),
        "separation_hours": _finite_number(reference.pair.get("separation_hours")),
        "track_distance_km": _finite_number(reference.pair.get("track_distance_km")),
        "status": reference.pair.get("status"),
        **{flag: bool(reference.pair.get(flag, False)) for flag in CHAIN_FLAGS},
    }


def _catalog_descriptor(catalog: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = catalog.get("trajectory", {})
    force_model = trajectory.get("force_model") if isinstance(trajectory, Mapping) else None
    return {
        "mode": catalog.get("mode", "unspecified climate model"),
        "schema_version": catalog.get("schema_version"),
        "force_model": force_model,
    }


def _coordinate_shift_km(
    baseline_latitude: float | None,
    baseline_longitude: float | None,
    enhanced_latitude: float | None,
    enhanced_longitude: float | None,
) -> float | None:
    if None in (
        baseline_latitude,
        baseline_longitude,
        enhanced_latitude,
        enhanced_longitude,
    ):
        return None
    _, _, distance_m = WGS84_GEOD.inv(
        float(baseline_longitude),
        float(baseline_latitude),
        float(enhanced_longitude),
        float(enhanced_latitude),
    )
    return abs(float(distance_m)) / 1_000.0


def _write_matched_csv(path: Path, rows: Any) -> None:
    if not isinstance(rows, list):
        raise ValueError("matched event rows must be a list")
    fieldnames = [
        "domain",
        "body",
        "baseline_catalog_index",
        "enhanced_catalog_index",
        "baseline_maximum_utc",
        "enhanced_maximum_utc",
        "time_shift_seconds",
        "absolute_time_shift_seconds",
        "baseline_type",
        "enhanced_type",
        "type_changed",
        "baseline_latitude_deg",
        "baseline_longitude_deg",
        "enhanced_latitude_deg",
        "enhanced_longitude_deg",
        "global_maximum_point_shift_km",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _parse_utc(value: str) -> datetime:
    if not value:
        raise ValueError("event maximum time is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid event maximum UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _group_sort_key(group: tuple[str, str]) -> tuple[int, int, str, str]:
    domain, body = group
    return (
        EVENT_DOMAINS.index(domain) if domain in EVENT_DOMAINS else len(EVENT_DOMAINS),
        BODY_ORDER.index(body) if body in BODY_ORDER else len(BODY_ORDER),
        domain,
        body,
    )


def _time_plot_scale(max_abs_seconds: float) -> tuple[float, str]:
    if max_abs_seconds < 6.0 * 3_600.0:
        return 60.0, "minutes"
    if max_abs_seconds < 3.0 * 86_400.0:
        return 3_600.0, "hours"
    return 86_400.0, "days"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and enhanced eclipse-climate JSON catalogs."
    )
    parser.add_argument("baseline", type=Path, help="Baseline climate.json")
    parser.add_argument("enhanced", type=Path, help="Enhanced climate.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-match-days",
        type=float,
        default=10.0,
        help="Maximum event-time difference allowed for one-to-one matching (default: 10)",
    )
    parser.add_argument(
        "--pair-match-days",
        type=float,
        default=None,
        help="Maximum component-time difference for matching rapid solar pairs",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    comparison = write_comparison(
        arguments.baseline,
        arguments.enhanced,
        arguments.output_dir,
        max_match_days=arguments.max_match_days,
        pair_match_days=arguments.pair_match_days,
    )
    summary = comparison["summary"]
    pair_summary = comparison["pair_and_chain_changes"]
    print(
        json.dumps(
            {
                "matched": summary["matched_event_count"],
                "added": summary["added_event_count"],
                "removed": summary["removed_event_count"],
                "type_changes": summary["type_change_count"],
                "temporal_pair_delta": pair_summary["pair_count_delta"],
                "output_dir": str(arguments.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
