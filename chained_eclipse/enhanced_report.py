"""Generate the result-first report for the enhanced eclipse-climate run.

The generator consumes the saved enhanced ``climate.json`` and the companion
baseline comparison JSON.  It writes equivalent Markdown and portable HTML
reports beside the enhanced catalog.  Optional validation and convergence JSON
files are summarized when supplied; no scientific result is recomputed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BODY_LABELS = {"real_moon": "Real Moon", "second_moon": "Second moon"}
DOMAIN_LABELS = {"solar": "Solar", "lunar": "Lunar"}


@dataclass(frozen=True, slots=True)
class Artifact:
    label: str
    relative_path: str
    caption: str | None = None
    image: bool = False


@dataclass(slots=True)
class ReportView:
    title: str
    period: str
    headline: str
    summary_points: list[str]
    total_events: int
    temporal_pair_count: int
    double_totality_pair_count: int
    count_rows: list[list[str]]
    inclination_rows: list[list[str]]
    pair_rows: list[list[str]]
    rotation_rows: list[list[str]]
    comparison_rows: list[list[str]]
    count_comparison_rows: list[list[str]]
    force_items: list[str]
    assumptions: list[str]
    limitations: list[str]
    validation_rows: list[list[str]]
    validation_notes: list[str]
    convergence_notes: list[str]
    next_checks: list[str]
    figures: list[Artifact]
    artifacts: list[Artifact]


def generate_enhanced_report(
    enhanced_climate_path: str | Path = (
        "outputs/coupled/eclipse_climate_30y_enhanced/climate.json"
    ),
    comparison_path: str | Path = (
        "outputs/coupled/eclipse_climate_30y_enhanced/comparison/comparison.json"
    ),
    *,
    output_dir: str | Path | None = None,
    validation_path: str | Path | None = None,
    convergence_path: str | Path | Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Write ``report.md`` and standalone ``report.html`` for an enhanced run."""

    climate_source = Path(enhanced_climate_path).resolve()
    comparison_source = Path(comparison_path).resolve()
    climate = _load_object(climate_source, "enhanced climate")
    comparison = _load_object(comparison_source, "comparison")
    if str(climate.get("dynamics_model", "")).lower() != "enhanced":
        raise ValueError("enhanced climate JSON must declare dynamics_model='enhanced'")
    _require_mapping(climate, "summary", "enhanced climate")
    _require_mapping(climate, "trajectory", "enhanced climate")
    _require_mapping(comparison, "summary", "comparison")

    validation_source = Path(validation_path).resolve() if validation_path else None
    convergence_sources = _normalize_paths(convergence_path)
    validation = (
        _load_object(validation_source, "validation") if validation_source else None
    )
    convergences = [
        _load_object(source, "convergence") for source in convergence_sources
    ]
    output = climate_source.parent if output_dir is None else Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    view = _build_view(
        climate,
        comparison,
        output=output,
        climate_source=climate_source,
        comparison_source=comparison_source,
        validation=validation,
        validation_source=validation_source,
        convergences=convergences,
        convergence_sources=convergence_sources,
    )
    markdown_path = output / "report.md"
    html_path = output / "report.html"
    markdown_path.write_text(_render_markdown(view), encoding="utf-8")
    html_path.write_text(_render_html(view), encoding="utf-8")
    return {
        "schema_version": "1.0",
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "headline": view.headline,
        "total_events": view.total_events,
        "temporal_pair_count": view.temporal_pair_count,
        "double_totality_pair_count": view.double_totality_pair_count,
    }


def _build_view(
    climate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    output: Path,
    climate_source: Path,
    comparison_source: Path,
    validation: Mapping[str, Any] | None,
    validation_source: Path | None,
    convergences: Sequence[Mapping[str, Any]],
    convergence_sources: Sequence[Path],
) -> ReportView:
    summary = _mapping(climate.get("summary"))
    trajectory = _mapping(climate.get("trajectory"))
    pairs = _mapping_list(climate.get("temporal_solar_pairs_within_12h"))
    comparison_summary = _mapping(comparison.get("summary"))

    count_rows, total_events = _event_count_rows(summary)
    inclination_rows = _inclination_rows(summary)
    pair_rows = [_pair_row(pair) for pair in pairs]
    headline_pair = _headline_pair(pairs)
    headline = _headline(headline_pair) if headline_pair else (
        "No rapid two-moon solar-eclipse pair was retained in the enhanced catalog."
    )
    double_totality_count = sum(
        bool(pair.get("both_total_at_common_location")) for pair in pairs
    )
    rotation = _mapping(trajectory.get("alternate_earth_rotation"))
    rotation_rows = _rotation_rows(rotation)
    comparison_rows = _comparison_rows(comparison_summary, comparison)
    count_comparison_rows = _count_comparison_rows(comparison)
    force_items = _force_items(trajectory)
    assumptions = _assumptions(climate, trajectory, comparison)
    limitations = _limitations(climate, trajectory, comparison)
    validation_rows, validation_notes = _validation_summary(validation)
    convergence_notes = _convergence_summary(convergences, convergence_sources)
    figures, artifacts = _artifacts(
        output,
        climate_source,
        comparison_source,
        validation_source,
        convergence_sources,
    )

    same_location_count = int(summary.get("definition_a_same_location_pair_count", 0))
    regional_count = int(summary.get("definition_b_500km_pair_count", 0))
    event_delta = _integer(comparison_summary.get("event_count_delta"))
    max_shift = _finite(comparison_summary.get("maximum_global_maximum_point_shift_km"))
    rotation_lon = _finite(rotation.get("final_longitude_shift_deg"))
    summary_points = [
        (
            f"The enhanced run contains {total_events:,} eclipses: "
            f"{_domain_total(summary, 'solar'):,} solar and "
            f"{_domain_total(summary, 'lunar'):,} lunar."
        ),
        (
            f"It retains {len(pairs)} solar pairs within 12 hours. Same-location chains: "
            f"{same_location_count}; tracks within 500 km: {regional_count}; "
            f"same-location double totalities: {double_totality_count}."
        ),
        (
            f"Relative to the baseline catalog, the enhanced model changes the inventory by "
            f"{event_delta:+d} event(s), with {int(comparison_summary.get('type_change_count', 0))} "
            "matched classification change(s)."
        ),
    ]
    if max_shift is not None:
        summary_points.append(
            "Among matched solar events, the largest saved global-maximum displacement is "
            f"{max_shift:,.1f} km; this compares maximum points, not complete paths."
        )
    if rotation_lon is not None:
        summary_points.append(
            "The modeled second-moon contribution to terrestrial orientation reaches "
            f"{rotation_lon:+.6f}° of longitude by the end of the run."
        )

    next_checks = [
        (
            "Repeat the enhanced catalog at a finer detector and trajectory cadence before "
            "treating grazing classifications or late ground-track coordinates as converged."
            if not convergences
            else "Retain the supplied cadence comparison as a regression test whenever the force model changes."
        ),
        (
            "Vary the constant tidal lag and Earth-spin treatment around the 2055 partial-only "
            "pair; it is a useful late-epoch sensitivity target but does not satisfy either "
            "chained-eclipse definition."
        ),
        (
            "Extend the enhanced stability integration beyond 30 years while tracking close "
            "approaches, spin angular momentum, and conservative-plus-dissipative energy bookkeeping."
        ),
    ]
    period = f"{_date_only(climate.get('start_utc'))} to {_date_only(climate.get('end_utc'))}"
    return ReportView(
        title="Enhanced two-moon eclipse climate",
        period=period,
        headline=headline,
        summary_points=summary_points,
        total_events=total_events,
        temporal_pair_count=len(pairs),
        double_totality_pair_count=double_totality_count,
        count_rows=count_rows,
        inclination_rows=inclination_rows,
        pair_rows=pair_rows,
        rotation_rows=rotation_rows,
        comparison_rows=comparison_rows,
        count_comparison_rows=count_comparison_rows,
        force_items=force_items,
        assumptions=assumptions,
        limitations=limitations,
        validation_rows=validation_rows,
        validation_notes=validation_notes,
        convergence_notes=convergence_notes,
        next_checks=next_checks,
        figures=figures,
        artifacts=artifacts,
    )


def _event_count_rows(summary: Mapping[str, Any]) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    grand_total = 0
    for domain, key in (("solar", "solar_event_counts"), ("lunar", "lunar_event_counts")):
        groups = _mapping(summary.get(key))
        for body in ("real_moon", "second_moon"):
            counts = _mapping(groups.get(body))
            total = sum(_integer(value) for value in counts.values())
            grand_total += total
            breakdown = " · ".join(
                f"{str(kind).title()} {_integer(value):,}"
                for kind, value in counts.items()
            )
            rows.append(
                [DOMAIN_LABELS[domain], BODY_LABELS[body], f"{total:,}", breakdown]
            )
    return rows, grand_total


def _domain_total(summary: Mapping[str, Any], domain: str) -> int:
    groups = _mapping(summary.get(f"{domain}_event_counts"))
    return sum(
        _integer(value)
        for counts in groups.values()
        for value in _mapping(counts).values()
    )


def _inclination_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    ranges = _mapping(summary.get("inclination_ranges_deg"))
    rows = []
    for body in ("real_moon", "second_moon"):
        values = _mapping(ranges.get(body))
        rows.append(
            [
                BODY_LABELS[body],
                _fmt(values.get("minimum"), 3, "°"),
                _fmt(values.get("maximum"), 3, "°"),
            ]
        )
    return rows


def _headline_pair(pairs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not pairs:
        return None
    same_location = [pair for pair in pairs if pair.get("definition_a_same_location")]
    candidates = same_location or list(pairs)
    return min(
        candidates,
        key=lambda pair: min(
            str(pair.get("real_maximum_utc", "")),
            str(pair.get("second_maximum_utc", "")),
        ),
    )


def _headline(pair: Mapping[str, Any]) -> str:
    sequence = _pair_sequence(pair)
    first = sequence[0]
    second = sequence[1]
    local_hours = _finite(pair.get("local_maximum_separation_hours"))
    separation_seconds = (
        local_hours * 3_600.0
        if local_hours is not None
        else (_finite(pair.get("separation_hours")) or 0.0) * 3_600.0
    )
    latitude = _finite(pair.get("best_common_latitude_deg"))
    longitude = _finite(pair.get("best_common_longitude_deg"))
    location = (
        f"at {latitude:.3f}°, {longitude:.3f}°"
        if latitude is not None and longitude is not None
        else "at the optimized common location"
    )
    totality = (
        "two separate totalities"
        if pair.get("both_total_at_common_location")
        else "two separate solar eclipses"
    )
    chain_kind = (
        "same-location chain"
        if pair.get("definition_a_same_location")
        else "rapid pair"
    )
    return (
        f"The enhanced model's earliest {chain_kind} produces {totality} {location}: "
        f"{first['label']} reaches {first['kind']} at {_short_utc(first['time'])}, followed "
        f"{_fmt_duration(separation_seconds)} later by {second['label']} at "
        f"{_short_utc(second['time'])} ({second['kind']})."
    )


def _pair_sequence(pair: Mapping[str, Any]) -> list[dict[str, str]]:
    events = []
    for prefix, label in (("real", "Real Moon"), ("second", "Second moon")):
        local = _mapping(pair.get(f"{prefix}_local"))
        time = str(
            local.get("maximum_utc")
            or pair.get(f"{prefix}_maximum_utc")
            or "unknown time"
        )
        kind = str(local.get("eclipse_type") or pair.get(f"{prefix}_type") or "unknown")
        events.append({"label": label, "time": time, "kind": kind})
    return sorted(events, key=lambda event: event["time"])


def _pair_row(pair: Mapping[str, Any]) -> list[str]:
    sequence = _pair_sequence(pair)
    sequence_text = " → ".join(
        f"{event['label']} {event['kind']} at {_short_utc(event['time'])}"
        for event in sequence
    )
    latitude = _finite(pair.get("best_common_latitude_deg"))
    longitude = _finite(pair.get("best_common_longitude_deg"))
    location = (
        f"{latitude:.3f}°, {longitude:.3f}°"
        if latitude is not None and longitude is not None
        else "not solved"
    )
    global_hours = _finite(pair.get("separation_hours"))
    local_hours = _finite(pair.get("local_maximum_separation_hours"))
    distance = _finite(pair.get("track_distance_km"))
    flags = []
    if pair.get("definition_a_same_location"):
        flags.append("A")
    if pair.get("definition_b_500km"):
        flags.append("B-500")
    if pair.get("both_total_at_common_location"):
        flags.append("double totality")
    return [
        sequence_text,
        _fmt_duration((global_hours or 0.0) * 3_600.0),
        _fmt_duration((local_hours or global_hours or 0.0) * 3_600.0),
        _fmt(distance, 1, " km"),
        location,
        ", ".join(flags) or "temporal pair only",
    ]


def _rotation_rows(rotation: Mapping[str, Any]) -> list[list[str]]:
    if not rotation:
        return []
    definitions = (
        ("Final mean-solar LOD delta", "final_delta_mean_solar_lod_ms", 6, " ms"),
        ("Final UT1 delta", "final_delta_ut1_s", 6, " s"),
        ("Final longitude shift", "final_longitude_shift_deg", 6, "°"),
        (
            "Maximum absolute longitude shift",
            "maximum_abs_longitude_shift_deg",
            6,
            "°",
        ),
        ("Final pole separation", "final_pole_separation_arcsec", 3, " arcsec"),
        (
            "Maximum pole separation",
            "maximum_pole_separation_arcsec",
            3,
            " arcsec",
        ),
    )
    return [
        [label, _fmt(rotation.get(key), digits, unit, signed=True)]
        for label, key, digits, unit in definitions
        if _finite(rotation.get(key)) is not None
    ]


def _comparison_rows(
    summary: Mapping[str, Any], comparison: Mapping[str, Any]
) -> list[list[str]]:
    matching = _mapping(comparison.get("matching"))
    rows = [
        ["Baseline events", f"{_integer(summary.get('baseline_event_count')):,}"],
        ["Enhanced events", f"{_integer(summary.get('enhanced_event_count')):,}"],
        ["Inventory delta", f"{_integer(summary.get('event_count_delta')):+d}"],
        ["Matched events", f"{_integer(summary.get('matched_event_count')):,}"],
        ["Added / removed", f"{_integer(summary.get('added_event_count'))} / {_integer(summary.get('removed_event_count'))}"],
        ["Classification changes", f"{_integer(summary.get('type_change_count')):,}"],
        [
            "Median absolute time shift",
            _fmt_duration(_finite(summary.get("median_absolute_time_shift_seconds"))),
        ],
        [
            "Maximum absolute time shift",
            _fmt_duration(_finite(summary.get("maximum_absolute_time_shift_seconds"))),
        ],
        [
            "Median solar maximum-point shift",
            _fmt(summary.get("median_global_maximum_point_shift_km"), 1, " km"),
        ],
        [
            "Maximum solar maximum-point shift",
            _fmt(summary.get("maximum_global_maximum_point_shift_km"), 1, " km"),
        ],
    ]
    if matching:
        rows.append(
            ["One-to-one match window", _fmt(matching.get("max_match_days"), 1, " days")]
        )
    pair_changes = _mapping(comparison.get("pair_and_chain_changes"))
    if pair_changes:
        rows.extend(
            [
                [
                    "Rapid solar pairs (baseline → enhanced)",
                    f"{_integer(pair_changes.get('baseline_pair_count'))} → "
                    f"{_integer(pair_changes.get('enhanced_pair_count'))} "
                    f"({_integer(pair_changes.get('pair_count_delta')):+d})",
                ],
                [
                    "Matched / added / removed rapid pairs",
                    f"{_integer(pair_changes.get('matched_pair_count'))} / "
                    f"{_integer(pair_changes.get('added_pair_count'))} / "
                    f"{_integer(pair_changes.get('removed_pair_count'))}",
                ],
            ]
        )
        flag_counts = _mapping(pair_changes.get("chain_flag_counts"))
        for flag, label in (
            ("definition_a_same_location", "Same-location chains"),
            ("definition_b_500km", "Regional chains within 500 km"),
            ("both_total_at_common_location", "Double-totality chains"),
        ):
            counts = _mapping(flag_counts.get(flag))
            if counts:
                rows.append(
                    [
                        f"{label} (baseline → enhanced)",
                        f"{_integer(counts.get('baseline'))} → "
                        f"{_integer(counts.get('enhanced'))} "
                        f"({_integer(counts.get('delta')):+d})",
                    ]
                )
    return rows


def _count_comparison_rows(comparison: Mapping[str, Any]) -> list[list[str]]:
    counts = _mapping(comparison.get("event_counts"))
    rows = []
    for row in _mapping_list(counts.get("by_domain_and_body")):
        rows.append(
            [
                DOMAIN_LABELS.get(str(row.get("domain")), str(row.get("domain", ""))),
                BODY_LABELS.get(str(row.get("body")), str(row.get("body", ""))),
                str(_integer(row.get("baseline"))),
                str(_integer(row.get("enhanced"))),
                f"{_integer(row.get('delta')):+d}",
            ]
        )
    return rows


def _force_items(trajectory: Mapping[str, Any]) -> list[str]:
    force = _mapping(trajectory.get("force_model"))
    planets = _mapping(trajectory.get("planetary_perturbers"))
    tides = _mapping(force.get("earth_tides_and_spin"))
    corrections = _mapping(force.get("earth_j2_and_first_post_newtonian"))
    j2 = _mapping(corrections.get("earth_j2"))
    relativity = _mapping(corrections.get("first_post_newtonian"))
    items = []
    newtonian = force.get("newtonian")
    if newtonian:
        items.append(str(newtonian) + ".")
    bodies = [str(value).title() for value in _sequence(planets.get("bodies"))]
    if bodies:
        items.append(
            "Active DE440-initialized planetary perturbers: " + ", ".join(bodies) + "."
        )
    if j2.get("enabled"):
        items.append(
            f"Earth J2 = {_fmt(j2.get('j2'), 10)} with equatorial radius "
            f"{_fmt(j2.get('equatorial_radius_km'), 3, ' km')}; the spin axis is fixed in ICRF."
        )
    if relativity.get("enabled"):
        items.append(
            "First-order post-Newtonian gravity is active through REBOUNDx gr_full for all "
            "active-body interactions."
        )
    if tides.get("enabled"):
        items.append(
            f"Earth tides and spin use {tides.get('model', 'a constant-time-lag model')}, "
            f"calibrated to {tides.get('calibration_target', 'the saved recession target')}."
        )
    integrator = str(trajectory.get("integrator", "IAS15"))
    integrator_label = (
        integrator if integrator.casefold().startswith("rebound") else f"REBOUND {integrator}"
    )
    items.append(
        f"{integrator_label} uses {_fmt(trajectory.get('sample_step_seconds'), 0, ' s')} "
        "saved trajectory knots; eclipse maxima and contacts are refined between knots."
    )
    return items


def _assumptions(
    climate: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    rotation = _mapping(trajectory.get("alternate_earth_rotation"))
    matching = _mapping(comparison.get("matching"))
    assumptions = [
        (
            f"Scope: {_date_only(climate.get('start_utc'))} through "
            f"{_date_only(climate.get('end_utc'))}; the hypothetical moon's saved initial "
            "Cartesian state and mass remain fixed for the entire run."
        ),
        (
            "DE440s supplies real-body states at the 2026 epoch only. The counterfactual system "
            "then evolves freely rather than being forced back onto the real Solar System ephemeris."
        ),
        (
            "Mars through Neptune are planetary-system barycentres with system GMs; planetary "
            "satellites are not integrated separately."
        ),
    ]
    if rotation:
        assumptions.append(
            "Alternate Earth orientation is the differential spin phase and pole between the "
            "massive-second-moon run and a matched massless-second-moon control, composed onto "
            "Skyfield's observed-Earth orientation reference. It is not an absolute UT1 forecast."
        )
    if matching:
        assumptions.append(
            f"Catalog comparison uses one-to-one nearest-time assignment within "
            f"{_fmt(matching.get('max_match_days'), 1, ' days')}, separately by eclipse domain and moon."
        )
        assumptions.append(
            "The comparison labels an event added or removed when it cannot be paired inside "
            "that time window. Those labels do not by themselves prove that a physical eclipse "
            "appeared or disappeared; near-threshold partial and penumbral events can also cross "
            "the detector boundary."
        )
    return assumptions


def _limitations(
    climate: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    candidates = [
        *_sequence(climate.get("limitations")),
        *_sequence(trajectory.get("omissions")),
        *_sequence(comparison.get("limitations")),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip().rstrip(".") + "."
        text = text[:1].upper() + text[1:]
        key = text.casefold()
        if text != "." and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _validation_summary(
    validation: Mapping[str, Any] | None,
) -> tuple[list[list[str]], list[str]]:
    if validation is None:
        return [], []
    rows = []
    for result in _mapping_list(validation.get("results")):
        reference = _mapping(result.get("reference"))
        rows.append(
            [
                str(reference.get("event_id") or result.get("event_id") or "event"),
                str(result.get("model_eclipse_type") or reference.get("eclipse_type") or "—"),
                _fmt(result.get("timing_error_tt_s"), 3, " s"),
                _fmt(result.get("coordinate_error_wgs84_km"), 3, " km"),
                _fmt(result.get("central_duration_error_s"), 3, " s", signed=True),
                "Pass" if result.get("passed") else "Fail",
            ]
        )
    notes = []
    if "all_passed" in validation:
        notes.append(
            f"Published real-Moon eclipse validation: {'all checks passed' if validation.get('all_passed') else 'one or more checks failed'}."
        )
    authority = validation.get("reference_authority")
    if authority:
        notes.append(
            f"Reference authority: {authority}. This validates the real-system geometry pipeline, "
            "not the late counterfactual trajectory itself."
        )
    return rows, notes


def _convergence_summary(
    convergences: Sequence[Mapping[str, Any]],
    sources: Sequence[Path],
) -> list[str]:
    if not convergences:
        return []
    notes: list[str] = []
    for convergence, source in zip(convergences, sources, strict=True):
        label = source.parent.name.replace("_", " ")
        notes.extend(_one_convergence_summary(convergence, label=label))
    return notes


def _one_convergence_summary(
    convergence: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    notes: list[str] = []
    display_label = label[:1].upper() + label[1:]
    detector = _mapping(convergence.get("detector_cadence_comparison"))
    if detector:
        max_deltas = [
            value
            for value in _mapping(detector.get("maximum_axis_time_delta_seconds")).values()
            if _finite(value) is not None
        ]
        maximum = max((_finite(value) or 0.0 for value in max_deltas), default=0.0)
        notes.append(
            f"{display_label}: detector cadence "
            f"{_fmt(detector.get('baseline_detector_step_seconds'), 0, ' s')} versus "
            f"{_fmt(detector.get('comparison_detector_step_seconds'), 0, ' s')}: "
            f"{_integer(detector.get('solar_type_mismatches'))} solar and "
            f"{_integer(detector.get('lunar_type_mismatches'))} lunar type mismatches; "
            f"maximum saved axis-time difference {_fmt(maximum, 6, ' s')}."
        )
    interpolation = _mapping(convergence.get("trajectory_interpolation_comparison"))
    if interpolation:
        center_deltas = [
            _finite(_mapping(interpolation.get(body)).get("maximum_center_delta_km"))
            for body in ("real_moon", "second_moon")
        ]
        maximum_center = max((value for value in center_deltas if value is not None), default=None)
        notes.append(
            f"{display_label}: trajectory-knot comparison has maximum retained central-point displacement "
            f"{_fmt(maximum_center, 6, ' km')}."
        )
    comparisons = _mapping(convergence.get("comparisons"))
    for cadence_label, raw in comparisons.items():
        comparison = _mapping(raw)
        notes.append(
            f"{display_label}, cadence {str(cadence_label).replace('_', ' ')}: event counts "
            f"{'match' if comparison.get('event_counts_match') else 'do not match'}; "
            f"maximum axis-time difference "
            f"{_fmt(comparison.get('max_axis_time_difference_s'), 6, ' s')}."
        )
    interpretation = convergence.get("interpretation")
    if interpretation:
        notes.append(str(interpretation))
    comparison_summary = _mapping(convergence.get("summary"))
    if comparison_summary and "matched_event_count" in comparison_summary:
        notes.append(
            f"{display_label}: {_integer(comparison_summary.get('baseline_event_count'))} versus "
            f"{_integer(comparison_summary.get('enhanced_event_count'))} events, "
            f"{_integer(comparison_summary.get('type_change_count'))} type changes, maximum "
            f"time difference {_fmt(comparison_summary.get('maximum_absolute_time_shift_seconds'), 6, ' s')}, "
            "and maximum saved solar-point difference "
            f"{_fmt(comparison_summary.get('maximum_global_maximum_point_shift_km'), 6, ' km')}."
        )
    if not notes:
        notes.append(
            f"{display_label}: a convergence file was supplied; inspect the linked JSON for its full schema."
        )
    return notes


def _artifacts(
    output: Path,
    climate_source: Path,
    comparison_source: Path,
    validation_source: Path | None,
    convergence_sources: Sequence[Path],
) -> tuple[list[Artifact], list[Artifact]]:
    figure_specs = (
        ("eclipse_climate.png", "Thirty-year eclipse climate", "Inclinations and eclipse activity over the enhanced run."),
        ("standout_tracks.png", "Standout total-eclipse tracks", "Selected central tracks from the enhanced catalog."),
        ("stability.png", "Enhanced stability diagnostics", "Orbital and numerical diagnostics for the enhanced force model."),
    )
    figures = [
        Artifact(label, _relative(output, output / name), caption, image=True)
        for name, label, caption in figure_specs
        if (output / name).exists()
    ]
    tide_directory = output.parent / "tides_30d"
    tide_poster = tide_directory / "two_moon_equilibrium_tides.png"
    if tide_poster.exists():
        figures.append(
            Artifact(
                "Thirty-day two-moon equilibrium tide",
                _relative(output, tide_poster),
                "Exact tide-generating potential from both moons; an equilibrium-ocean proxy, not a coastal water-level forecast.",
                image=True,
            )
        )
    comparison_figure = next(
        (
            candidate
            for candidate in (output / "comparison" / "comparison.png", output / "comparison.png")
            if candidate.exists()
        ),
        None,
    )
    if comparison_figure is not None:
        figures.insert(
            1,
            Artifact(
                "Baseline comparison",
                _relative(output, comparison_figure),
                "Timing and maximum-point shifts introduced by enhanced dynamics.",
                image=True,
            ),
        )
    source_specs: list[tuple[str, Path | None]] = [
        ("Enhanced climate JSON", climate_source),
        ("Baseline comparison JSON", comparison_source),
        ("Validation JSON", validation_source),
        *[
            (f"Convergence JSON · {source.parent.name.replace('_', ' ')}", source)
            for source in convergence_sources
        ],
    ]
    local_specs = (
        ("Event catalog CSV", "events.csv"),
        ("Annual counts CSV", "annual_counts.csv"),
        ("Inclination samples CSV", "inclinations.csv"),
        ("Notable events CSV", "notable_events.csv"),
        ("Rapid-pair CSV", "temporal_solar_pairs.csv"),
        ("Track coordinates JSON", "standout_tracks.json"),
        ("Climate figure SVG", "eclipse_climate.svg"),
        ("Track figure SVG", "standout_tracks.svg"),
    )
    artifacts: list[Artifact] = []
    seen: set[Path] = set()
    for label, path in source_specs:
        if path is not None and path.exists() and path not in seen:
            artifacts.append(Artifact(label, _relative(output, path)))
            seen.add(path)
    for label, name in local_specs:
        path = (output / name).resolve()
        if path.exists() and path not in seen:
            artifacts.append(Artifact(label, _relative(output, path)))
            seen.add(path)
    for label, name in (
        ("Thirty-day two-moon tide animation", "two_moon_equilibrium_tides.mp4"),
        ("Two-moon tide frame data", "two_moon_equilibrium_tides.csv"),
        ("Two-moon tide model manifest", "two_moon_equilibrium_tides.json"),
    ):
        path = (tide_directory / name).resolve()
        if path.exists() and path not in seen:
            artifacts.append(Artifact(label, _relative(output, path)))
            seen.add(path)
    matched_events = next(
        (
            candidate
            for candidate in (
                output / "comparison" / "matched_events.csv",
                output / "matched_events.csv",
            )
            if candidate.exists()
        ),
        None,
    )
    if matched_events is not None and matched_events.resolve() not in seen:
        artifacts.append(
            Artifact("Matched-event CSV", _relative(output, matched_events))
        )
    return figures, artifacts


def _render_markdown(view: ReportView) -> str:
    lines = [
        f"# {view.title}",
        "",
        f"**Modeled period:** {view.period}",
        "",
        f"> **Headline result:** {view.headline}",
        "",
        "## Technical summary",
        "",
        *[f"- {point}" for point in view.summary_points],
        "",
    ]
    lines.extend(_markdown_figures(view.figures[:1]))
    lines.extend(
        [
            "## One double-totality chain survives the enhanced force model",
            "",
            (
                "Definition A requires both eclipses at one observing location within 12 hours. "
                "Definition B requires their central or maximum tracks to approach within 500 km."
            ),
            "",
            _markdown_table(
                ["Chronological local sequence", "Global gap", "Local gap", "Track gap", "Common location", "Result"],
                view.pair_rows,
            ),
            "",
        ]
    )
    tide_figures = [
        figure
        for figure in view.figures
        if figure.label == "Thirty-day two-moon equilibrium tide"
    ]
    if tide_figures:
        lines.extend(
            [
                "## Thirty days of interacting lunar tides",
                "",
                (
                    "The companion animation follows the exact equilibrium tide-generating "
                    "potential of both moons on the rotating WGS84 Earth. It isolates lunar "
                    "interference and is not a hydrodynamic or coastal-flood forecast."
                ),
                "",
            ]
        )
        lines.extend(_markdown_figures(tide_figures))
    lines.extend(_markdown_figures([figure for figure in view.figures if "track" in figure.label.lower()]))
    lines.extend(
        [
            "## The enhanced system produces a dense 30-year eclipse climate",
            "",
            _markdown_table(["Domain", "Occulter", "Count", "Breakdown"], view.count_rows),
            "",
            "The inclination exchange remains the dominant eclipse-rate regulator:",
            "",
            _markdown_table(["Moon", "Minimum inclination", "Maximum inclination"], view.inclination_rows),
            "",
        ]
    )
    lines.extend(
        [
            "## Enhanced dynamics measurably changes timing and geography",
            "",
            _markdown_table(["Comparison metric", "Result"], view.comparison_rows),
            "",
            _markdown_table(["Domain", "Moon", "Baseline", "Enhanced", "Delta"], view.count_comparison_rows),
            "",
            (
                "Geographic comparison uses the WGS84 distance between saved global-maximum "
                "points. It does not claim that entire eclipse paths move by that distance."
            ),
            "",
        ]
    )
    lines.extend(
        _markdown_figures(
            [figure for figure in view.figures if figure.label == "Baseline comparison"]
        )
    )
    lines.extend(
        [
            "## Earth rotation changes enough to matter for late ground tracks",
            "",
        ]
    )
    if view.rotation_rows:
        lines.extend(
            [
                _markdown_table(["Differential orientation metric", "Value"], view.rotation_rows),
                "",
                (
                    "These are full-system minus massless-second-moon control deltas layered onto "
                    "Skyfield Earth orientation, not predictions of absolute future UT1 or polar motion."
                ),
                "",
            ]
        )
    else:
        lines.extend(["No alternate-Earth-rotation diagnostics were saved in this catalog.", ""])
    lines.extend(
        [
            "## Model specification and active forces",
            "",
            *[f"- {item}" for item in view.force_items],
            "",
            "## Validation and numerical robustness",
            "",
        ]
    )
    if view.validation_notes or view.validation_rows:
        lines.extend([*[f"- {note}" for note in view.validation_notes], ""])
        if view.validation_rows:
            lines.extend(
                [
                    _markdown_table(
                        ["Reference eclipse", "Type", "Timing error", "Position error", "Duration error", "Status"],
                        view.validation_rows,
                    ),
                    "",
                ]
            )
    else:
        lines.extend(["No external validation JSON was supplied to this report build.", ""])
    if view.convergence_notes:
        lines.extend([*[f"- {note}" for note in view.convergence_notes], ""])
    else:
        lines.extend(["No enhanced-run convergence JSON was supplied to this report build.", ""])
    lines.extend(
        [
            "## Assumptions that govern interpretation",
            "",
            *[f"- {item}" for item in view.assumptions],
            "",
            "## Limitations and remaining uncertainty",
            "",
            *[f"- {item}" for item in view.limitations],
            "",
            "## Next checks",
            "",
            *[f"- {item}" for item in view.next_checks],
            "",
            "## Generated files",
            "",
            *[f"- [{artifact.label}]({artifact.relative_path})" for artifact in view.artifacts],
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_html(view: ReportView) -> str:
    figures_by_label = {figure.label: figure for figure in view.figures}
    sections = [
        _html_section(
            "Technical summary",
            "".join(f"<li>{escape(point)}</li>" for point in view.summary_points),
            list_container=True,
        ),
    ]
    climate_figure = figures_by_label.get("Thirty-year eclipse climate")
    if climate_figure:
        sections.append(_html_figure(climate_figure))
    pair_body = (
        "<p>Definition A requires both eclipses at one observing location within 12 hours; "
        "Definition B requires the central or maximum tracks to approach within 500 km.</p>"
        + _html_table(
            ["Chronological local sequence", "Global gap", "Local gap", "Track gap", "Common location", "Result"],
            view.pair_rows,
        )
    )
    track_figure = figures_by_label.get("Standout total-eclipse tracks")
    if track_figure:
        pair_body += _html_figure(track_figure)
    sections.append(
        _html_section("One double-totality chain survives the enhanced force model", pair_body)
    )
    sections.append(
        _html_section(
            "The enhanced system produces a dense 30-year eclipse climate",
            _html_table(["Domain", "Occulter", "Count", "Breakdown"], view.count_rows)
            + "<p>The inclination exchange remains the dominant eclipse-rate regulator.</p>"
            + _html_table(["Moon", "Minimum inclination", "Maximum inclination"], view.inclination_rows),
        )
    )
    comparison_body = _html_table(["Comparison metric", "Result"], view.comparison_rows)
    comparison_body += _html_table(
        ["Domain", "Moon", "Baseline", "Enhanced", "Delta"],
        view.count_comparison_rows,
    )
    comparison_body += (
        "<p class=\"note\">Geographic comparison uses WGS84 distance between saved "
        "global-maximum points, not complete eclipse paths.</p>"
    )
    comparison_figure = figures_by_label.get("Baseline comparison")
    if comparison_figure:
        comparison_body += _html_figure(comparison_figure)
    sections.append(
        _html_section("Enhanced dynamics measurably changes timing and geography", comparison_body)
    )
    rotation_body = (
        _html_table(["Differential orientation metric", "Value"], view.rotation_rows)
        + "<p class=\"note\">These are full-system minus massless-second-moon control "
        "deltas layered onto Skyfield Earth orientation, not absolute future UT1 or polar motion.</p>"
        if view.rotation_rows
        else "<p>No alternate-Earth-rotation diagnostics were saved in this catalog.</p>"
    )
    sections.append(
        _html_section("Earth rotation changes enough to matter for late ground tracks", rotation_body)
    )
    tide_figure = figures_by_label.get("Thirty-day two-moon equilibrium tide")
    if tide_figure:
        sections.append(
            _html_section(
                "Thirty days of interacting lunar tides",
                (
                    "<p>The companion animation follows the exact equilibrium "
                    "tide-generating potential of both moons on the rotating WGS84 Earth. "
                    "It isolates lunar interference and is not a hydrodynamic or "
                    "coastal-flood forecast.</p>"
                    + _html_figure(tide_figure)
                ),
            )
        )
    sections.append(
        _html_section(
            "Model specification and active forces",
            "".join(f"<li>{escape(item)}</li>" for item in view.force_items),
            list_container=True,
        )
    )
    validation_body = ""
    if view.validation_notes:
        validation_body += "<ul>" + "".join(
            f"<li>{escape(note)}</li>" for note in view.validation_notes
        ) + "</ul>"
    if view.validation_rows:
        validation_body += _html_table(
            ["Reference eclipse", "Type", "Timing error", "Position error", "Duration error", "Status"],
            view.validation_rows,
        )
    if not validation_body:
        validation_body = "<p>No external validation JSON was supplied to this report build.</p>"
    if view.convergence_notes:
        validation_body += "<ul>" + "".join(
            f"<li>{escape(note)}</li>" for note in view.convergence_notes
        ) + "</ul>"
    else:
        validation_body += "<p>No enhanced-run convergence JSON was supplied.</p>"
    sections.append(_html_section("Validation and numerical robustness", validation_body))
    sections.append(
        _html_section(
            "Assumptions that govern interpretation",
            "".join(f"<li>{escape(item)}</li>" for item in view.assumptions),
            list_container=True,
        )
    )
    sections.append(
        _html_section(
            "Limitations and remaining uncertainty",
            "".join(f"<li>{escape(item)}</li>" for item in view.limitations),
            list_container=True,
        )
    )
    sections.append(
        _html_section(
            "Next checks",
            "".join(f"<li>{escape(item)}</li>" for item in view.next_checks),
            list_container=True,
        )
    )
    artifact_links = "".join(
        f'<li><a href="{escape(item.relative_path, quote=True)}">{escape(item.label)}</a></li>'
        for item in view.artifacts
    )
    sections.append(_html_section("Generated files", artifact_links, list_container=True))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{escape(view.title)}</title>
  <style>
    :root {{
      --bg: #f5f7fa; --surface: #ffffff; --ink: #132236; --muted: #58697d;
      --line: #d9e1e8; --accent: #1f648f; --accent-soft: #e3f1f7;
      --second: #c66b20; --shadow: 0 10px 30px rgba(24, 43, 64, .08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #101820; --surface: #17232e; --ink: #e8eef3; --muted: #afbecb;
        --line: #344654; --accent: #79bddf; --accent-soft: #1d3847;
        --second: #efad74; --shadow: none;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.62 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: var(--accent); }}
    header {{ background: linear-gradient(135deg, var(--surface), var(--accent-soft)); border-bottom: 1px solid var(--line); }}
    .wrap {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; }}
    header .wrap {{ padding: 64px 0 48px; }}
    .eyebrow {{ color: var(--muted); letter-spacing: .09em; text-transform: uppercase; font-size: .78rem; font-weight: 700; }}
    h1 {{ margin: .25rem 0 1rem; font-size: clamp(2rem, 6vw, 4.2rem); line-height: 1.03; letter-spacing: -.04em; }}
    .headline {{ max-width: 920px; font-size: 1.18rem; line-height: 1.55; margin: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 30px; }}
    .metric {{ background: var(--surface); border: 1px solid var(--line); padding: 18px; border-radius: 14px; box-shadow: var(--shadow); }}
    .metric strong {{ display: block; font-size: 1.65rem; line-height: 1.15; }}
    .metric span {{ color: var(--muted); font-size: .88rem; }}
    main {{ padding: 28px 0 72px; }}
    section {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: clamp(22px, 5vw, 42px); margin: 20px 0; box-shadow: var(--shadow); }}
    h2 {{ margin: 0 0 18px; font-size: clamp(1.35rem, 4vw, 2rem); line-height: 1.2; letter-spacing: -.02em; }}
    p:first-child, ul:first-child {{ margin-top: 0; }}
    p:last-child, ul:last-child {{ margin-bottom: 0; }}
    li + li {{ margin-top: .45rem; }}
    .table-wrap {{ overflow-x: auto; margin: 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 620px; font-size: .9rem; }}
    th {{ text-align: left; color: var(--muted); font-size: .76rem; letter-spacing: .055em; text-transform: uppercase; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    figure {{ margin: 24px 0 4px; }}
    figure img {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 12px; background: white; }}
    figcaption, .note {{ color: var(--muted); font-size: .87rem; }}
    figcaption {{ margin-top: 8px; }}
    footer {{ color: var(--muted); border-top: 1px solid var(--line); padding: 24px 0 40px; font-size: .85rem; }}
    @media (max-width: 680px) {{ .metrics {{ grid-template-columns: 1fr; }} header .wrap {{ padding-top: 42px; }} }}
    @media print {{ body {{ background: white; }} section, .metric {{ box-shadow: none; break-inside: avoid; }} a {{ color: inherit; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="eyebrow">Enhanced N-body eclipse simulation · {escape(view.period)}</div>
      <h1>{escape(view.title)}</h1>
      <p class="headline"><strong>Headline result.</strong> {escape(view.headline)}</p>
      <div class="metrics" aria-label="Key metrics">
        <div class="metric"><strong>{view.total_events:,}</strong><span>solar and lunar eclipses</span></div>
        <div class="metric"><strong>{view.temporal_pair_count}</strong><span>solar pairs within 12 hours</span></div>
        <div class="metric"><strong>{view.double_totality_pair_count}</strong><span>same-location double-totality pair</span></div>
      </div>
    </div>
  </header>
  <main class="wrap">
    {''.join(sections)}
  </main>
  <footer><div class="wrap">Generated from the saved enhanced climate and baseline-comparison catalogs. Values are scenario-model outputs, not observational forecasts.</div></footer>
</body>
</html>
"""


def _html_section(title: str, body: str, *, list_container: bool = False) -> str:
    content = f"<ul>{body}</ul>" if list_container else body
    return f"<section><h2>{escape(title)}</h2>{content}</section>"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "<p class=\"note\">No rows were supplied.</p>"
    head = "".join(f"<th scope=\"col\">{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _html_figure(artifact: Artifact) -> str:
    caption = f"<figcaption>{escape(artifact.caption or artifact.label)}</figcaption>"
    return (
        f'<figure><img src="{escape(artifact.relative_path, quote=True)}" '
        f'alt="{escape(artifact.label, quote=True)}" loading="lazy">{caption}</figure>'
    )


def _markdown_figures(figures: Sequence[Artifact]) -> list[str]:
    lines: list[str] = []
    for figure in figures:
        lines.extend(
            [
                f"![{figure.label}]({figure.relative_path})",
                "",
                f"*{figure.caption or figure.label}*",
                "",
            ]
        )
    return lines


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_No rows were supplied._"
    escaped_headers = [_markdown_cell(value) for value in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object: {path}")
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str, label: str) -> None:
    if not isinstance(payload.get(key), Mapping):
        raise ValueError(f"{label} JSON is missing object field {key!r}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _normalize_paths(
    value: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if value is None:
        return []
    raw = [value] if isinstance(value, (str, Path)) else list(value)
    return [Path(item).resolve() for item in raw]


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(
    value: Any,
    digits: int = 1,
    unit: str = "",
    *,
    signed: bool = False,
) -> str:
    number = _finite(value)
    if number is None:
        return "not available"
    sign = "+" if signed else ""
    return f"{number:{sign},.{digits}f}{unit}"


def _fmt_duration(seconds: float | None) -> str:
    value = _finite(seconds)
    if value is None:
        return "not available"
    sign = "−" if value < 0.0 else ""
    value = abs(value)
    if value >= 86_400.0:
        days = int(value // 86_400.0)
        hours = int((value % 86_400.0) // 3_600.0)
        return f"{sign}{days}d {hours}h"
    if value >= 3_600.0:
        hours = int(value // 3_600.0)
        minutes = int(round((value % 3_600.0) / 60.0))
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{sign}{hours}h {minutes:02d}m"
    if value >= 60.0:
        minutes = int(value // 60.0)
        remaining = value % 60.0
        return f"{sign}{minutes}m {remaining:04.1f}s"
    return f"{sign}{value:.3f}s"


def _date_only(value: Any) -> str:
    text = str(value or "not specified")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _short_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _relative(output: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--climate",
        type=Path,
        default=Path("outputs/coupled/eclipse_climate_30y_enhanced/climate.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(
            "outputs/coupled/eclipse_climate_30y_enhanced/comparison/comparison.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--convergence",
        type=Path,
        action="append",
        help="Optional convergence JSON; repeat for detector and trajectory checks",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    result = generate_enhanced_report(
        arguments.climate,
        arguments.comparison,
        output_dir=arguments.output_dir,
        validation_path=arguments.validation,
        convergence_path=arguments.convergence,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
