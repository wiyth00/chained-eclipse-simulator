"""Command-line orchestration for validation, design, fixed search, and reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .constants import MODEL_VERSION, SECOND_MOON_MASS_KG
from .eclipse_geometry import angular_series, maximum_visibility_grid
from .ephemeris import enumerate_real_solar_eclipses, load_ephemeris
from .mapping import (
    plot_angular_geometry,
    plot_contact_timeline,
    plot_stability_elements,
    plot_world_tracks,
)
from .models import ChainedEvent, OrbitalElements
from .optimize import optimize_earliest_design, screen_design_opportunities
from .orbital_dynamics import elements_to_state, integrate_restricted
from .report import build_report_artifact
from .search import design_mode_event, fixed_system_search
from .sensitivity import run_sensitivity_analysis
from .stability import StabilityConfig, run_stability_check
from .validation import build_validation_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "baseline.yaml"
DEFAULT_OUTPUTS = ROOT / "outputs"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _baseline_elements(config: dict[str, Any]) -> OrbitalElements:
    moon = config["second_moon"]
    model = config["model"]
    return OrbitalElements(
        semimajor_axis_km=float(moon["semimajor_axis_km"]),
        eccentricity=float(moon["eccentricity"]),
        inclination_deg=float(moon["inclination_deg"]),
        longitude_ascending_node_deg=float(moon.get("longitude_ascending_node_deg") or 0.0),
        argument_periapsis_deg=float(moon.get("argument_periapsis_deg") or 0.0),
        mean_anomaly_deg=float(moon.get("mean_anomaly_deg") or 0.0),
        epoch_utc=str(model["epoch_utc"]),
        radius_km=float(moon["radius_km"]),
        density_kg_m3=float(moon["density_kg_m3"]),
        mass_kg=SECOND_MOON_MASS_KG,
    )


def _save_optimized_config(
    path: Path,
    elements: OrbitalElements,
    diagnostics: dict[str, Any],
) -> None:
    state = elements_to_state(elements)
    payload = {
        "schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "mode": "fixed-system",
        "model_boundary": {
            "eclipse_search": "restricted DE440s-forced real bodies; second moon has no back-reaction",
            "stability": "fully coupled Newtonian four-body REBOUND experiment",
        },
        "orbital_elements": elements.to_dict(),
        "epoch_state": {
            "frame": "Earth-centred ICRF/J2000",
            "position_km": state[:3].tolist(),
            "velocity_km_s": state[3:].tolist(),
        },
        "mass_calculation": {
            "formula": "4/3*pi*radius^3*density",
            "mass_kg": elements.mass_kg,
        },
        "design_diagnostics": diagnostics,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_optimized_elements(path: Path) -> OrbitalElements:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OrbitalElements(**payload["orbital_elements"])


def _flatten_event(event: ChainedEvent) -> dict[str, Any]:
    return {
        "rank": event.rank,
        "event_id": event.event_id,
        "definition_a": event.definition_a,
        "definition_b": event.definition_b,
        "best_latitude_deg": event.best_latitude_deg,
        "best_longitude_deg": event.best_longitude_deg,
        "real_type": event.real_eclipse.eclipse_type,
        "real_maximum_utc": event.real_eclipse.maximum_utc,
        "real_c1_utc": event.real_eclipse.c1_utc,
        "real_c2_utc": event.real_eclipse.c2_utc,
        "real_c3_utc": event.real_eclipse.c3_utc,
        "real_c4_utc": event.real_eclipse.c4_utc,
        "real_magnitude": event.real_eclipse.magnitude,
        "real_obscuration": event.real_eclipse.obscuration,
        "real_central_duration_s": event.real_eclipse.central_duration_s,
        "real_solar_altitude_deg": event.real_eclipse.solar_altitude_deg,
        "second_type": event.second_eclipse.eclipse_type,
        "second_maximum_utc": event.second_eclipse.maximum_utc,
        "second_c1_utc": event.second_eclipse.c1_utc,
        "second_c2_utc": event.second_eclipse.c2_utc,
        "second_c3_utc": event.second_eclipse.c3_utc,
        "second_c4_utc": event.second_eclipse.c4_utc,
        "second_magnitude": event.second_eclipse.magnitude,
        "second_obscuration": event.second_eclipse.obscuration,
        "second_central_duration_s": event.second_eclipse.central_duration_s,
        "second_solar_altitude_deg": event.second_eclipse.solar_altitude_deg,
        "midpoint_separation_s": event.midpoint_separation_s,
        "track_distance_km": event.track_distance_km,
        "both_total": event.both_total,
        "two_totality_components": event.two_totality_components,
        "external_contact_intervals_disjoint": event.external_contact_intervals_disjoint,
        **event.thresholds,
    }


def _plot_event_bundle(
    context,
    event: ChainedEvent,
    tracks: tuple[dict[str, Any], dict[str, Any]],
    trajectory,
    output: Path,
) -> dict[str, str]:
    event_dir = output / "figures" / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    real_track, second_track = tracks
    real_tt = float(context.time_utc(event.real_eclipse.maximum_utc).tt)
    second_tt = float(context.time_utc(event.second_eclipse.maximum_utc).tt)
    real_grid = maximum_visibility_grid(
        context, real_tt, "real_moon", grid_step_deg=2.0, time_step_minutes=5.0
    )
    second_grid = maximum_visibility_grid(
        context,
        second_tt,
        "second_moon",
        position_provider=trajectory.position,
        grid_step_deg=2.0,
        time_step_minutes=5.0,
    )
    map_path = event_dir / "world_tracks.png"
    fig, _ = plot_world_tracks(
        {
            "label": "Real Moon",
            "longitude_deg": real_track["longitude_deg"],
            "latitude_deg": real_track["latitude_deg"],
            "grid_longitude_deg": real_grid["longitude_deg"],
            "grid_latitude_deg": real_grid["latitude_deg"],
            "grid_values": real_grid["visibility_mask"],
            "grid_boundary_level": 0.5,
            "boundary_note": "2° grid-derived envelope",
        },
        {
            "label": "Second moon",
            "longitude_deg": second_track["longitude_deg"],
            "latitude_deg": second_track["latitude_deg"],
            "grid_longitude_deg": second_grid["longitude_deg"],
            "grid_latitude_deg": second_grid["latitude_deg"],
            "grid_values": second_grid["visibility_mask"],
            "grid_boundary_level": 0.5,
            "boundary_note": "2° grid-derived envelope",
        },
        (event.best_latitude_deg, event.best_longitude_deg, "Best common location"),
        title=f"Chained eclipse ground tracks · {event.real_eclipse.maximum_utc[:10]}",
        subtitle="Central lines are 120 s samples; partial boundaries are topocentric 2° grid envelopes.",
        output_path=map_path,
    )
    plt.close(fig)
    timeline_path = event_dir / "contact_timeline.png"
    fig, _ = plot_contact_timeline(
        event.real_eclipse,
        event.second_eclipse,
        output_path=timeline_path,
    )
    plt.close(fig)
    real_angular = angular_series(
        context,
        real_tt,
        event.best_latitude_deg,
        event.best_longitude_deg,
        "real_moon",
        half_window_minutes=90.0,
        step_seconds=20.0,
    )
    second_angular = angular_series(
        context,
        second_tt,
        event.best_latitude_deg,
        event.best_longitude_deg,
        "second_moon",
        position_provider=trajectory.position,
        half_window_minutes=30.0,
        step_seconds=10.0,
    )
    angular_path = event_dir / "angular_geometry.png"
    fig, _ = plot_angular_geometry(
        [
            {
                "label": "Real Moon",
                "time_offset_seconds": real_angular["offset_seconds"],
                "center_separation_deg": real_angular["separation_deg"],
                "solar_angular_diameter_deg": 2.0 * real_angular["sun_radius_deg"],
                "occulter_angular_diameter_deg": 2.0 * real_angular["moon_radius_deg"],
                "maximum_utc": event.real_eclipse.maximum_utc,
            },
            {
                "label": "Second moon",
                "time_offset_seconds": second_angular["offset_seconds"],
                "center_separation_deg": second_angular["separation_deg"],
                "solar_angular_diameter_deg": 2.0 * second_angular["sun_radius_deg"],
                "occulter_angular_diameter_deg": 2.0 * second_angular["moon_radius_deg"],
                "maximum_utc": event.second_eclipse.maximum_utc,
            },
        ],
        output_path=angular_path,
    )
    plt.close(fig)
    return {
        "map": str(map_path.relative_to(output)),
        "timeline": str(timeline_path.relative_to(output)),
        "angular_geometry": str(angular_path.relative_to(output)),
    }


def _stability_figure(stability: dict[str, Any], output: Path) -> str:
    path = output / "figures" / "stability_elements.png"
    data = {
        "time_years": stability["time_years"],
        "series": {
            "Real Moon": {
                "time_years": stability["time_years"],
                "semimajor_axis_km": stability["real_moon"]["semimajor_axis_km"],
                "eccentricity": stability["real_moon"]["eccentricity"],
                "inclination_deg": stability["real_moon"]["inclination_deg"],
            },
            "Second moon": {
                "time_years": stability["time_years"],
                "semimajor_axis_km": stability["second_moon"]["semimajor_axis_km"],
                "eccentricity": stability["second_moon"]["eccentricity"],
                "inclination_deg": stability["second_moon"]["inclination_deg"],
            },
        },
        "relative_energy_error": stability["relative_energy_error"],
    }
    fig, _ = plot_stability_elements(data, output_path=path)
    plt.close(fig)
    return str(path.relative_to(output))


def _headline(event: ChainedEvent) -> str:
    latitude = f"{abs(event.best_latitude_deg):.4f}° {'N' if event.best_latitude_deg >= 0 else 'S'}"
    longitude = f"{abs(event.best_longitude_deg):.4f}° {'E' if event.best_longitude_deg >= 0 else 'W'}"
    gap_minutes = int(event.midpoint_separation_s // 60)
    gap_seconds = event.midpoint_separation_s - 60 * gap_minutes
    return (
        f"Under the optimized stable configuration, the earliest chained eclipse occurs on "
        f"{event.real_eclipse.maximum_utc[:10]}. At {latitude}, {longitude}, the real Moon "
        f"produces a {event.real_eclipse.eclipse_type} eclipse at {event.real_eclipse.maximum_utc}, "
        f"followed {gap_minutes} minutes {gap_seconds:.1f} seconds later by the second moon's "
        f"{event.second_eclipse.eclipse_type} eclipse at {event.second_eclipse.maximum_utc}."
    )


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = load_ephemeris(ROOT / config["model"]["ephemeris_cache"])
    start = context.time_utc(config["model"]["epoch_utc"])
    end_utc = args.end or config["model"]["end_utc"]
    end = context.time_utc(end_utc)
    print("[1/7] Enumerating and independently verifying real-Moon eclipses…", flush=True)
    real_eclipses = enumerate_real_solar_eclipses(context, start, end)
    pd.DataFrame([event.to_dict() for event in real_eclipses]).to_csv(
        output / "real_moon_eclipses.csv", index=False
    )
    _json_dump(output / "real_moon_eclipses.json", [event.to_dict() for event in real_eclipses])
    baseline = _baseline_elements(config)
    design_opportunities = screen_design_opportunities(
        context,
        real_eclipses,
        baseline,
        target_gap_minutes=float(config["optimization"]["design_totality_gap_minutes"]),
    )
    _json_dump(output / "design_mode_opportunities.json", design_opportunities)
    pd.DataFrame(design_opportunities).to_csv(
        output / "design_mode_opportunities.csv", index=False
    )

    print("[2/7] Validating against NASA/GSFC reference eclipses…", flush=True)
    validation = build_validation_report(context)
    _json_dump(output / "validation_report.json", validation)
    pd.DataFrame(
        [
            {
                "event": item["reference"]["event_id"],
                "type": item["model_eclipse_type"],
                "time_error_tt_s": item["timing_error_tt_s"],
                "position_error_wgs84_km": item["coordinate_error_wgs84_km"],
                "duration_error_s": item["central_duration_error_s"],
                "source_url": item["reference"]["source_url"],
            }
            for item in validation["results"]
        ]
    ).to_csv(output / "validation_results.csv", index=False)

    print("[3/7] Optimizing the earliest design-mode chain…", flush=True)
    design = optimize_earliest_design(
        context,
        real_eclipses,
        baseline,
        target_gap_minutes=float(config["optimization"]["design_totality_gap_minutes"]),
        run_global_search=not args.no_global,
        random_seed=int(config["optimization"]["random_seed"]),
    )
    design_event, _, _ = design_mode_event(context, design)
    optimized_path = output / "optimized_system.yaml"
    _save_optimized_config(optimized_path, design.optimized_elements, design.diagnostics)
    # Mirror the requested project-level configuration path.
    _save_optimized_config(ROOT / "config" / "optimized_system.yaml", design.optimized_elements, design.diagnostics)
    _json_dump(output / "design_mode_event.json", design_event.to_dict())

    print("[4/7] Propagating the fixed system and searching through 2100…", flush=True)
    fixed_trajectory = integrate_restricted(
        context,
        design.optimized_elements,
        float(end.tt),
        rtol=1e-10,
        max_step_seconds=10_800.0,
    )
    fixed_events, fixed_tracks = fixed_system_search(
        context, real_eclipses, fixed_trajectory
    )
    if not fixed_events:
        raise RuntimeError("fixed-system search produced no qualifying chained eclipses")
    rows = [_flatten_event(event) for event in fixed_events]
    pd.DataFrame(rows).to_csv(output / "all_candidates.csv", index=False)
    pd.DataFrame(rows[:10]).to_csv(output / "ranked_top10.csv", index=False)

    print("[5/7] Running the 1,000-year coupled REBOUND stability check…", flush=True)
    sample_days = args.stability_years * 365.25 / 500.0
    stability_result = run_stability_check(
        context,
        design.optimized_elements,
        config=StabilityConfig(
            years=args.stability_years,
            sample_interval_days=sample_days,
            ias15_epsilon=1e-10,
        ),
    )
    stability = stability_result.to_dict()
    _json_dump(output / "stability.json", stability)
    if not stability_result.stable:
        raise RuntimeError(
            f"optimized configuration failed stability: {stability_result.termination_reason}"
        )

    print("[6/7] Measuring sensitivity and rendering maps/diagnostic figures…", flush=True)
    sensitivity = run_sensitivity_analysis(
        context,
        design.optimized_elements,
        design.target_second_maximum_tt_jd,
        design.target_latitude_deg,
        design.target_longitude_deg,
    )
    _json_dump(output / "sensitivity.json", sensitivity)
    figure_index: dict[str, Any] = {"events": {}}
    for event in fixed_events[: args.max_maps]:
        figure_index["events"][event.event_id] = _plot_event_bundle(
            context, event, fixed_tracks[event.event_id], fixed_trajectory, output
        )
    figure_index["stability"] = _stability_figure(stability, output)
    _json_dump(output / "figure_index.json", figure_index)

    events_payload = {
        "schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ephemeris": {
            "kernel": context.kernel_path.name,
            "sha256": _sha256(context.kernel_path),
            "path": str(context.kernel_path),
        },
        "model_boundary": {
            "search": "restricted DE440s-forced dynamics",
            "stability": "fully coupled Newtonian four-body dynamics",
        },
        "optimized_elements": design.optimized_elements.to_dict(),
        "design_mode_opportunities": {
            "screened": len(design_opportunities),
            "feasible": sum(bool(item["feasible"]) for item in design_opportunities),
            "artifact": "design_mode_opportunities.json",
        },
        "design_mode": design_event.to_dict(),
        "fixed_system_events": [event.to_dict() for event in fixed_events],
        "validation": validation,
        "sensitivity": sensitivity,
        "stability_summary": {
            key: stability[key]
            for key in (
                "stable",
                "completed",
                "requested_years",
                "min_moon_moon_distance_km",
                "max_abs_relative_energy_error",
                "orbit_crossing_detected",
            )
        },
        "figures": figure_index,
    }
    _json_dump(output / "events.json", events_payload)
    _json_dump(
        output / "run_manifest.json",
        {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_required": ">=3.12",
            "command": "MPLBACKEND=Agg .venv/bin/python run_search.py --mode full",
            "ephemeris_sha256": events_payload["ephemeris"]["sha256"],
            "counts": {
                "real_eclipses": len(real_eclipses),
                "design_opportunities_screened": len(design_opportunities),
                "fixed_chains": len(fixed_events),
                "stability_samples": len(stability["time_years"]),
            },
            "packages": {
                name: package_version(name)
                for name in (
                    "astropy",
                    "cartopy",
                    "matplotlib",
                    "numpy",
                    "pandas",
                    "rebound",
                    "scipy",
                    "skyfield",
                )
            },
        },
    )

    print("[7/7] Building and verifying the self-contained HTML technical report…", flush=True)
    assumptions = [
        "The search forces the real Sun, Earth, and Moon to DE440s and suppresses second-moon back-reaction.",
        "The hypothetical moon is a smooth sphere; its roughly 0.6 s light time is iterated once, but synthetic aberration is omitted.",
        "Future UTC labels use the installed leap-second table. Validation uses TT/TDT because future UT1 and leap seconds are not knowable exactly.",
        "Partial boundaries in maps are 2-degree grid-derived envelopes; central-line distances use 120 s track sampling.",
        "The 1,000-year stability run omits Earth J2, tides, relativity, major planets, and lunar figure terms.",
        "No 100,000-year tidal-evolution claim is made; that mode remains optional.",
    ]
    _, report_path, report_receipt = build_report_artifact(
        fixed_events,
        validation,
        stability,
        sensitivity,
        outputs_dir=output,
        assumptions=assumptions,
    )
    central_path_errors = [
        item["coordinate_error_wgs84_km"]
        for item in validation["results"]
        # Timing-only (non-central) references publish no coordinates.
        if item["coordinate_error_wgs84_km"] is not None
    ]
    headline = _headline(fixed_events[0])
    summary = (
        headline
        + "\n\n"
        + f"Definitions: A={fixed_events[0].definition_a}; B={fixed_events[0].definition_b}; "
        + f"two separate totality intervals={fixed_events[0].two_totality_components}.\n"
        + f"Validation: max timing error {max(item['timing_error_tt_s'] for item in validation['results']):.3f} s; "
        + f"max path error {max(central_path_errors):.3f} km.\n"
        + f"Stability: {stability['stable']} for {stability['integrated_years']:.0f} years; "
        + f"minimum moon-moon distance {stability['min_moon_moon_distance_km']:.1f} km.\n"
        + f"Fixed-system qualifying events through 2100: {len(fixed_events)}.\n"
        + f"Report: {report_path.name}\n"
    )
    (output / "terminal_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    return {
        "headline": headline,
        "event_count": len(fixed_events),
        "output_dir": str(output),
        "report": str(report_path),
        "report_receipt": report_receipt,
    }


def run_validate(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    context = load_ephemeris(ROOT / config["model"]["ephemeris_cache"])
    report = build_validation_report(context)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(output / "validation_report.json", report)
    print(json.dumps(report, indent=2))


def run_design_only(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = load_ephemeris(ROOT / config["model"]["ephemeris_cache"])
    start = context.time_utc(config["model"]["epoch_utc"])
    end = context.time_utc(args.end or config["model"]["end_utc"])
    real_eclipses = enumerate_real_solar_eclipses(context, start, end)
    baseline = _baseline_elements(config)
    opportunities = screen_design_opportunities(
        context,
        real_eclipses,
        baseline,
        target_gap_minutes=float(config["optimization"]["design_totality_gap_minutes"]),
    )
    _json_dump(output / "design_mode_opportunities.json", opportunities)
    pd.DataFrame(opportunities).to_csv(
        output / "design_mode_opportunities.csv", index=False
    )
    design = optimize_earliest_design(
        context,
        real_eclipses,
        baseline,
        target_gap_minutes=float(config["optimization"]["design_totality_gap_minutes"]),
        run_global_search=not args.no_global,
        random_seed=int(config["optimization"]["random_seed"]),
    )
    event, _, _ = design_mode_event(context, design)
    _save_optimized_config(
        output / "optimized_system.yaml", design.optimized_elements, design.diagnostics
    )
    _save_optimized_config(
        ROOT / "config" / "optimized_system.yaml",
        design.optimized_elements,
        design.diagnostics,
    )
    _json_dump(output / "design_mode_event.json", event.to_dict())
    print(_headline(event))


def run_fixed_only(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = load_ephemeris(ROOT / config["model"]["ephemeris_cache"])
    start = context.time_utc(config["model"]["epoch_utc"])
    end = context.time_utc(args.end or config["model"]["end_utc"])
    optimized = output / "optimized_system.yaml"
    if not optimized.exists():
        optimized = ROOT / "config" / "optimized_system.yaml"
    elements = _load_optimized_elements(optimized)
    real_eclipses = enumerate_real_solar_eclipses(context, start, end)
    trajectory = integrate_restricted(
        context,
        elements,
        float(end.tt),
        rtol=1e-10,
        max_step_seconds=10_800.0,
    )
    events, _ = fixed_system_search(context, real_eclipses, trajectory)
    rows = [_flatten_event(event) for event in events]
    pd.DataFrame(rows).to_csv(output / "all_candidates.csv", index=False)
    pd.DataFrame(rows[:10]).to_csv(output / "ranked_top10.csv", index=False)
    _json_dump(output / "fixed_system_events.json", [event.to_dict() for event in events])
    print(f"Fixed-system qualifying events: {len(events)}")
    if events:
        print(_headline(events[0]))


def run_stability_only(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    context = load_ephemeris(ROOT / config["model"]["ephemeris_cache"])
    optimized = Path(args.output).resolve() / "optimized_system.yaml"
    if not optimized.exists():
        optimized = ROOT / "config" / "optimized_system.yaml"
    elements = _load_optimized_elements(optimized)
    result = run_stability_check(
        context,
        elements,
        config=StabilityConfig(
            years=args.stability_years,
            sample_interval_days=args.stability_years * 365.25 / 500.0,
        ),
    )
    _json_dump(Path(args.output).resolve() / "stability.json", result.to_dict())
    print(json.dumps({"stable": result.stable, "years": result.integrated_years}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "validate", "design", "fixed", "stability"),
        default="full",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--end", default=None, help="UTC ISO end date; defaults to 2101-01-01")
    parser.add_argument("--stability-years", type=float, default=1_000.0)
    parser.add_argument("--max-maps", type=int, default=10)
    parser.add_argument("--no-global", action="store_true", help="skip the coarse DE check")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "validate":
        run_validate(args)
    elif args.mode == "design":
        run_design_only(args)
    elif args.mode == "fixed":
        run_fixed_only(args)
    elif args.mode == "stability":
        run_stability_only(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
