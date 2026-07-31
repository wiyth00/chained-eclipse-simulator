"""Numerical audit for the three-structured-body rotational/tidal phase."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .enhanced_ephemeris import EnhancedEphemeris
from .ephemeris import load_ephemeris
from .moon_architecture import architecture_from_config, elements_from_config
from .rotational_dynamics import rotational_tide_config_from_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "bound_binary_giant.yaml"
DEFAULT_OUTPUT = ROOT / "docs" / "audits" / "lunar_spin_tides_10y_audit.json"
TRACKED_PREVIOUS_AUDIT = ROOT / "docs" / "audits" / "enhanced_physics_10y_audit.json"


def run_rotational_audit(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    years: int = 10,
    sample_step_seconds: float = 86_400.0,
    ias15_epsilon: float = 1.0e-10,
    convergence_days: int = 30,
    trajectory_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the primary audit and bounded tolerance/cadence comparisons."""

    path = Path(config_path)
    payload = yaml.safe_load(path.read_text())
    elements = elements_from_config(payload)
    architecture = architecture_from_config(payload)
    tides = rotational_tide_config_from_mapping(payload)
    if architecture is None or tides is None:
        raise ValueError("rotational audit requires binary and rotational scenarios")
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    start = _parse_utc(elements.epoch_utc)
    end = start.replace(year=start.year + years)
    ephemeris = EnhancedEphemeris(
        context,
        elements,
        _format_utc(end),
        sample_step_seconds=sample_step_seconds,
        ias15_epsilon=ias15_epsilon,
        binary_architecture=architecture,
        tide_config=tides,
        trajectory_cache_dir=trajectory_cache_dir,
    )
    sample_tt = ephemeris.epoch_tt_jd + ephemeris.seconds / 86_400.0
    primary = _trajectory_summary(ephemeris, sample_tt)
    checkpoints = {
        "one_year": _checkpoint(
            ephemeris,
            float(context.time_utc(_format_utc(start.replace(year=start.year + 1))).tt),
        ),
        f"{years}_years": _checkpoint(ephemeris, ephemeris.end_tt_jd),
    }
    previous = json.loads(TRACKED_PREVIOUS_AUDIT.read_text())
    result: dict[str, Any] = {
        "schema": 1,
        "system": "bound-binary-giant",
        "model_version": "0.2.0",
        "active_rotational_tide_scenario": tides.active_scenario,
        "start_utc": elements.epoch_utc,
        "end_utc": _format_utc(end),
        "trajectory_sample_step_seconds": sample_step_seconds,
        "ias15_epsilon": ias15_epsilon,
        "force_model": ephemeris.metadata["force_model"],
        "trajectory": primary,
        "checkpoints": checkpoints,
        "rotational_conservation_diagnostic": ephemeris.metadata[
            "rotational_conservation_diagnostic"
        ],
        "newtonian_energy_diagnostic": ephemeris.metadata[
            "newtonian_energy_diagnostic"
        ],
        "barycentric_initialization": {
            "four_body_frame": ephemeris.metadata["frame_contract"],
            "planetary_recentering": ephemeris.metadata["planetary_perturbers"][
                "com_recentering"
            ],
        },
        "comparison_to_previous_earth_only_10y_audit": _previous_comparison(
            checkpoints[f"{years}_years"],
            primary,
            previous,
        )
        if years == 10
        else None,
        "validation_scope": {
            "full_horizon_audit_years": years,
            "convergence_horizon_days": convergence_days,
            "endpoint_status": (
                "production result; not independently converged over the full "
                "ten-year horizon"
                if years == 10 and convergence_days < 3_650
                else "supported by full-horizon convergence comparisons"
            ),
            "strongest_full_horizon_check": (
                "The separately tracked one-year audit includes 365-day IAS15 "
                "tolerance and output-cadence comparisons."
                if years == 10
                else "This audit's convergence comparisons cover its full horizon."
            ),
        },
        "convergence": _run_convergence(
            context,
            elements,
            architecture,
            tides,
            start=start,
            days=convergence_days,
        )
        if convergence_days > 0
        else None,
        "interpretation": [
            "Positive-lag CTL tides are dissipative; mechanical energy should decrease.",
            "Production angular momentum is a bounded diagnostic because Earth J2 lacks its source-spin reaction.",
            "Strict isolated J2-off conservation is covered by test_rotational_conservation.py.",
            "Giant-moon Love number, time lag, inertia, spin, J2, and C22 are sensitivity assumptions.",
            "Permanent figures and physical libration remain disabled pending a reaction-aware attitude backend.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


def _trajectory_summary(
    ephemeris: EnhancedEphemeris,
    sample_tt: np.ndarray,
) -> dict[str, Any]:
    real = ephemeris.position("real_moon", sample_tt)
    second = ephemeris.position("second_moon", sample_tt)
    earth = ephemeris.position("earth", sample_tt)
    masses = ephemeris.metadata["masses_kg"]
    pair_mass = masses["real_moon"] + masses["second_moon"]
    barycenter = (
        masses["real_moon"] * real + masses["second_moon"] * second
    ) / pair_mass
    separation = np.linalg.norm(second - real, axis=1)
    outer_distance = np.linalg.norm(barycenter - earth, axis=1)
    eccentricity = _mutual_eccentricity(ephemeris, sample_tt)
    return {
        "mutual_separation_km": _range_summary(separation),
        "mutual_osculating_eccentricity": _range_summary(eccentricity),
        "moon_pair_barycenter_distance_from_earth_km": _range_summary(
            outer_distance
        ),
    }


def _mutual_eccentricity(
    ephemeris: EnhancedEphemeris,
    sample_tt: np.ndarray,
) -> np.ndarray:
    position = (
        ephemeris.position("second_moon", sample_tt)
        - ephemeris.position("real_moon", sample_tt)
    )
    velocity = (
        ephemeris.velocity("second_moon", sample_tt)
        - ephemeris.velocity("real_moon", sample_tt)
    )
    masses = ephemeris.metadata["masses_kg"]
    mu = ephemeris.metadata["gravitational_constant_km3_kg_s2"] * (
        masses["real_moon"] + masses["second_moon"]
    )
    angular_momentum = np.cross(position, velocity)
    radii = np.linalg.norm(position, axis=1)
    eccentricity_vectors = (
        np.cross(velocity, angular_momentum) / mu
        - position / radii[:, None]
    )
    return np.linalg.norm(eccentricity_vectors, axis=1)


def _checkpoint(ephemeris: EnhancedEphemeris, tt_jd: float) -> dict[str, Any]:
    rotation = ephemeris.earth_rotation_diagnostics(tt_jd)
    spin: dict[str, Any] = {}
    architecture = ephemeris.binary_architecture
    if architecture is None:
        raise ValueError("rotational checkpoint requires a binary architecture")
    masses = ephemeris.metadata["masses_kg"]
    mutual_mean_motion = np.sqrt(
        ephemeris.metadata["gravitational_constant_km3_kg_s2"]
        * (masses["real_moon"] + masses["second_moon"])
        / architecture.mutual_orbit.semimajor_axis_km**3
    )
    for body in ("earth", "real_moon", "second_moon"):
        vector = ephemeris.spin_vector_rad_s(body, tt_jd)
        rate = float(np.linalg.norm(vector))
        spin[body] = {
            "spin_vector_rad_s": vector.tolist(),
            "spin_rate_rad_s": rate,
        }
        if body != "earth":
            spin[body]["spin_to_initial_mutual_mean_motion"] = (
                rate / mutual_mean_motion
            )
    return {
        "earth_rotation_vs_real_moon_only_control": rotation,
        "spin": spin,
    }


def _run_convergence(
    context: Any,
    elements: Any,
    architecture: Any,
    tides: Any,
    *,
    start: datetime,
    days: int,
) -> dict[str, Any]:
    end = _format_utc(start + timedelta(days=days))
    reference = EnhancedEphemeris(
        context,
        elements,
        end,
        sample_step_seconds=21_600.0,
        ias15_epsilon=1.0e-10,
        binary_architecture=architecture,
        tide_config=tides,
        cache_trajectory=False,
    )
    tight = EnhancedEphemeris(
        context,
        elements,
        end,
        sample_step_seconds=21_600.0,
        ias15_epsilon=1.0e-12,
        binary_architecture=architecture,
        tide_config=tides,
        cache_trajectory=False,
    )
    fine_output = EnhancedEphemeris(
        context,
        elements,
        end,
        sample_step_seconds=3_600.0,
        ias15_epsilon=1.0e-10,
        binary_architecture=architecture,
        tide_config=tides,
        cache_trajectory=False,
    )
    return {
        "duration_days": days,
        "ias15_tolerance": _terminal_difference(reference, tight),
        "output_cadence": _terminal_difference(reference, fine_output),
    }


def _terminal_difference(
    reference: EnhancedEphemeris,
    comparison: EnhancedEphemeris,
) -> dict[str, Any]:
    epoch = reference.end_tt_jd
    return {
        "mutual_relative_position_difference_km": float(
            np.linalg.norm(
                (
                    reference.position("second_moon", epoch)
                    - reference.position("real_moon", epoch)
                )
                - (
                    comparison.position("second_moon", epoch)
                    - comparison.position("real_moon", epoch)
                )
            )
        ),
        "earth_spin_vector_difference_rad_s": float(
            np.linalg.norm(
                reference.spin_vector_rad_s("earth", epoch)
                - comparison.spin_vector_rad_s("earth", epoch)
            )
        ),
        "real_moon_spin_vector_difference_rad_s": float(
            np.linalg.norm(
                reference.spin_vector_rad_s("real_moon", epoch)
                - comparison.spin_vector_rad_s("real_moon", epoch)
            )
        ),
        "second_moon_spin_vector_difference_rad_s": float(
            np.linalg.norm(
                reference.spin_vector_rad_s("second_moon", epoch)
                - comparison.spin_vector_rad_s("second_moon", epoch)
            )
        ),
        "longitude_shift_difference_deg": abs(
            float(reference.longitude_offset_deg(epoch))
            - float(comparison.longitude_offset_deg(epoch))
        ),
    }


def _previous_comparison(
    final_checkpoint: dict[str, Any],
    trajectory: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    current_rotation = final_checkpoint[
        "earth_rotation_vs_real_moon_only_control"
    ]
    previous_rotation = previous["earth_rotation_vs_real_moon_only_control"]
    current_separation = trajectory["mutual_separation_km"]
    previous_separation = previous["mutual_separation_km"]
    return {
        "scope": (
            "current three-structured-body nominal CTL scenario minus the "
            "tracked Earth-only CTL enhanced audit"
        ),
        "delta_final_mean_solar_lod_ms": (
            current_rotation["delta_mean_solar_lod_ms"]
            - previous_rotation["final_delta_mean_solar_lod_ms"]
        ),
        "delta_final_ut1_s": (
            current_rotation["delta_ut1_s"]
            - previous_rotation["final_delta_ut1_s"]
        ),
        "delta_final_longitude_shift_deg": (
            current_rotation["longitude_shift_deg"]
            - previous_rotation["final_longitude_shift_deg"]
        ),
        "delta_final_pole_separation_arcsec": (
            current_rotation["pole_separation_arcsec"]
            - previous_rotation["final_pole_separation_arcsec"]
        ),
        "delta_minimum_mutual_separation_km": (
            current_separation["minimum"] - previous_separation["minimum"]
        ),
        "delta_maximum_mutual_separation_km": (
            current_separation["maximum"] - previous_separation["maximum"]
        ),
        "delta_final_mutual_separation_km": (
            current_separation["final"] - previous_separation["final"]
        ),
    }


def _range_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "final": float(values[-1]),
    }


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--sample-step-seconds", type=float, default=86_400.0)
    parser.add_argument("--ias15-epsilon", type=float, default=1.0e-10)
    parser.add_argument("--convergence-days", type=int, default=30)
    args = parser.parse_args()
    result = run_rotational_audit(
        args.config,
        years=args.years,
        sample_step_seconds=args.sample_step_seconds,
        ias15_epsilon=args.ias15_epsilon,
        convergence_days=args.convergence_days,
        trajectory_cache_dir=ROOT / "data" / "trajectories",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["run_rotational_audit"]
