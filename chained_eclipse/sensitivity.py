"""Numerical and osculating-element sensitivity checks for the best design."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from pyproj import Geod
from scipy.optimize import minimize_scalar

from .constants import SECONDS_PER_DAY
from .eclipse_geometry import central_point_hypothetical, hypothetical_shadow_state, solve_local_circumstances
from .ephemeris import EphemerisContext
from .models import OrbitalElements
from .orbital_dynamics import integrate_restricted


@dataclass(slots=True)
class SensitivityCase:
    case: str
    perturbation: str
    global_maximum_tt_jd: float
    global_maximum_shift_s: float
    central_latitude_deg: float | None
    central_longitude_deg: float | None
    central_track_shift_km: float | None
    local_eclipse_type: str
    local_magnitude: float
    local_maximum_utc: str
    local_maximum_shift_s: float
    integrator_rtol: float
    max_step_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_sensitivity_analysis(
    context: EphemerisContext,
    elements: OrbitalElements,
    target_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
) -> dict[str, object]:
    """Compare integration tolerances and small initial-element perturbations."""

    cases: list[tuple[str, str, OrbitalElements, float, float]] = [
        ("reference", "none", elements, 3e-12, 1_800.0),
        ("integration_loose", "rtol=1e-9, max step=6 h", elements, 1e-9, 21_600.0),
        (
            "node_plus_1e-4_deg",
            "Omega +0.0001 deg",
            replace(elements, longitude_ascending_node_deg=elements.longitude_ascending_node_deg + 1e-4),
            3e-12,
            1_800.0,
        ),
        (
            "periapsis_plus_1e-4_deg",
            "omega +0.0001 deg",
            replace(elements, argument_periapsis_deg=elements.argument_periapsis_deg + 1e-4),
            3e-12,
            1_800.0,
        ),
        (
            "mean_anomaly_plus_1e-4_deg",
            "M0 +0.0001 deg",
            replace(elements, mean_anomaly_deg=elements.mean_anomaly_deg + 1e-4),
            3e-12,
            1_800.0,
        ),
        (
            "semimajor_axis_plus_1_km",
            "a +1 km",
            replace(elements, semimajor_axis_km=elements.semimajor_axis_km + 1.0),
            3e-12,
            1_800.0,
        ),
        (
            "inclination_plus_0p001_deg",
            "i +0.001 deg",
            replace(elements, inclination_deg=elements.inclination_deg + 0.001),
            3e-12,
            1_800.0,
        ),
    ]
    raw: list[dict[str, object]] = []
    for name, perturbation, candidate, rtol, max_step in cases:
        trajectory = integrate_restricted(
            context,
            candidate,
            target_tt_jd + 5.0 / 24.0,
            rtol=rtol,
            max_step_seconds=max_step,
        )

        def miss(offset_seconds: float) -> float:
            tt = target_tt_jd + offset_seconds / SECONDS_PER_DAY
            return hypothetical_shadow_state(
                context,
                context.tt_jd(tt),
                trajectory.position(tt),
                moon_radius_km=candidate.radius_km,
            ).axis_miss_km

        optimized = minimize_scalar(
            miss,
            bounds=(-1_800.0, 1_800.0),
            method="bounded",
            options={"xatol": 0.05},
        )
        global_tt = target_tt_jd + float(optimized.x) / SECONDS_PER_DAY
        central = central_point_hypothetical(
            context,
            context.tt_jd(global_tt),
            trajectory.position(global_tt),
            moon_radius_km=candidate.radius_km,
        )
        local = solve_local_circumstances(
            context,
            target_tt_jd,
            latitude_deg,
            longitude_deg,
            "second_moon",
            position_provider=trajectory.position,
            body_radius_km=candidate.radius_km,
            bracket_step_seconds=60.0,
        )
        raw.append(
            {
                "case": name,
                "perturbation": perturbation,
                "global_tt": global_tt,
                "central": central,
                "local": local,
                "rtol": rtol,
                "max_step": max_step,
            }
        )

    reference = raw[0]
    ref_central = reference["central"]
    ref_local = reference["local"]
    assert ref_central is not None
    geod = Geod(ellps="WGS84")
    results: list[SensitivityCase] = []
    for item in raw:
        central = item["central"]
        local = item["local"]
        if central is None:
            central_lat = central_lon = track_shift = None
        else:
            central_lat, central_lon = float(central[0]), float(central[1])
            _, _, distance_m = geod.inv(
                float(ref_central[1]),
                float(ref_central[0]),
                central_lon,
                central_lat,
            )
            track_shift = float(distance_m / 1_000.0)
        local_shift = (
            context.time_utc(local.maximum_utc).tt - context.time_utc(ref_local.maximum_utc).tt
        ) * SECONDS_PER_DAY
        results.append(
            SensitivityCase(
                case=str(item["case"]),
                perturbation=str(item["perturbation"]),
                global_maximum_tt_jd=float(item["global_tt"]),
                global_maximum_shift_s=float(
                    (float(item["global_tt"]) - float(reference["global_tt"])) * SECONDS_PER_DAY
                ),
                central_latitude_deg=central_lat,
                central_longitude_deg=central_lon,
                central_track_shift_km=track_shift,
                local_eclipse_type=local.eclipse_type,
                local_magnitude=local.magnitude,
                local_maximum_utc=local.maximum_utc,
                local_maximum_shift_s=float(local_shift),
                integrator_rtol=float(item["rtol"]),
                max_step_seconds=float(item["max_step"]),
            )
        )
    return {
        "method": (
            "One-at-a-time epoch-element perturbations; every case is re-integrated in the "
            "DE440s-forced restricted model and re-solves the local maximum."
        ),
        "cases": [result.to_dict() for result in results],
    }
