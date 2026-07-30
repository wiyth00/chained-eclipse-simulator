"""Design-mode optimization for the earliest deliberately chained eclipse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ephemeris import EphemerisContext
from .eclipse_geometry import apparent_disk_geometry, maximum_surface_point
from .models import OrbitalElements, RealEclipse, Trajectory
from .orbital_dynamics import (
    analytic_design_seed,
    coarse_global_design,
    integrate_restricted,
    propagate_two_body,
    refine_design_shooting,
)


@dataclass(slots=True)
class DesignResult:
    real_eclipse: RealEclipse
    optimized_elements: OrbitalElements
    trajectory: Trajectory
    real_maximum_tt_jd: float
    target_second_maximum_tt_jd: float
    target_latitude_deg: float
    target_longitude_deg: float
    diagnostics: dict[str, object]

    def summary_dict(self) -> dict[str, object]:
        return {
            "real_eclipse": self.real_eclipse.to_dict(),
            "optimized_elements": self.optimized_elements.to_dict(),
            "real_maximum_tt_jd": self.real_maximum_tt_jd,
            "target_second_maximum_tt_jd": self.target_second_maximum_tt_jd,
            "target_latitude_deg": self.target_latitude_deg,
            "target_longitude_deg": self.target_longitude_deg,
            "diagnostics": self.diagnostics,
            "trajectory_metadata": self.trajectory.metadata,
        }


def screen_design_opportunities(
    context: EphemerisContext,
    real_eclipses: list[RealEclipse],
    baseline: OrbitalElements,
    *,
    target_gap_minutes: float = 12.0,
) -> list[dict[str, object]]:
    """Coarsely solve Ω, ω, and M0 for every real eclipse chronologically.

    The constructive two-body solution is the coarse Stage-2 search.  Once the
    earliest feasible event is found, only that event needs expensive forced-
    ephemeris shooting to establish the chronological optimum.
    """

    opportunities: list[dict[str, object]] = []
    for event in real_eclipses:
        real_tt = float(context.time_utc(event.maximum_utc).tt)
        if event.latitude_deg is None or event.longitude_deg is None:
            point = maximum_surface_point(
                context, context.tt_jd(real_tt), "real_moon"
            )
            if point is None:
                opportunities.append(
                    {
                        "event_id": event.event_id,
                        "real_maximum_utc": event.maximum_utc,
                        "feasible": False,
                        "reason": "no visible maximum-eclipse point recovered",
                    }
                )
                continue
            latitude, longitude = point[:2]
        else:
            latitude = float(event.latitude_deg)
            longitude = float(event.longitude_deg)
        target_tt = real_tt + target_gap_minutes * 60.0 / 86_400.0
        try:
            seed, target_position = analytic_design_seed(
                context, baseline, target_tt, latitude, longitude
            )
        except ValueError as exc:
            opportunities.append(
                {
                    "event_id": event.event_id,
                    "real_maximum_utc": event.maximum_utc,
                    "feasible": False,
                    "reason": str(exc),
                }
            )
            continue
        epoch_tt = float(context.time_utc(baseline.epoch_utc).tt)
        propagated = propagate_two_body(
            seed, (target_tt - epoch_tt) * 86_400.0
        )[:3]
        target_error = float(np.linalg.norm(propagated - target_position))
        geometry = apparent_disk_geometry(
            context,
            context.tt_jd(target_tt),
            latitude,
            longitude,
            0.0,
            "second_moon",
            position_provider=lambda _jd, position=target_position: position,
            body_radius_km=baseline.radius_km,
        )
        opportunities.append(
            {
                "event_id": event.event_id,
                "real_maximum_utc": event.maximum_utc,
                "real_eclipse_type": event.eclipse_type,
                "target_second_maximum_utc": context.tt_jd(target_tt).utc_iso(places=3),
                "target_latitude_deg": latitude,
                "target_longitude_deg": longitude,
                "target_gap_minutes": target_gap_minutes,
                "feasible": geometry.eclipse_type in {"total", "annular"},
                "coarse_second_type": geometry.eclipse_type,
                "coarse_second_magnitude": geometry.magnitude,
                "two_body_target_error_km": target_error,
                "longitude_ascending_node_deg": seed.longitude_ascending_node_deg,
                "argument_periapsis_deg": seed.argument_periapsis_deg,
                "mean_anomaly_deg": seed.mean_anomaly_deg,
                "reason": "constructive observer-aligned perigee seed",
            }
        )
    return opportunities


def optimize_earliest_design(
    context: EphemerisContext,
    real_eclipses: list[RealEclipse],
    baseline: OrbitalElements,
    *,
    target_gap_minutes: float = 12.0,
    run_global_search: bool = True,
    random_seed: int = 20260710,
) -> DesignResult:
    """Design a second-moon total eclipse after the first real eclipse.

    With Ω, ω, and M0 free, the first post-epoch real eclipse is constructively
    feasible.  The global search is retained as a coarse numerical check, then
    a perturbed/J2 shooting solve targets the chosen observer exactly.
    """

    first = next(
        event
        for event in real_eclipses
        if event.latitude_deg is not None and event.longitude_deg is not None
    )
    real_time = context.time_utc(first.maximum_utc)
    real_tt = float(real_time.tt)
    target_tt = real_tt + target_gap_minutes * 60.0 / 86_400.0
    latitude = float(first.latitude_deg)
    longitude = float(first.longitude_deg)
    analytic_seed, target_position = analytic_design_seed(
        context, baseline, target_tt, latitude, longitude
    )
    epoch_tt = float(context.time_utc(baseline.epoch_utc).tt)
    delta_seconds = (target_tt - epoch_tt) * 86_400.0

    diagnostics: dict[str, object] = {
        "design_degeneracy": (
            "Free orientation and phase make a conjunction constructively feasible at the "
            "first real eclipse; the date is a design consequence, not a random discovery."
        ),
        "target_gap_minutes": target_gap_minutes,
        "analytic_seed": analytic_seed.to_dict(),
    }
    seed_for_refinement = analytic_seed
    if run_global_search:
        global_candidate, global_diagnostics = coarse_global_design(
            baseline,
            target_position,
            delta_seconds,
            seed=random_seed,
        )
        analytic_position = propagate_two_body(analytic_seed, delta_seconds)[:3]
        analytic_error = float(np.linalg.norm(analytic_position - target_position))
        diagnostics["coarse_global"] = global_diagnostics
        diagnostics["analytic_two_body_error_km"] = analytic_error
        # A constructive seed is allowed to beat stochastic sampling; retain
        # the global result as independent evidence that the full search space
        # was exercised.
        if float(global_diagnostics["objective_km"]) < analytic_error:
            seed_for_refinement = global_candidate

    refined, short_trajectory, shooting = refine_design_shooting(
        context, seed_for_refinement, target_tt, target_position
    )
    diagnostics["shooting"] = shooting
    # Contacts extend beyond maximum; integrate a six-hour guard after target.
    final_trajectory = integrate_restricted(
        context,
        refined,
        target_tt + 6.0 / 24.0,
        include_j2=True,
        rtol=3e-12,
        max_step_seconds=1_800.0,
    )
    diagnostics["short_trajectory_metadata"] = short_trajectory.metadata
    diagnostics["target_state_icrf_km_km_s"] = final_trajectory.state(target_tt).tolist()
    return DesignResult(
        real_eclipse=first,
        optimized_elements=refined,
        trajectory=final_trajectory,
        real_maximum_tt_jd=real_tt,
        target_second_maximum_tt_jd=target_tt,
        target_latitude_deg=latitude,
        target_longitude_deg=longitude,
        diagnostics=diagnostics,
    )
