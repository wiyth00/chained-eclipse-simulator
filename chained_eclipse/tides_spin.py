"""Optional Earth-tide and alternate-Earth-rotation approximations.

This module deliberately does *not* claim to be a complete tidal ephemeris.  It
implements the instantaneous constant-time-lag (CTL) force of Mignard for a
tide raised on a spherical Earth, together with the equal-and-opposite force
and spin torque required by momentum conservation.  It also provides:

* a fixed-spin-axis integrator for deriving Earth rotation phase from sampled
  Earth-relative moon trajectories; and
* a circular, orbit-averaged secular audit that estimates the 30-year size of
  semimajor-axis and length-of-day changes.

The effective ``k2 * time_lag`` is calibrated to a nominal 38.2 mm/year lunar
recession at 384,400 km.  This gives about 181 s (equivalent to a roughly
604 s lag when ``k2 = 0.3``).  The calibration absorbs the real oceans into one
frequency-independent number; it is therefore an order-of-magnitude
counterfactual model, not an ocean-tide prediction for a second moon.

References
----------
Mignard, F. (1979), *The Evolution of the Lunar Orbit Revisited. I*.
Hut, P. (1981), *Tidal evolution in close binary systems*.

Units are kilometres, kilograms, seconds, and radians throughout unless a
name explicitly states otherwise.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    EARTH_MASS_KG,
    JULIAN_YEAR_DAYS,
    MU_EARTH_KM3_S2,
    REAL_MOON_MASS_KG,
    SECOND_MOON_MASS_KG,
    SECONDS_PER_DAY,
    WGS84_A_KM,
)
from .models import OrbitalElements


EARTH_SIDEREAL_DAY_S = 86_164.0905
EARTH_MEAN_SOLAR_DAY_S = 86_400.0
EARTH_POLAR_MOMENT_FACTOR = 0.3307
NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM = 384_400.0
NOMINAL_LUNAR_RECESSION_M_PER_YEAR = 0.0382

# The project fixes Earth's mass and GM independently.  Their ratio is the
# internally consistent gravitational constant for this optional model.
G_KM3_KG_S2 = MU_EARTH_KM3_S2 / EARTH_MASS_KG


@dataclass(frozen=True, slots=True)
class TidalBody:
    """A satellite used by the secular or sampled-trajectory tide model."""

    name: str
    mass_kg: float
    semimajor_axis_km: float

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("tidal body name must not be empty")
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError(f"{self.name} mass must be finite and positive")
        if not math.isfinite(self.semimajor_axis_km) or self.semimajor_axis_km <= 0.0:
            raise ValueError(f"{self.name} semimajor axis must be finite and positive")


@dataclass(frozen=True, slots=True)
class MignardPairEffect:
    """Instantaneous force, accelerations, and balancing primary spin torque."""

    force_on_satellite_kg_km_s2: np.ndarray
    primary_acceleration_km_s2: np.ndarray
    satellite_acceleration_km_s2: np.ndarray
    primary_spin_torque_kg_km2_s2: np.ndarray


@dataclass(slots=True)
class SpinPhaseHistory:
    """Fixed-axis Earth-spin solution derived from sampled moon trajectories."""

    time_seconds: list[float]
    spin_rate_rad_s: list[float]
    sidereal_day_s: list[float]
    mean_solar_lod_s: list[float]
    phase_offset_from_uniform_rad: list[float]
    axial_torque_kg_km2_s2: list[float]
    included_bodies: list[str]
    model_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RotationCorrection:
    """Two-moon minus one-moon Earth-rotation correction.

    A negative phase means that the two-moon Earth has rotated less far than
    the real-Moon-only baseline.  The corresponding Earth-fixed ground-track
    longitude correction is the negative of that phase, so positive longitude
    shifts are eastward.
    """

    time_seconds: list[float]
    delta_spin_rate_rad_s: list[float]
    delta_mean_solar_lod_ms: list[float]
    delta_ut1_s: list[float]
    earth_fixed_longitude_shift_deg: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SecularTidalSpinConfig:
    """Settings for the explicitly circular, orbit-averaged tide audit."""

    years: float = 30.0
    sample_interval_days: float = 30.0
    earth_mass_kg: float = EARTH_MASS_KG
    earth_radius_km: float = WGS84_A_KM
    earth_polar_moment_factor: float = EARTH_POLAR_MOMENT_FACTOR
    initial_sidereal_day_s: float = EARTH_SIDEREAL_DAY_S
    initial_mean_solar_day_s: float = EARTH_MEAN_SOLAR_DAY_S
    reference_recession_m_per_year: float = NOMINAL_LUNAR_RECESSION_M_PER_YEAR
    reference_semimajor_axis_km: float = NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM
    real_moon: TidalBody = field(
        default_factory=lambda: TidalBody(
            "real_moon", REAL_MOON_MASS_KG, NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM
        )
    )
    second_moon: TidalBody = field(
        default_factory=lambda: TidalBody("second_moon", SECOND_MOON_MASS_KG, 180_000.0)
    )
    relative_tolerance: float = 1.0e-11
    absolute_tolerance: float = 1.0e-12

    def validate(self) -> None:
        self.real_moon.validate()
        self.second_moon.validate()
        positive_values = {
            "years": self.years,
            "sample_interval_days": self.sample_interval_days,
            "earth_mass_kg": self.earth_mass_kg,
            "earth_radius_km": self.earth_radius_km,
            "earth_polar_moment_factor": self.earth_polar_moment_factor,
            "initial_sidereal_day_s": self.initial_sidereal_day_s,
            "initial_mean_solar_day_s": self.initial_mean_solar_day_s,
            "reference_recession_m_per_year": self.reference_recession_m_per_year,
            "reference_semimajor_axis_km": self.reference_semimajor_axis_km,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        initial_spin = 2.0 * math.pi / self.initial_sidereal_day_s
        reference_motion = mean_motion_rad_s(
            self.reference_semimajor_axis_km,
            self.real_moon.mass_kg,
            self.earth_mass_kg,
        )
        if initial_spin <= reference_motion:
            raise ValueError("the calibration orbit must lie outside synchronous orbit")


@dataclass(slots=True)
class SecularScenarioHistory:
    """One circular-equivalent secular tide history."""

    time_years: list[float]
    semimajor_axis_km: dict[str, list[float]]
    orbital_phase_offset_rad: dict[str, list[float]]
    orbital_timing_offset_s: dict[str, list[float]]
    spin_rate_rad_s: list[float]
    sidereal_day_s: list[float]
    mean_solar_lod_s: list[float]
    spin_phase_offset_rad: list[float]
    relative_axial_angular_momentum_error: list[float]
    mechanical_energy_change_j: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SecularTidalSpinResult:
    """Absolute histories plus the correction relative to real-Moon-only Earth."""

    calibration_k2_delta_t_s: float
    equivalent_time_lag_for_k2_0_3_s: float
    initial_recession_m_per_year: dict[str, float]
    two_moon: SecularScenarioHistory
    real_moon_only: SecularScenarioHistory
    counterfactual_rotation: RotationCorrection
    model: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def earth_polar_moment_kg_km2(
    mass_kg: float = EARTH_MASS_KG,
    radius_km: float = WGS84_A_KM,
    factor: float = EARTH_POLAR_MOMENT_FACTOR,
) -> float:
    """Return ``C = factor * M * R**2`` for the rotating Earth."""

    if min(mass_kg, radius_km, factor) <= 0.0:
        raise ValueError("Earth moment parameters must be positive")
    return factor * mass_kg * radius_km**2


def mean_motion_rad_s(
    semimajor_axis_km: float,
    satellite_mass_kg: float,
    earth_mass_kg: float = EARTH_MASS_KG,
) -> float:
    """Two-body mean motion using the module's internally consistent ``G``."""

    if min(semimajor_axis_km, satellite_mass_kg, earth_mass_kg) <= 0.0:
        raise ValueError("mean-motion inputs must be positive")
    return math.sqrt(G_KM3_KG_S2 * (earth_mass_kg + satellite_mass_kg) / semimajor_axis_km**3)


def circular_orbital_angular_momentum_kg_km2_s(
    semimajor_axis_km: float,
    satellite_mass_kg: float,
    earth_mass_kg: float = EARTH_MASS_KG,
) -> float:
    """Relative two-body angular momentum for a circular orbit."""

    mean_motion_rad_s(semimajor_axis_km, satellite_mass_kg, earth_mass_kg)
    reduced_mass = earth_mass_kg * satellite_mass_kg / (earth_mass_kg + satellite_mass_kg)
    gravitational_parameter = G_KM3_KG_S2 * (earth_mass_kg + satellite_mass_kg)
    return reduced_mass * math.sqrt(gravitational_parameter * semimajor_axis_km)


def calibrate_earth_k2_delta_t_s(
    *,
    recession_m_per_year: float = NOMINAL_LUNAR_RECESSION_M_PER_YEAR,
    semimajor_axis_km: float = NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM,
    moon_mass_kg: float = REAL_MOON_MASS_KG,
    earth_mass_kg: float = EARTH_MASS_KG,
    earth_radius_km: float = WGS84_A_KM,
    earth_spin_rad_s: float = 2.0 * math.pi / EARTH_SIDEREAL_DAY_S,
) -> float:
    """Calibrate ``k2 * Delta-t`` from a circular lunar recession rate.

    For a circular prograde orbit, the Mignard torque used here is

    ``T = 3 (k2 Delta-t) G m**2 R**5 (Omega - n) / a**6``.

    Setting ``dL/dt = T`` and ``dL/da = L/(2a)`` makes this function an exact
    inverse of :func:`circular_recession_rate_km_s` at the reference state.
    """

    values = (
        recession_m_per_year,
        semimajor_axis_km,
        moon_mass_kg,
        earth_mass_kg,
        earth_radius_km,
        earth_spin_rad_s,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("calibration inputs must be finite and positive")
    mean_motion = mean_motion_rad_s(semimajor_axis_km, moon_mass_kg, earth_mass_kg)
    frequency_difference = earth_spin_rad_s - mean_motion
    if frequency_difference <= 0.0:
        raise ValueError("calibration orbit must lie outside synchronous orbit")
    recession_km_s = (
        recession_m_per_year / 1_000.0 / (JULIAN_YEAR_DAYS * SECONDS_PER_DAY)
    )
    angular_momentum = circular_orbital_angular_momentum_kg_km2_s(
        semimajor_axis_km, moon_mass_kg, earth_mass_kg
    )
    required_torque = angular_momentum * recession_km_s / (2.0 * semimajor_axis_km)
    return (
        required_torque
        * semimajor_axis_km**6
        / (
            3.0
            * G_KM3_KG_S2
            * moon_mass_kg**2
            * earth_radius_km**5
            * frequency_difference
        )
    )


def mignard_pair_effect(
    relative_position_km: Sequence[float] | np.ndarray,
    relative_velocity_km_s: Sequence[float] | np.ndarray,
    *,
    satellite_mass_kg: float,
    primary_mass_kg: float = EARTH_MASS_KG,
    primary_radius_km: float = WGS84_A_KM,
    primary_spin_vector_rad_s: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    k2_delta_t_s: float,
) -> MignardPairEffect:
    """Evaluate the instantaneous Mignard CTL tide raised on the primary.

    ``relative_position`` and ``relative_velocity`` are satellite minus
    primary in one inertial Cartesian frame.  The returned accelerations are
    equal-and-opposite in force, and ``primary_spin_torque`` is exactly the
    negative of the orbital torque ``r x F``.  Applying all three quantities
    therefore conserves total linear and angular momentum to roundoff.

    This is the tide raised on Earth only.  Tides raised within either moon,
    permanent-figure torques, obliquity evolution, and frequency-dependent
    ocean response are outside this approximation.
    """

    position = _vector3(relative_position_km, "relative_position_km")
    velocity = _vector3(relative_velocity_km_s, "relative_velocity_km_s")
    spin = _vector3(primary_spin_vector_rad_s, "primary_spin_vector_rad_s")
    positive = {
        "satellite_mass_kg": satellite_mass_kg,
        "primary_mass_kg": primary_mass_kg,
        "primary_radius_km": primary_radius_km,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(k2_delta_t_s) or k2_delta_t_s < 0.0:
        raise ValueError("k2_delta_t_s must be finite and non-negative")
    radius = float(np.linalg.norm(position))
    if radius <= primary_radius_km:
        raise ValueError("the Mignard satellite position must lie outside the primary")

    coefficient = (
        -3.0
        * k2_delta_t_s
        * G_KM3_KG_S2
        * satellite_mass_kg**2
        * primary_radius_km**5
        / radius**10
    )
    bracket = (
        2.0 * position * float(np.dot(position, velocity))
        + radius**2 * (np.cross(position, spin) + velocity)
    )
    force_on_satellite = coefficient * bracket
    satellite_acceleration = force_on_satellite / satellite_mass_kg
    primary_acceleration = -force_on_satellite / primary_mass_kg
    spin_torque = -np.cross(position, force_on_satellite)
    return MignardPairEffect(
        force_on_satellite_kg_km_s2=force_on_satellite,
        primary_acceleration_km_s2=primary_acceleration,
        satellite_acceleration_km_s2=satellite_acceleration,
        primary_spin_torque_kg_km2_s2=spin_torque,
    )


def circular_recession_rate_km_s(
    body: TidalBody,
    earth_spin_rad_s: float,
    k2_delta_t_s: float,
    *,
    earth_mass_kg: float = EARTH_MASS_KG,
    earth_radius_km: float = WGS84_A_KM,
) -> float:
    """Return the circular orbit-averaged ``da/dt`` implied by the CTL force."""

    body.validate()
    if not math.isfinite(earth_spin_rad_s) or earth_spin_rad_s <= 0.0:
        raise ValueError("earth_spin_rad_s must be finite and positive")
    if not math.isfinite(k2_delta_t_s) or k2_delta_t_s <= 0.0:
        raise ValueError("k2_delta_t_s must be finite and positive")
    angular_momentum = circular_orbital_angular_momentum_kg_km2_s(
        body.semimajor_axis_km, body.mass_kg, earth_mass_kg
    )
    mean_motion = mean_motion_rad_s(
        body.semimajor_axis_km, body.mass_kg, earth_mass_kg
    )
    torque = (
        3.0
        * k2_delta_t_s
        * G_KM3_KG_S2
        * body.mass_kg**2
        * earth_radius_km**5
        * (earth_spin_rad_s - mean_motion)
        / body.semimajor_axis_km**6
    )
    return 2.0 * body.semimajor_axis_km * torque / angular_momentum


def integrate_sampled_earth_rotation(
    time_seconds: Sequence[float] | np.ndarray,
    relative_positions_km: Mapping[str, np.ndarray],
    relative_velocities_km_s: Mapping[str, np.ndarray],
    satellite_masses_kg: Mapping[str, float],
    *,
    included_bodies: Sequence[str] | None = None,
    initial_spin_rad_s: float = 2.0 * math.pi / EARTH_SIDEREAL_DAY_S,
    spin_axis: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    earth_mass_kg: float = EARTH_MASS_KG,
    earth_radius_km: float = WGS84_A_KM,
    earth_polar_moment_factor: float = EARTH_POLAR_MOMENT_FACTOR,
    initial_mean_solar_day_s: float = EARTH_MEAN_SOLAR_DAY_S,
    k2_delta_t_s: float | None = None,
) -> SpinPhaseHistory:
    """Integrate Earth spin and phase over sampled Earth-relative trajectories.

    Each sample interval freezes the linearly interpolated midpoint geometry.
    Because Mignard torque is affine in spin rate, the scalar spin ODE is then
    solved analytically across the interval.  This is stable for hour-scale
    samples and avoids subtracting two enormous absolute rotation angles.

    The spin axis and polar moment are held fixed.  Only torque projected on
    that axis changes rotation rate; transverse torque and polar motion are
    intentionally omitted.  The instantaneous force evaluator still returns
    the full three-dimensional balancing torque for a vector-spin integrator.
    """

    times = np.asarray(time_seconds, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("time_seconds must be a finite one-dimensional array of length >= 2")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_seconds must start at zero and increase strictly")
    if not math.isfinite(initial_spin_rad_s) or initial_spin_rad_s <= 0.0:
        raise ValueError("initial_spin_rad_s must be finite and positive")
    if not math.isfinite(initial_mean_solar_day_s) or initial_mean_solar_day_s <= 0.0:
        raise ValueError("initial_mean_solar_day_s must be finite and positive")
    axis = _vector3(spin_axis, "spin_axis")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("spin_axis must be nonzero")
    axis = axis / axis_norm
    names = list(relative_positions_km) if included_bodies is None else list(included_bodies)
    if not names:
        raise ValueError("at least one included body is required")
    validated_positions: dict[str, np.ndarray] = {}
    validated_velocities: dict[str, np.ndarray] = {}
    for name in names:
        if name not in relative_positions_km or name not in relative_velocities_km_s:
            raise ValueError(f"missing sampled state for {name}")
        if name not in satellite_masses_kg:
            raise ValueError(f"missing satellite mass for {name}")
        position = np.asarray(relative_positions_km[name], dtype=float)
        velocity = np.asarray(relative_velocities_km_s[name], dtype=float)
        if position.shape != (len(times), 3) or velocity.shape != (len(times), 3):
            raise ValueError(f"sampled state for {name} must have shape ({len(times)}, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError(f"sampled state for {name} must be finite")
        if not math.isfinite(satellite_masses_kg[name]) or satellite_masses_kg[name] <= 0.0:
            raise ValueError(f"satellite mass for {name} must be finite and positive")
        validated_positions[name] = position
        validated_velocities[name] = velocity

    effective_k2_lag = (
        calibrate_earth_k2_delta_t_s(
            earth_mass_kg=earth_mass_kg,
            earth_radius_km=earth_radius_km,
            earth_spin_rad_s=initial_spin_rad_s,
        )
        if k2_delta_t_s is None
        else float(k2_delta_t_s)
    )
    if not math.isfinite(effective_k2_lag) or effective_k2_lag <= 0.0:
        raise ValueError("k2_delta_t_s must be finite and positive")
    moment = earth_polar_moment_kg_km2(
        earth_mass_kg, earth_radius_km, earth_polar_moment_factor
    )
    spin = np.empty(len(times), dtype=float)
    phase_offset = np.empty(len(times), dtype=float)
    torque = np.empty(len(times), dtype=float)
    spin[0] = initial_spin_rad_s
    phase_offset[0] = 0.0
    torque[0] = _sample_axial_torque(
        0,
        spin[0],
        axis,
        names,
        validated_positions,
        validated_velocities,
        satellite_masses_kg,
        earth_mass_kg,
        earth_radius_km,
        effective_k2_lag,
    )

    for index, step in enumerate(np.diff(times)):
        midpoint_positions = {
            name: 0.5 * (validated_positions[name][index] + validated_positions[name][index + 1])
            for name in names
        }
        midpoint_velocities = {
            name: 0.5 * (
                validated_velocities[name][index] + validated_velocities[name][index + 1]
            )
            for name in names
        }
        intercept, slope = _axial_torque_affine_coefficients(
            axis,
            names,
            midpoint_positions,
            midpoint_velocities,
            satellite_masses_kg,
            earth_mass_kg,
            earth_radius_km,
            effective_k2_lag,
        )
        constant = intercept / moment
        linear = slope / moment
        spin[index + 1], phase_increment = _advance_affine_spin(
            spin[index], constant, linear, float(step)
        )
        phase_offset[index + 1] = (
            phase_offset[index] + phase_increment - initial_spin_rad_s * float(step)
        )
        torque[index + 1] = _sample_axial_torque(
            index + 1,
            spin[index + 1],
            axis,
            names,
            validated_positions,
            validated_velocities,
            satellite_masses_kg,
            earth_mass_kg,
            earth_radius_km,
            effective_k2_lag,
        )

    sidereal_day = 2.0 * math.pi / spin
    mean_solar_lod = initial_mean_solar_day_s * initial_spin_rad_s / spin
    return SpinPhaseHistory(
        time_seconds=times.tolist(),
        spin_rate_rad_s=spin.tolist(),
        sidereal_day_s=sidereal_day.tolist(),
        mean_solar_lod_s=mean_solar_lod.tolist(),
        phase_offset_from_uniform_rad=phase_offset.tolist(),
        axial_torque_kg_km2_s2=torque.tolist(),
        included_bodies=names,
        model_notes=[
            "Mignard constant-time-lag tide raised on a spherical Earth",
            "fixed Earth spin axis and fixed polar moment; only axial torque is integrated",
            "sampled trajectory is not modified by this post-processing function",
            "apply returned pair accelerations during N-body propagation for force-level coupling",
        ],
    )


def compare_rotation_histories(
    two_moon: SpinPhaseHistory,
    real_moon_only: SpinPhaseHistory,
    *,
    initial_spin_rad_s: float = 2.0 * math.pi / EARTH_SIDEREAL_DAY_S,
) -> RotationCorrection:
    """Return the added-moon correction suitable for overlay on real-Earth UT1."""

    times = np.asarray(two_moon.time_seconds, dtype=float)
    baseline_times = np.asarray(real_moon_only.time_seconds, dtype=float)
    if times.shape != baseline_times.shape or not np.array_equal(times, baseline_times):
        raise ValueError("rotation histories must use identical sample times")
    full_spin = np.asarray(two_moon.spin_rate_rad_s)
    baseline_spin = np.asarray(real_moon_only.spin_rate_rad_s)
    full_lod = np.asarray(two_moon.mean_solar_lod_s)
    baseline_lod = np.asarray(real_moon_only.mean_solar_lod_s)
    delta_phase = np.asarray(two_moon.phase_offset_from_uniform_rad) - np.asarray(
        real_moon_only.phase_offset_from_uniform_rad
    )
    return RotationCorrection(
        time_seconds=times.tolist(),
        delta_spin_rate_rad_s=(full_spin - baseline_spin).tolist(),
        delta_mean_solar_lod_ms=((full_lod - baseline_lod) * 1_000.0).tolist(),
        delta_ut1_s=(delta_phase / initial_spin_rad_s).tolist(),
        earth_fixed_longitude_shift_deg=(-np.degrees(delta_phase)).tolist(),
    )


def simulate_secular_tidal_spin(
    config: SecularTidalSpinConfig | None = None,
) -> SecularTidalSpinResult:
    """Run the 30-year circular-equivalent tidal/spin magnitude audit.

    Earth spin is calculated algebraically from the conserved initial axial
    angular momentum at every derivative evaluation.  Consequently the
    reported spin-plus-orbit angular momentum error is limited to floating
    point roundoff, rather than ODE drift.

    The full and baseline scenarios start with the same Earth rotation rate.
    Their difference is the correction that can be overlaid on Skyfield's
    real-Earth rotation without counting the real Moon's tidal braking twice.
    """

    settings = SecularTidalSpinConfig() if config is None else config
    settings.validate()
    initial_spin = 2.0 * math.pi / settings.initial_sidereal_day_s
    k2_lag = calibrate_earth_k2_delta_t_s(
        recession_m_per_year=settings.reference_recession_m_per_year,
        semimajor_axis_km=settings.reference_semimajor_axis_km,
        moon_mass_kg=settings.real_moon.mass_kg,
        earth_mass_kg=settings.earth_mass_kg,
        earth_radius_km=settings.earth_radius_km,
        earth_spin_rad_s=initial_spin,
    )
    two_moon = _integrate_secular_scenario(
        settings,
        (settings.real_moon, settings.second_moon),
        initial_spin,
        k2_lag,
    )
    baseline = _integrate_secular_scenario(
        settings,
        (settings.real_moon,),
        initial_spin,
        k2_lag,
    )
    rotation = compare_rotation_histories(
        _secular_spin_history(two_moon, ("real_moon", "second_moon")),
        _secular_spin_history(baseline, ("real_moon",)),
        initial_spin_rad_s=initial_spin,
    )
    initial_rates = {
        body.name: circular_recession_rate_km_s(
            body,
            initial_spin,
            k2_lag,
            earth_mass_kg=settings.earth_mass_kg,
            earth_radius_km=settings.earth_radius_km,
        )
        * JULIAN_YEAR_DAYS
        * SECONDS_PER_DAY
        * 1_000.0
        for body in (settings.real_moon, settings.second_moon)
    }
    return SecularTidalSpinResult(
        calibration_k2_delta_t_s=k2_lag,
        equivalent_time_lag_for_k2_0_3_s=k2_lag / 0.3,
        initial_recession_m_per_year=initial_rates,
        two_moon=two_moon,
        real_moon_only=baseline,
        counterfactual_rotation=rotation,
        model={
            "name": "optional circular-equivalent Mignard CTL tide/spin audit",
            "calibration": (
                f"real-Moon recession {settings.reference_recession_m_per_year:.4f} m/year "
                f"at {settings.reference_semimajor_axis_km:.1f} km"
            ),
            "angular_momentum": "Earth spin plus circular orbital angular momenta, axial scalar",
            "earth_polar_moment_factor": settings.earth_polar_moment_factor,
            "baseline": "real Moon only; subtract before overlaying correction on real-Earth UT1",
        },
        warnings=[
            "This is not a full dissipative N-body ephemeris.",
            "The secular audit assumes circular, coplanar prograde orbits and fixed Earth moment.",
            "It omits eccentricity and inclination tides, moon tides/spins, ocean modes, and polar motion.",
            "A second moon's ocean-tide response need not share the calibration inferred from our Moon.",
            "Use mignard_pair_effect during N-body integration for force-level eclipse refinement.",
        ],
    )


def _integrate_secular_scenario(
    config: SecularTidalSpinConfig,
    bodies: tuple[TidalBody, ...],
    initial_spin: float,
    k2_lag: float,
) -> SecularScenarioHistory:
    from scipy.integrate import solve_ivp

    duration_s = config.years * JULIAN_YEAR_DAYS * SECONDS_PER_DAY
    sample_step_s = config.sample_interval_days * SECONDS_PER_DAY
    sample_times = np.arange(0.0, duration_s, sample_step_s)
    sample_times = np.append(sample_times, duration_s)
    moment = earth_polar_moment_kg_km2(
        config.earth_mass_kg,
        config.earth_radius_km,
        config.earth_polar_moment_factor,
    )
    initial_axes = np.asarray([body.semimajor_axis_km for body in bodies], dtype=float)
    initial_motions = np.asarray(
        [
            mean_motion_rad_s(body.semimajor_axis_km, body.mass_kg, config.earth_mass_kg)
            for body in bodies
        ]
    )
    initial_orbital_momenta = np.asarray(
        [
            circular_orbital_angular_momentum_kg_km2_s(
                body.semimajor_axis_km, body.mass_kg, config.earth_mass_kg
            )
            for body in bodies
        ]
    )
    total_angular_momentum = moment * initial_spin + float(np.sum(initial_orbital_momenta))

    # State: semimajor axes, Earth phase offset, then orbital phase offsets.
    initial_state = np.concatenate((initial_axes, (0.0,), np.zeros(len(bodies))))

    def derivative(_time_s: float, state: np.ndarray) -> np.ndarray:
        axes = state[: len(bodies)]
        orbital_momenta = np.asarray(
            [
                circular_orbital_angular_momentum_kg_km2_s(
                    axis, body.mass_kg, config.earth_mass_kg
                )
                for axis, body in zip(axes, bodies, strict=True)
            ]
        )
        spin = (total_angular_momentum - float(np.sum(orbital_momenta))) / moment
        axis_rates = np.asarray(
            [
                circular_recession_rate_km_s(
                    TidalBody(body.name, body.mass_kg, float(axis)),
                    spin,
                    k2_lag,
                    earth_mass_kg=config.earth_mass_kg,
                    earth_radius_km=config.earth_radius_km,
                )
                for axis, body in zip(axes, bodies, strict=True)
            ]
        )
        motions = np.asarray(
            [
                mean_motion_rad_s(axis, body.mass_kg, config.earth_mass_kg)
                for axis, body in zip(axes, bodies, strict=True)
            ]
        )
        return np.concatenate(
            (axis_rates, (spin - initial_spin,), motions - initial_motions)
        )

    solution = solve_ivp(
        derivative,
        (0.0, duration_s),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=config.relative_tolerance,
        atol=config.absolute_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"secular tide integration failed: {solution.message}")
    axes = solution.y[: len(bodies)]
    spin_phase = solution.y[len(bodies)]
    orbital_phase = solution.y[len(bodies) + 1 :]
    orbital_momenta = np.asarray(
        [
            [
                circular_orbital_angular_momentum_kg_km2_s(
                    axis, body.mass_kg, config.earth_mass_kg
                )
                for axis in body_axes
            ]
            for body, body_axes in zip(bodies, axes, strict=True)
        ]
    )
    spin = (total_angular_momentum - np.sum(orbital_momenta, axis=0)) / moment
    reconstructed = moment * spin + np.sum(orbital_momenta, axis=0)
    angular_momentum_error = (reconstructed - total_angular_momentum) / abs(
        total_angular_momentum
    )
    energy = 0.5 * moment * spin**2
    for body, body_axes in zip(bodies, axes, strict=True):
        energy -= (
            G_KM3_KG_S2
            * config.earth_mass_kg
            * body.mass_kg
            / (2.0 * body_axes)
        )
    energy_change_j = (energy - energy[0]) * 1.0e6
    sidereal_day = 2.0 * math.pi / spin
    mean_solar_lod = config.initial_mean_solar_day_s * initial_spin / spin
    timing_offsets = {
        body.name: (orbital_phase[index] / initial_motions[index]).tolist()
        for index, body in enumerate(bodies)
    }
    return SecularScenarioHistory(
        time_years=(solution.t / (JULIAN_YEAR_DAYS * SECONDS_PER_DAY)).tolist(),
        semimajor_axis_km={
            body.name: axes[index].tolist() for index, body in enumerate(bodies)
        },
        orbital_phase_offset_rad={
            body.name: orbital_phase[index].tolist() for index, body in enumerate(bodies)
        },
        orbital_timing_offset_s=timing_offsets,
        spin_rate_rad_s=spin.tolist(),
        sidereal_day_s=sidereal_day.tolist(),
        mean_solar_lod_s=mean_solar_lod.tolist(),
        spin_phase_offset_rad=spin_phase.tolist(),
        relative_axial_angular_momentum_error=angular_momentum_error.tolist(),
        mechanical_energy_change_j=energy_change_j.tolist(),
    )


def _secular_spin_history(
    history: SecularScenarioHistory,
    included_bodies: tuple[str, ...],
) -> SpinPhaseHistory:
    time_seconds = (
        np.asarray(history.time_years) * JULIAN_YEAR_DAYS * SECONDS_PER_DAY
    ).tolist()
    return SpinPhaseHistory(
        time_seconds=time_seconds,
        spin_rate_rad_s=history.spin_rate_rad_s,
        sidereal_day_s=history.sidereal_day_s,
        mean_solar_lod_s=history.mean_solar_lod_s,
        phase_offset_from_uniform_rad=history.spin_phase_offset_rad,
        axial_torque_kg_km2_s2=[0.0] * len(time_seconds),
        included_bodies=list(included_bodies),
        model_notes=["converted from circular-equivalent secular audit"],
    )


def _sample_axial_torque(
    index: int,
    spin: float,
    axis: np.ndarray,
    names: list[str],
    positions: Mapping[str, np.ndarray],
    velocities: Mapping[str, np.ndarray],
    masses: Mapping[str, float],
    earth_mass: float,
    earth_radius: float,
    k2_lag: float,
) -> float:
    return float(
        sum(
            np.dot(
                axis,
                mignard_pair_effect(
                    positions[name][index],
                    velocities[name][index],
                    satellite_mass_kg=masses[name],
                    primary_mass_kg=earth_mass,
                    primary_radius_km=earth_radius,
                    primary_spin_vector_rad_s=axis * spin,
                    k2_delta_t_s=k2_lag,
                ).primary_spin_torque_kg_km2_s2,
            )
            for name in names
        )
    )


def _axial_torque_affine_coefficients(
    axis: np.ndarray,
    names: list[str],
    positions: Mapping[str, np.ndarray],
    velocities: Mapping[str, np.ndarray],
    masses: Mapping[str, float],
    earth_mass: float,
    earth_radius: float,
    k2_lag: float,
) -> tuple[float, float]:
    intercept = 0.0
    slope = 0.0
    for name in names:
        zero_spin = mignard_pair_effect(
            positions[name],
            velocities[name],
            satellite_mass_kg=masses[name],
            primary_mass_kg=earth_mass,
            primary_radius_km=earth_radius,
            primary_spin_vector_rad_s=np.zeros(3),
            k2_delta_t_s=k2_lag,
        ).primary_spin_torque_kg_km2_s2
        unit_spin = mignard_pair_effect(
            positions[name],
            velocities[name],
            satellite_mass_kg=masses[name],
            primary_mass_kg=earth_mass,
            primary_radius_km=earth_radius,
            primary_spin_vector_rad_s=axis,
            k2_delta_t_s=k2_lag,
        ).primary_spin_torque_kg_km2_s2
        axial_zero = float(np.dot(axis, zero_spin))
        intercept += axial_zero
        slope += float(np.dot(axis, unit_spin - zero_spin))
    return intercept, slope


def _advance_affine_spin(
    spin: float,
    constant_rad_s2: float,
    linear_per_s: float,
    step_s: float,
) -> tuple[float, float]:
    """Advance ``dOmega/dt = constant + linear*Omega`` and integrate Omega."""

    exponent = linear_per_s * step_s
    initial_derivative = constant_rad_s2 + linear_per_s * spin
    if abs(exponent) < 1.0e-8:
        final_spin = spin + initial_derivative * step_s * (1.0 + 0.5 * exponent)
        phase = (
            spin * step_s
            + 0.5 * initial_derivative * step_s**2
            + initial_derivative * linear_per_s * step_s**3 / 6.0
        )
        return final_spin, phase
    equilibrium = -constant_rad_s2 / linear_per_s
    exponential_minus_one = math.expm1(exponent)
    final_spin = spin + (spin - equilibrium) * exponential_minus_one
    phase = equilibrium * step_s + (spin - equilibrium) * exponential_minus_one / linear_per_s
    return final_spin, phase


def _vector3(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def write_secular_tidal_audit(
    output_dir: str | Path,
    *,
    config: SecularTidalSpinConfig | None = None,
    dpi: int = 180,
) -> dict[str, Any]:
    """Run the secular audit and write strict JSON plus a static PNG figure."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result = simulate_secular_tidal_spin(config)
    full = result.two_moon
    baseline = result.real_moon_only
    correction = result.counterfactual_rotation
    real_recession_m = (
        full.semimajor_axis_km["real_moon"][-1]
        - full.semimajor_axis_km["real_moon"][0]
    ) * 1_000.0
    second_recession_m = (
        full.semimajor_axis_km["second_moon"][-1]
        - full.semimajor_axis_km["second_moon"][0]
    ) * 1_000.0
    summary = {
        "period_years": full.time_years[-1],
        "real_moon_recession_m": real_recession_m,
        "second_moon_recession_m": second_recession_m,
        "two_moon_lod_change_ms": (
            full.mean_solar_lod_s[-1] - full.mean_solar_lod_s[0]
        )
        * 1_000.0,
        "real_moon_only_lod_change_ms": (
            baseline.mean_solar_lod_s[-1] - baseline.mean_solar_lod_s[0]
        )
        * 1_000.0,
        "added_lod_ms": correction.delta_mean_solar_lod_ms[-1],
        "added_ut1_offset_s": correction.delta_ut1_s[-1],
        "earth_fixed_longitude_shift_deg_east": (
            correction.earth_fixed_longitude_shift_deg[-1]
        ),
        "equatorial_ground_shift_km_east": (
            math.radians(correction.earth_fixed_longitude_shift_deg[-1]) * WGS84_A_KM
        ),
        "second_moon_orbital_timing_offset_s": (
            full.orbital_timing_offset_s["second_moon"][-1]
        ),
        "maximum_abs_relative_axial_angular_momentum_error": max(
            map(abs, full.relative_axial_angular_momentum_error)
        ),
    }
    payload = {
        "schema_version": "1.0",
        "summary": summary,
        "audit": result.to_dict(),
    }
    json_path = destination / "tidal_spin_audit.json"
    figure_path = destination / "tidal_spin_audit.png"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_secular_tidal_audit(result, figure_path, dpi=dpi)
    return payload


def plot_secular_tidal_audit(
    result: SecularTidalSpinResult,
    output_path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Render a compact three-panel summary of the secular tidal audit."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full = result.two_moon
    baseline = result.real_moon_only
    correction = result.counterfactual_rotation
    years = np.asarray(full.time_years)
    real_axis_change_m = (
        np.asarray(full.semimajor_axis_km["real_moon"])
        - full.semimajor_axis_km["real_moon"][0]
    ) * 1_000.0
    second_axis_change_m = (
        np.asarray(full.semimajor_axis_km["second_moon"])
        - full.semimajor_axis_km["second_moon"][0]
    ) * 1_000.0
    full_lod_change_ms = (
        np.asarray(full.mean_solar_lod_s) - full.mean_solar_lod_s[0]
    ) * 1_000.0
    baseline_lod_change_ms = (
        np.asarray(baseline.mean_solar_lod_s) - baseline.mean_solar_lod_s[0]
    ) * 1_000.0
    added_ut1_lag_s = -np.asarray(correction.delta_ut1_s)

    ink = "#17212B"
    muted = "#5D6873"
    grid = "#D7DDE2"
    real_color = "#176B87"
    second_color = "#D97706"
    correction_color = "#5B4B8A"
    figure = Figure(figsize=(11.2, 8.2), facecolor="white", constrained_layout=False)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.34})
    figure.subplots_adjust(left=0.105, right=0.94, top=0.88, bottom=0.12)
    figure.suptitle(
        f"Tidal evolution over {years[-1]:g} years",
        x=0.105,
        ha="left",
        fontsize=18,
        fontweight=400,
        color=ink,
    )
    figure.text(
        0.105,
        0.905,
        "Circular-equivalent Mignard tide · two-moon Earth compared with a real-Moon-only baseline",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=muted,
    )

    axis = axes[0]
    axis.plot(years, second_axis_change_m, color=second_color, linewidth=2.3)
    axis.plot(years, real_axis_change_m, color=real_color, linewidth=2.0)
    axis.set_ylabel("Orbit expansion (m)", color=ink)
    _direct_label(axis, years[-1], second_axis_change_m[-1], "Second moon", second_color)
    _direct_label(axis, years[-1], real_axis_change_m[-1], "Real Moon", real_color)

    axis = axes[1]
    axis.plot(years, full_lod_change_ms, color=second_color, linewidth=2.3)
    axis.plot(years, baseline_lod_change_ms, color=real_color, linewidth=2.0)
    axis.fill_between(
        years,
        baseline_lod_change_ms,
        full_lod_change_ms,
        color=second_color,
        alpha=0.11,
        linewidth=0.0,
    )
    axis.set_ylabel("Length-of-day increase (ms)", color=ink)
    _direct_label(axis, years[-1], full_lod_change_ms[-1], "Both moons", second_color)
    _direct_label(
        axis,
        years[-1],
        baseline_lod_change_ms[-1],
        "Real Moon only",
        real_color,
    )

    axis = axes[2]
    axis.plot(years, added_ut1_lag_s, color=correction_color, linewidth=2.4)
    axis.fill_between(years, 0.0, added_ut1_lag_s, color=correction_color, alpha=0.09)
    axis.set_ylabel("Added UT1 lag (s)", color=ink)
    axis.set_xlabel("Years after the orbital epoch", color=ink)
    longitude = correction.earth_fixed_longitude_shift_deg[-1]
    ground_km = math.radians(longitude) * WGS84_A_KM
    axis.annotate(
        f"{added_ut1_lag_s[-1]:.3f} s lag\n{longitude:.5f}° E · {ground_km:.2f} km at equator",
        xy=(years[-1], added_ut1_lag_s[-1]),
        xytext=(-12, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.5,
        color=correction_color,
    )

    for axis in axes:
        axis.grid(axis="y", color=grid, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(grid)
        axis.tick_params(colors=muted, labelsize=9.5)
        axis.margins(x=0.02, y=0.12)
    figure.text(
        0.105,
        0.035,
        (
            "Optional magnitude audit only: fixed Earth spin axis and moment; circular coplanar orbits; "
            "no frequency-dependent oceans, satellite tides, or polar motion."
        ),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color=muted,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _direct_label(axis: Any, x: float, y: float, label: str, color: str) -> None:
    axis.annotate(
        f"{label}  {y:.3f}",
        xy=(x, y),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9.2,
        color=color,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the standalone secular tidal audit."""

    parser = argparse.ArgumentParser(
        description="Run the optional two-moon secular tide and Earth-spin audit."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coupled/tidal_spin_audit",
        help="Directory for tidal_spin_audit.json and tidal_spin_audit.png",
    )
    parser.add_argument("--years", type=float, default=30.0)
    parser.add_argument("--sample-days", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional baseline or optimized config supplying second-moon mass and orbit.",
    )
    arguments = parser.parse_args(argv)
    config_kwargs: dict[str, Any] = {
        "years": arguments.years,
        "sample_interval_days": arguments.sample_days,
    }
    if arguments.config is not None:
        from .moon_scaling import load_elements

        elements: OrbitalElements = load_elements(arguments.config)
        config_kwargs["second_moon"] = TidalBody(
            "second_moon",
            float(elements.mass_kg),
            elements.semimajor_axis_km,
        )
    config = SecularTidalSpinConfig(
        **config_kwargs,
    )
    payload = write_secular_tidal_audit(
        arguments.output_dir,
        config=config,
        dpi=arguments.dpi,
    )
    summary = payload["summary"]
    print(f"Tidal audit complete: {summary['period_years']:.3f} years")
    print(
        "Second-moon recession: "
        f"{summary['second_moon_recession_m']:.3f} m; "
        f"added LOD: {summary['added_lod_ms']:.3f} ms"
    )
    print(
        "Alternate-Earth correction: "
        f"UT1 {summary['added_ut1_offset_s']:.3f} s; "
        f"longitude {summary['earth_fixed_longitude_shift_deg_east']:.5f}° east"
    )
    print(f"Wrote {Path(arguments.output_dir).resolve()}")
    return 0


__all__ = [
    "EARTH_MEAN_SOLAR_DAY_S",
    "EARTH_POLAR_MOMENT_FACTOR",
    "EARTH_SIDEREAL_DAY_S",
    "G_KM3_KG_S2",
    "MignardPairEffect",
    "NOMINAL_LUNAR_RECESSION_M_PER_YEAR",
    "NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM",
    "RotationCorrection",
    "SecularScenarioHistory",
    "SecularTidalSpinConfig",
    "SecularTidalSpinResult",
    "SpinPhaseHistory",
    "TidalBody",
    "calibrate_earth_k2_delta_t_s",
    "circular_orbital_angular_momentum_kg_km2_s",
    "circular_recession_rate_km_s",
    "compare_rotation_histories",
    "earth_polar_moment_kg_km2",
    "integrate_sampled_earth_rotation",
    "mean_motion_rad_s",
    "mignard_pair_effect",
    "plot_secular_tidal_audit",
    "simulate_secular_tidal_spin",
    "write_secular_tidal_audit",
]


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
