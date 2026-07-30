"""Enhanced coupled trajectory used by the 30-year eclipse-climate search.

This module is the integration point for the project's optional higher-fidelity
counterfactual model.  It combines:

* active Newtonian point masses for all eight major planets;
* Earth's :math:`J_2` quadrupole;
* full first-order post-Newtonian interactions through REBOUNDx ``gr_full``;
* a vector-spin constant-time-lag Earth tide; and
* a differential Earth attitude history used to move eclipse ground tracks.

DE440s supplies all real-body states only at the 2026 epoch.  The alternate
system is then propagated freely by REBOUND IAS15.  Skyfield still supplies
the observed real-Earth orientation reference; modeled spin phase and pole
differences from a matched massless-second-moon control are composed onto that
reference.  This is deliberately explicit rather than being presented as a
new fitted ephemeris or observational Earth-orientation solution.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from skyfield.framelib import itrs

from .constants import SECONDS_PER_DAY, WGS84_A_KM
from .coupled_eclipse import CoupledEphemeris, PARTICLE_NAMES
from .enhanced_dynamics import (
    EnhancedDynamicsConfig,
    attach_reboundx_enhanced_forces,
)
from .ephemeris import EphemerisContext
from .models import OrbitalElements
from .moon_architecture import BinaryMoonArchitecture
from .planetary_dynamics import build_planetary_simulation
from .tides_spin import (
    EARTH_MEAN_SOLAR_DAY_S,
    EARTH_POLAR_MOMENT_FACTOR,
    EARTH_SIDEREAL_DAY_S,
    G_KM3_KG_S2,
    calibrate_earth_k2_delta_t_s,
    mignard_pair_effect,
)


@dataclass(frozen=True, slots=True)
class NBodyTideConfig:
    """Settings for the coupled Earth-tide force and spin model."""

    enabled: bool = True
    satellite_names: tuple[str, ...] = ("real_moon", "second_moon")
    earth_spin_axis_icrf: tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_sidereal_day_s: float = EARTH_SIDEREAL_DAY_S
    earth_love_number_k2: float = 0.3
    earth_polar_moment_factor: float = EARTH_POLAR_MOMENT_FACTOR
    evolve_spin: bool = True
    k2_delta_t_s: float | None = None

    def validate(self) -> None:
        if not self.satellite_names:
            raise ValueError("satellite_names must not be empty")
        if len(set(self.satellite_names)) != len(self.satellite_names):
            raise ValueError("satellite_names must be unique")
        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("earth_spin_axis_icrf must be a finite three-vector")
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("earth_spin_axis_icrf must be non-zero")
        if not math.isfinite(self.initial_sidereal_day_s) or self.initial_sidereal_day_s <= 0.0:
            raise ValueError("initial_sidereal_day_s must be finite and positive")
        if not math.isfinite(self.earth_love_number_k2) or self.earth_love_number_k2 <= 0.0:
            raise ValueError("earth_love_number_k2 must be finite and positive")
        if (
            not math.isfinite(self.earth_polar_moment_factor)
            or self.earth_polar_moment_factor <= 0.0
        ):
            raise ValueError("earth_polar_moment_factor must be finite and positive")
        if self.k2_delta_t_s is not None and (
            not math.isfinite(self.k2_delta_t_s) or self.k2_delta_t_s <= 0.0
        ):
            raise ValueError("k2_delta_t_s must be finite and positive when supplied")

    def normalized_spin_axis(self) -> np.ndarray:
        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        return axis / np.linalg.norm(axis)


def earth_tidal_accelerations(
    simulation: Any,
    config: NBodyTideConfig,
    *,
    k2_delta_t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Mignard pair accelerations and the balancing Earth-spin torque."""

    config.validate()
    accelerations = np.zeros((simulation.N, 3), dtype=float)
    total_spin_torque = np.zeros(3, dtype=float)
    if not config.enabled:
        return accelerations, total_spin_torque

    earth = simulation.particles["earth"]
    earth_index = int(earth.index)
    earth_position = np.asarray((earth.x, earth.y, earth.z), dtype=float)
    earth_velocity = np.asarray((earth.vx, earth.vy, earth.vz), dtype=float)
    spin_vector = (
        config.normalized_spin_axis()
        * (2.0 * math.pi / config.initial_sidereal_day_s)
    )
    for name in config.satellite_names:
        satellite = simulation.particles[name]
        relative_position = (
            np.asarray((satellite.x, satellite.y, satellite.z), dtype=float)
            - earth_position
        )
        relative_velocity = (
            np.asarray((satellite.vx, satellite.vy, satellite.vz), dtype=float)
            - earth_velocity
        )
        effect = mignard_pair_effect(
            relative_position,
            relative_velocity,
            satellite_mass_kg=float(satellite.m),
            primary_mass_kg=float(earth.m),
            primary_spin_vector_rad_s=spin_vector,
            k2_delta_t_s=k2_delta_t_s,
        )
        accelerations[earth_index] += effect.primary_acceleration_km_s2
        accelerations[int(satellite.index)] += effect.satellite_acceleration_km_s2
        total_spin_torque += effect.primary_spin_torque_kg_km2_s2
    return accelerations, total_spin_torque


def attach_earth_tides(
    simulation: Any,
    config: NBodyTideConfig | None = None,
) -> dict[str, Any]:
    """Attach the Earth-tide force, composing with an existing REBOUND callback."""

    settings = NBodyTideConfig() if config is None else config
    settings.validate()
    earth = simulation.particles["earth"]
    for name in settings.satellite_names:
        simulation.particles[name]
    k2_lag = (
        calibrate_earth_k2_delta_t_s(
            moon_mass_kg=float(simulation.particles["real_moon"].m),
            earth_mass_kg=float(earth.m),
            earth_spin_rad_s=2.0 * math.pi / settings.initial_sidereal_day_s,
        )
        if settings.k2_delta_t_s is None
        else float(settings.k2_delta_t_s)
    )

    existing_field = simulation._additional_forces
    existing_force = None
    if bool(existing_field):
        existing_force = getattr(simulation, "_afp", None)
        if existing_force is None:  # pragma: no cover - relevant to compiled plugins
            address = ctypes.cast(existing_field, ctypes.c_void_p).value
            existing_force = type(existing_field)(address)

    if settings.enabled:
        earth_index = int(earth.index)
        earth_mass = float(earth.m)
        satellite_indices = tuple(
            int(simulation.particles[name].index) for name in settings.satellite_names
        )
        satellite_masses = tuple(
            float(simulation.particles[name].m) for name in settings.satellite_names
        )
        spin = settings.normalized_spin_axis() * (
            2.0 * math.pi / settings.initial_sidereal_day_s
        )
        spin_x, spin_y, spin_z = map(float, spin)
        force_scale = -3.0 * k2_lag * G_KM3_KG_S2 * WGS84_A_KM**5

        def combined_force(simulation_pointer: Any) -> None:
            if existing_force is not None:
                existing_force(simulation_pointer)
            active = simulation_pointer.contents
            # This scalar hot path is algebraically identical to
            # mignard_pair_effect(), but avoids several short-lived NumPy
            # arrays on every IAS15 force evaluation.
            primary = active.particles[earth_index]
            for satellite_index, satellite_mass in zip(
                satellite_indices,
                satellite_masses,
                strict=True,
            ):
                satellite = active.particles[satellite_index]
                x = satellite.x - primary.x
                y = satellite.y - primary.y
                z = satellite.z - primary.z
                vx = satellite.vx - primary.vx
                vy = satellite.vy - primary.vy
                vz = satellite.vz - primary.vz
                radius_squared = x * x + y * y + z * z
                radial_dot = x * vx + y * vy + z * vz
                cross_x = y * spin_z - z * spin_y
                cross_y = z * spin_x - x * spin_z
                cross_z = x * spin_y - y * spin_x
                coefficient = (
                    force_scale
                    * satellite_mass
                    * satellite_mass
                    / radius_squared**5
                )
                force_x = coefficient * (
                    2.0 * x * radial_dot + radius_squared * (cross_x + vx)
                )
                force_y = coefficient * (
                    2.0 * y * radial_dot + radius_squared * (cross_y + vy)
                )
                force_z = coefficient * (
                    2.0 * z * radial_dot + radius_squared * (cross_z + vz)
                )
                satellite.ax += force_x / satellite_mass
                satellite.ay += force_y / satellite_mass
                satellite.az += force_z / satellite_mass
                primary.ax -= force_x / earth_mass
                primary.ay -= force_y / earth_mass
                primary.az -= force_z / earth_mass

        simulation.additional_forces = combined_force
        simulation.force_is_velocity_dependent = 1

    return {
        "enabled": settings.enabled,
        "model": "Mignard constant-time-lag tide raised on a spherical Earth",
        "satellites": list(settings.satellite_names),
        "k2_delta_t_s": k2_lag,
        "equivalent_time_lag_for_k2_0_3_s": k2_lag / 0.3,
        "initial_sidereal_day_s": settings.initial_sidereal_day_s,
        "earth_spin_axis_icrf": list(settings.earth_spin_axis_icrf),
        "force_coupling": (
            "orbital force uses the initial Earth spin; sampled torque is post-processed "
            "into the alternate rotation phase"
        ),
        "composed_with_existing_additional_forces": existing_force is not None,
        "omissions": [
            "frequency-dependent ocean response",
            "tides and spin evolution inside either moon",
            "transverse Earth-spin torque, polar motion, and changing inertia",
            "iterating the sub-millisecond 30-year spin change back into the tidal force",
        ],
    }


def attach_reboundx_earth_tides(
    simulation: Any,
    config: NBodyTideConfig | None = None,
) -> dict[str, Any]:
    """Attach REBOUNDx's compiled vector-spin constant-time-lag tide."""

    try:
        import reboundx
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("The coupled tidal-spin model requires reboundx") from exc

    settings = NBodyTideConfig() if config is None else config
    settings.validate()
    earth = simulation.particles["earth"]
    k2_lag = (
        calibrate_earth_k2_delta_t_s(
            moon_mass_kg=float(simulation.particles["real_moon"].m),
            earth_mass_kg=float(earth.m),
            earth_spin_rad_s=2.0 * math.pi / settings.initial_sidereal_day_s,
        )
        if settings.k2_delta_t_s is None
        else float(settings.k2_delta_t_s)
    )
    # REBOUNDx's Eggleton/Hut normalization produces twice the circular
    # recession of the Mignard convention used by calibrate_earth_k2_delta_t_s
    # for the same k2*tau.  The factor of one half is verified numerically
    # against 38.2 mm/year at 384,400 km in the test suite.
    physical_lag = k2_lag / (2.0 * settings.earth_love_number_k2)
    if settings.enabled:
        extras = getattr(simulation, "_extras_ref", None)
        if extras is None:
            extras = reboundx.Extras(simulation)
        tides = extras.load_force("tides_spin")
        extras.add_force(tides)
        earth.params["k2"] = settings.earth_love_number_k2
        earth.params["I"] = (
            settings.earth_polar_moment_factor * float(earth.m) * float(earth.r) ** 2
        )
        earth.params["Omega"] = (
            settings.normalized_spin_axis()
            * (2.0 * math.pi / settings.initial_sidereal_day_s)
        )
        earth.params["tau"] = physical_lag
        if settings.evolve_spin:
            extras.initialize_spin_ode(tides)

    return {
        "enabled": settings.enabled,
        "model": "REBOUNDx tides_spin vector constant-time-lag equilibrium tide",
        "implementation": "compiled C force plus coupled Earth-spin ODE",
        "structured_body": "earth",
        "point_mass_perturbers": "all other active bodies",
        "earth_love_number_k2": settings.earth_love_number_k2,
        "k2_delta_t_s": k2_lag,
        "constant_time_lag_s": physical_lag,
        "reboundx_normalization_factor": 0.5,
        "calibration_target": "38.2 mm/year circular real-Moon recession at 384,400 km",
        "initial_sidereal_day_s": settings.initial_sidereal_day_s,
        "earth_polar_moment_factor": settings.earth_polar_moment_factor,
        "earth_spin_axis_icrf": list(settings.earth_spin_axis_icrf),
        "spin_evolved_inside_nbody": bool(settings.enabled and settings.evolve_spin),
        "omissions": [
            "frequency-dependent ocean and solid-Earth response",
            "tides, deformation, and spin evolution inside either moon",
            "time-varying Earth inertia and atmosphere/ocean angular momentum",
        ],
    }


class EnhancedEphemeris(CoupledEphemeris):
    """Interpolated trajectory for the enhanced counterfactual dynamics mode."""

    def __init__(
        self,
        context: EphemerisContext,
        elements: OrbitalElements,
        end_utc: str,
        *,
        sample_step_seconds: float = 3_600.0,
        ias15_epsilon: float = 1.0e-10,
        dynamics_config: EnhancedDynamicsConfig | None = None,
        tide_config: NBodyTideConfig | None = None,
        binary_architecture: BinaryMoonArchitecture | None = None,
        include_pluto: bool = False,
        cache_trajectory: bool = True,
        trajectory_cache_dir: str | Path | None = None,
    ) -> None:
        if not math.isfinite(sample_step_seconds) or sample_step_seconds <= 0.0:
            raise ValueError("sample_step_seconds must be finite and positive")
        self.context = context
        self.elements = elements
        self.binary_architecture = binary_architecture
        self.epoch_tt_jd = float(context.time_utc(elements.epoch_utc).tt)
        self.end_tt_jd = float(context.time_utc(end_utc).tt)
        duration_seconds = (self.end_tt_jd - self.epoch_tt_jd) * SECONDS_PER_DAY
        if duration_seconds <= 0.0:
            raise ValueError("end_utc must be later than the orbital epoch")
        self.seconds = _sample_seconds(duration_seconds, sample_step_seconds)

        dynamics_settings = (
            EnhancedDynamicsConfig() if dynamics_config is None else dynamics_config
        )
        tide_settings = NBodyTideConfig() if tide_config is None else tide_config
        simulation, planetary_metadata = build_planetary_simulation(
            context,
            elements,
            binary_architecture=binary_architecture,
            ias15_epsilon=ias15_epsilon,
            include_pluto=include_pluto,
        )
        correction_metadata = attach_reboundx_enhanced_forces(
            simulation, dynamics_settings
        )
        tide_metadata = attach_reboundx_earth_tides(simulation, tide_settings)

        control_simulation = None
        if tide_settings.enabled and tide_settings.evolve_spin:
            # A real-Solar-System control avoids counting the observed real
            # Moon's tidal braking twice when the differential orientation is
            # composed onto Skyfield ITRS.  In binary mode the alternate real
            # Moon has a radically different state, so this is deliberately a
            # counterfactual reference rather than a same-geometry control.
            control_elements = replace(elements, mass_kg=0.0)
            control_simulation, _ = build_planetary_simulation(
                context,
                control_elements,
                ias15_epsilon=ias15_epsilon,
                include_pluto=include_pluto,
            )
            attach_reboundx_enhanced_forces(control_simulation, dynamics_settings)
            attach_reboundx_earth_tides(control_simulation, tide_settings)

        initial_spin_vector = (
            tide_settings.normalized_spin_axis()
            * (2.0 * math.pi / tide_settings.initial_sidereal_day_s)
        )
        cache_path = _trajectory_cache_path(
            context,
            elements,
            end_utc=end_utc,
            sample_step_seconds=sample_step_seconds,
            ias15_epsilon=ias15_epsilon,
            dynamics_config=dynamics_settings,
            tide_config=tide_settings,
            binary_architecture=binary_architecture,
            include_pluto=include_pluto,
            cache_dir=trajectory_cache_dir,
        )
        cache_hit = False
        if cache_trajectory and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cached_seconds = np.asarray(cached["seconds"], dtype=float)
                    if np.array_equal(cached_seconds, self.seconds):
                        positions = np.asarray(cached["positions"], dtype=float)
                        full_spin_vectors = np.asarray(
                            cached["full_spin_vectors"], dtype=float
                        )
                        control_spin_vectors = np.asarray(
                            cached["control_spin_vectors"], dtype=float
                        )
                        initial_newtonian_energy = float(
                            cached["initial_newtonian_energy"]
                        )
                        final_newtonian_energy = float(
                            cached["final_newtonian_energy"]
                        )
                        cache_hit = (
                            positions.shape
                            == (len(self.seconds), len(PARTICLE_NAMES), 3)
                            and full_spin_vectors.shape == (len(self.seconds), 3)
                            and control_spin_vectors.shape == (len(self.seconds), 3)
                        )
            except (KeyError, OSError, ValueError):
                # A partial or stale cache is never allowed to poison a run;
                # simply recompute it under the same content-addressed name.
                cache_hit = False
        if not cache_hit:
            initial_newtonian_energy = float(simulation.energy())
            positions = np.empty(
                (len(self.seconds), len(PARTICLE_NAMES), 3), dtype=float
            )
            full_spin_vectors = np.empty((len(self.seconds), 3), dtype=float)
            control_spin_vectors = np.empty_like(full_spin_vectors)
            for index, seconds in enumerate(self.seconds):
                simulation.integrate(float(seconds), exact_finish_time=1)
                for body_index, name in enumerate(PARTICLE_NAMES):
                    particle = simulation.particles[name]
                    positions[index, body_index] = (
                        particle.x,
                        particle.y,
                        particle.z,
                    )
                full_spin_vectors[index] = _particle_spin_vector(
                    simulation.particles["earth"],
                    fallback=initial_spin_vector,
                )
                if control_simulation is None:
                    control_spin_vectors[index] = initial_spin_vector
                else:
                    control_simulation.integrate(float(seconds), exact_finish_time=1)
                    control_spin_vectors[index] = _particle_spin_vector(
                        control_simulation.particles["earth"],
                        fallback=initial_spin_vector,
                    )
            final_newtonian_energy = float(simulation.energy())
            if cache_trajectory:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npz")
                np.savez(
                    temporary,
                    seconds=self.seconds,
                    positions=positions,
                    full_spin_vectors=full_spin_vectors,
                    control_spin_vectors=control_spin_vectors,
                    initial_newtonian_energy=initial_newtonian_energy,
                    final_newtonian_energy=final_newtonian_energy,
                )
                temporary.replace(cache_path)

        self._splines = {
            name: CubicSpline(self.seconds, positions[:, index, :], axis=0)
            for index, name in enumerate(PARTICLE_NAMES)
        }
        full_spin_rates = np.linalg.norm(full_spin_vectors, axis=1)
        control_spin_rates = np.linalg.norm(control_spin_vectors, axis=1)
        delta_spin_rate = full_spin_rates - control_spin_rates
        delta_phase = cumulative_trapezoid(
            delta_spin_rate,
            self.seconds,
            initial=0.0,
        )
        longitude_shift = -np.degrees(delta_phase)
        initial_spin_rate = float(np.linalg.norm(initial_spin_vector))
        full_lod = EARTH_MEAN_SOLAR_DAY_S * initial_spin_rate / full_spin_rates
        control_lod = EARTH_MEAN_SOLAR_DAY_S * initial_spin_rate / control_spin_rates
        delta_lod_ms = (full_lod - control_lod) * 1_000.0
        delta_ut1_s = delta_phase / initial_spin_rate
        full_spin_axes = full_spin_vectors / full_spin_rates[:, None]
        control_spin_axes = control_spin_vectors / control_spin_rates[:, None]
        pole_separation_rad = np.arccos(
            np.clip(np.sum(full_spin_axes * control_spin_axes, axis=1), -1.0, 1.0)
        )

        self._phase_offset_spline = CubicSpline(self.seconds, delta_phase)
        self._longitude_offset_spline = CubicSpline(self.seconds, longitude_shift)
        self._full_spin_axis_spline = CubicSpline(
            self.seconds, full_spin_axes, axis=0
        )
        self._control_spin_axis_spline = CubicSpline(
            self.seconds, control_spin_axes, axis=0
        )

        self.metadata = {
            **planetary_metadata,
            "dynamics_model": "enhanced",
            "integrator": "REBOUND IAS15",
            "ias15_epsilon": ias15_epsilon,
            "sample_step_seconds": sample_step_seconds,
            "sample_count": len(self.seconds),
            "trajectory_cache": {
                "enabled": cache_trajectory,
                "path": str(cache_path.resolve()),
                "hit": cache_hit,
            },
            "start_utc": elements.epoch_utc,
            "end_utc": end_utc,
            "initial_architecture": (
                None
                if binary_architecture is None
                else binary_architecture.to_dict()
            ),
            "force_model": {
                "newtonian": planetary_metadata.get("force_model"),
                "earth_tides_and_spin": tide_metadata,
                "earth_j2_and_first_post_newtonian": correction_metadata,
            },
            "newtonian_energy_diagnostic": {
                "initial": initial_newtonian_energy,
                "final": final_newtonian_energy,
                "fractional_change": (
                    (final_newtonian_energy - initial_newtonian_energy)
                    / abs(initial_newtonian_energy)
                ),
                "interpretation": (
                    "Not a total conserved-energy error: the diagnostic omits J2/1PN "
                    "Hamiltonian terms and the tide is dissipative."
                ),
            },
            "alternate_earth_rotation": {
                "method": (
                    "Earth's three-component spin is evolved inside both the full enhanced "
                    "N-body system and a real-Solar-System control; their "
                    "spin phase and pole differences are applied to Skyfield ITRS"
                ),
                "control": (
                    "same epoch, planets, J2, 1PN, and tide law; the second moon is "
                    "massless and the real Moon begins at its DE440 state"
                ),
                "comparison_scope": (
                    "alternate bound-binary Earth minus the real-Moon-only reference"
                    if binary_architecture is not None
                    else "independent-second-moon Earth minus the real-Moon-only reference"
                ),
                "skyfield_reference_retained": (
                    "observed real-Earth UT1, precession, and nutation as the zero-order reference"
                ),
                "orientation_application": (
                    "integrated differential spin phase plus the minimal 3-D rotation "
                    "mapping the control spin pole to the full-system spin pole"
                ),
                "final_delta_spin_rate_rad_s": float(delta_spin_rate[-1]),
                "final_delta_mean_solar_lod_ms": float(delta_lod_ms[-1]),
                "final_delta_ut1_s": float(delta_ut1_s[-1]),
                "final_longitude_shift_deg": float(longitude_shift[-1]),
                "maximum_abs_longitude_shift_deg": float(np.max(np.abs(longitude_shift))),
                "final_pole_separation_arcsec": float(
                    np.degrees(pole_separation_rad[-1]) * 3_600.0
                ),
                "maximum_pole_separation_arcsec": float(
                    np.degrees(np.max(pole_separation_rad)) * 3_600.0
                ),
            },
            "omissions": [
                "asteroids, trans-Neptunian objects, and planetary figure terms other than Earth J2",
                "lunar and second-moon permanent-figure forces",
                "post-1PN relativity, solar frame dragging, and solar quadrupole",
                "frequency-dependent ocean/solid-Earth tide response and satellite tides",
                "J2 reaction torque on Earth's spin, free nutation, and atmosphere/ocean angular momentum",
                "a refitted alternate-planet ephemeris and observational Earth-orientation series",
            ],
            "limitations": [
                "DE440s supplies real-body states only at the 2026 epoch; the alternate system then evolves freely.",
                "Major planets are point masses; Mars through Neptune use system barycentres and system GMs.",
                "Relativity uses REBOUNDx gr_full first-order post-Newtonian dynamics; higher orders and spin terms are omitted.",
                "The constant-time-lag tide is calibrated to present lunar recession and is not an ocean model.",
                "Earth's tidal spin vector is coupled inside REBOUNDx; its difference from a real-Moon-only control is overlaid on Skyfield's observed orientation.",
                "Eclipse shadows omit lunar limb topography and atmospheric enlargement of Earth's lunar-eclipse umbra.",
            ],
        }

    def longitude_offset_deg(self, tt_jd: float | np.ndarray) -> np.ndarray:
        """Return the enhanced-minus-real-Earth spin-phase longitude correction."""

        return np.asarray(
            self._longitude_offset_spline(self._seconds(tt_jd)),
            dtype=float,
        )

    def rotation_matrix_icrf_to_itrs(self, tt_jd: float) -> np.ndarray:
        """Return Skyfield ITRS with differential spin phase and pole applied."""

        seconds = float(self._seconds(tt_jd))
        full_axis = _unit_vector(self._full_spin_axis_spline(seconds))
        control_axis = _unit_vector(self._control_spin_axis_spline(seconds))
        pole_rotation = _rotation_from_to(control_axis, full_axis)
        phase_rotation = _axis_angle_rotation(
            full_axis,
            float(self._phase_offset_spline(seconds)),
        )
        base = np.asarray(itrs.rotation_at(self.time(tt_jd)), dtype=float)
        # Body-to-inertial orientation is phase_rotation @ pole_rotation @
        # base.T.  Transpose that product for the requested inertial-to-body
        # matrix.
        return base @ pole_rotation.T @ phase_rotation.T


def _sample_seconds(duration_seconds: float, step_seconds: float) -> np.ndarray:
    samples = np.arange(0.0, duration_seconds, step_seconds, dtype=float)
    if len(samples) == 0 or not math.isclose(
        float(samples[-1]), duration_seconds, rel_tol=0.0, abs_tol=1.0e-6
    ):
        samples = np.append(samples, duration_seconds)
    return samples


def _trajectory_cache_path(
    context: EphemerisContext,
    elements: OrbitalElements,
    *,
    end_utc: str,
    sample_step_seconds: float,
    ias15_epsilon: float,
    dynamics_config: EnhancedDynamicsConfig,
    tide_config: NBodyTideConfig,
    binary_architecture: BinaryMoonArchitecture | None,
    include_pluto: bool,
    cache_dir: str | Path | None,
) -> Path:
    """Return a content-addressed path for one enhanced trajectory.

    Every setting that can change the integrated state is part of the key.
    Eclipse-detector settings are intentionally excluded because they only
    consume the trajectory.
    """

    kernel = Path(context.kernel_path)
    kernel_stat = kernel.stat()
    tide_payload = asdict(tide_config)
    tide_payload["satellite_names"] = list(tide_config.satellite_names)
    tide_payload["earth_spin_axis_icrf"] = list(tide_config.earth_spin_axis_icrf)
    payload = {
        "schema": 4,
        "elements": elements.to_dict(),
        "binary_architecture": (
            None if binary_architecture is None else binary_architecture.to_dict()
        ),
        "end_utc": end_utc,
        "sample_step_seconds": sample_step_seconds,
        "ias15_epsilon": ias15_epsilon,
        "dynamics": dynamics_config.to_dict(),
        "tides": tide_payload,
        "include_pluto": include_pluto,
        "kernel": {
            "name": kernel.name,
            "size": kernel_stat.st_size,
            "mtime_ns": kernel_stat.st_mtime_ns,
        },
        "force_model": (
            "major-planets+gravitational_harmonics+gr_full+"
            "tides_spin-full-minus-real-moon-control"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    directory = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else context.cache_dir.parent / "trajectories"
    )
    return directory / f"enhanced_{digest}.npz"


def _particle_spin_vector(particle: Any, *, fallback: np.ndarray) -> np.ndarray:
    try:
        spin = particle.params["Omega"]
    except (AttributeError, KeyError):
        return np.asarray(fallback, dtype=float)
    return np.asarray((spin.x, spin.y, spin.z), dtype=float)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("rotation axis must be a finite non-zero three-vector")
    return result / norm


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = _unit_vector(axis)
    x, y, z = unit
    cross_matrix = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=float,
    )
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    return np.eye(3) + sine * cross_matrix + (1.0 - cosine) * (
        cross_matrix @ cross_matrix
    )


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = _unit_vector(source)
    second = _unit_vector(target)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine < 1.0e-15:
        if cosine > 0.0:
            return np.eye(3)
        trial = np.asarray((1.0, 0.0, 0.0))
        if abs(first[0]) > 0.8:
            trial = np.asarray((0.0, 1.0, 0.0))
        return _axis_angle_rotation(np.cross(first, trial), math.pi)
    axis = cross / sine
    return _axis_angle_rotation(axis, math.atan2(sine, cosine))


__all__ = [
    "EnhancedEphemeris",
    "NBodyTideConfig",
    "attach_earth_tides",
    "attach_reboundx_earth_tides",
    "earth_tidal_accelerations",
]
