"""Enhanced coupled trajectory used by the 30-year eclipse-climate search.

This module is the integration point for the project's optional higher-fidelity
counterfactual model.  It combines:

* active Newtonian point masses for all eight major planets;
* Earth's :math:`J_2` quadrupole;
* full first-order post-Newtonian interactions through REBOUNDx ``gr_full``;
* vector-spin constant-time-lag tides in Earth and both moons; and
* a differential Earth attitude history used to move eclipse ground tracks.

DE440s supplies all real-body states only at the 2026 epoch.  The alternate
system is then propagated freely by REBOUND IAS15.  Skyfield still supplies
the observed real-Earth orientation reference; differential Earth spin phase and pole
differences from a matched massless-second-moon control are composed onto that
reference.  This is deliberately explicit rather than being presented as a
new fitted ephemeris or observational Earth-orientation solution.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from skyfield.framelib import itrs

from .constants import SECONDS_PER_DAY, WGS84_A_KM
from .coupled_eclipse import CoupledEphemeris, PARTICLE_NAMES
from .enhanced_dynamics import (
    EnhancedDynamicsConfig,
    attach_reboundx_enhanced_forces,
)
from .ephemeris import EphemerisContext
from .interpolation import CubicSpline
from .models import OrbitalElements
from .moon_architecture import BinaryMoonArchitecture
from .planetary_dynamics import build_planetary_simulation
from .rotational_diagnostics import (
    relative_vector_change,
    rotational_diagnostic_snapshot,
)
from .rotational_dynamics import (
    NBodyTideConfig,
    attach_reboundx_rotational_tides as _attach_reboundx_rotational_tides,
)
from .tides_spin import (
    EARTH_MEAN_SOLAR_DAY_S,
    G_KM3_KG_S2,
    calibrate_earth_k2_delta_t_s,
    mignard_pair_effect,
)


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


def attach_reboundx_rotational_tides(
    simulation: Any,
    config: NBodyTideConfig | None = None,
) -> dict[str, Any]:
    """Compatibility export for the generalized structured-body tide."""

    return _attach_reboundx_rotational_tides(simulation, config)


def attach_reboundx_earth_tides(
    simulation: Any,
    config: NBodyTideConfig | None = None,
) -> dict[str, Any]:
    """Backward-compatible name for the generalized structured-body tide."""

    return attach_reboundx_rotational_tides(simulation, config)


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
        tide_metadata = attach_reboundx_rotational_tides(simulation, tide_settings)

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
            attach_reboundx_rotational_tides(control_simulation, tide_settings)

        structured_body_configs = tide_settings.resolved_bodies()
        spin_body_names = tuple(body.name for body in structured_body_configs)
        if "earth" not in spin_body_names:
            raise ValueError(
                "EnhancedEphemeris requires an Earth rotational state for "
                "the differential orientation model"
            )
        initial_spin_vectors = np.asarray(
            [
                body.initial_spin_vector_rad_s
                for body in structured_body_configs
            ],
            dtype=float,
        )
        earth_spin_index = spin_body_names.index("earth")
        initial_spin_vector = initial_spin_vectors[earth_spin_index]
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
                        velocities = np.asarray(cached["velocities"], dtype=float)
                        full_spin_vectors = np.asarray(
                            cached["full_spin_vectors"], dtype=float
                        )
                        control_spin_vectors = np.asarray(
                            cached["control_spin_vectors"], dtype=float
                        )
                        cached_spin_body_names = tuple(
                            str(name) for name in cached["spin_body_names"]
                        )
                        initial_newtonian_energy = float(
                            cached["initial_newtonian_energy"]
                        )
                        final_newtonian_energy = float(
                            cached["final_newtonian_energy"]
                        )
                        initial_rotational_diagnostic = json.loads(
                            str(cached["initial_rotational_diagnostic"].item())
                        )
                        final_rotational_diagnostic = json.loads(
                            str(cached["final_rotational_diagnostic"].item())
                        )
                        cache_hit = (
                            positions.shape
                            == (len(self.seconds), len(PARTICLE_NAMES), 3)
                            and velocities.shape
                            == (len(self.seconds), len(PARTICLE_NAMES), 3)
                            and cached_spin_body_names == spin_body_names
                            and full_spin_vectors.shape
                            == (len(self.seconds), len(spin_body_names), 3)
                            and control_spin_vectors.shape
                            == (len(self.seconds), len(spin_body_names), 3)
                        )
            except (KeyError, OSError, ValueError):
                # A partial or stale cache is never allowed to poison a run;
                # simply recompute it under the same content-addressed name.
                cache_hit = False
        if not cache_hit:
            initial_newtonian_energy = float(simulation.energy())
            initial_rotational_diagnostic = (
                rotational_diagnostic_snapshot(simulation).to_dict()
            )
            positions = np.empty(
                (len(self.seconds), len(PARTICLE_NAMES), 3), dtype=float
            )
            velocities = np.empty_like(positions)
            full_spin_vectors = np.empty(
                (len(self.seconds), len(spin_body_names), 3),
                dtype=float,
            )
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
                    velocities[index, body_index] = (
                        particle.vx,
                        particle.vy,
                        particle.vz,
                    )
                for spin_index, name in enumerate(spin_body_names):
                    full_spin_vectors[index, spin_index] = _particle_spin_vector(
                        simulation.particles[name],
                        fallback=initial_spin_vectors[spin_index],
                    )
                if control_simulation is None:
                    control_spin_vectors[index] = initial_spin_vectors
                else:
                    control_simulation.integrate(float(seconds), exact_finish_time=1)
                    for spin_index, name in enumerate(spin_body_names):
                        control_spin_vectors[index, spin_index] = (
                            _particle_spin_vector(
                                control_simulation.particles[name],
                                fallback=initial_spin_vectors[spin_index],
                            )
                        )
            final_newtonian_energy = float(simulation.energy())
            final_rotational_diagnostic = (
                rotational_diagnostic_snapshot(simulation).to_dict()
            )
            if cache_trajectory:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(
                    f".{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
                )
                np.savez(
                    temporary,
                    seconds=self.seconds,
                    positions=positions,
                    velocities=velocities,
                    spin_body_names=np.asarray(spin_body_names),
                    full_spin_vectors=full_spin_vectors,
                    control_spin_vectors=control_spin_vectors,
                    initial_newtonian_energy=initial_newtonian_energy,
                    final_newtonian_energy=final_newtonian_energy,
                    initial_rotational_diagnostic=np.asarray(
                        json.dumps(
                            initial_rotational_diagnostic,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    final_rotational_diagnostic=np.asarray(
                        json.dumps(
                            final_rotational_diagnostic,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                )
                temporary.replace(cache_path)

        self._splines = {
            name: CubicSpline(self.seconds, positions[:, index, :], axis=0)
            for index, name in enumerate(PARTICLE_NAMES)
        }
        self._velocity_splines = {
            name: CubicSpline(self.seconds, velocities[:, index, :], axis=0)
            for index, name in enumerate(PARTICLE_NAMES)
        }
        self._spin_splines = {
            name: CubicSpline(
                self.seconds,
                full_spin_vectors[:, index, :],
                axis=0,
            )
            for index, name in enumerate(spin_body_names)
        }
        self._control_spin_splines = {
            name: CubicSpline(
                self.seconds,
                control_spin_vectors[:, index, :],
                axis=0,
            )
            for index, name in enumerate(spin_body_names)
        }
        all_full_spin_rates = np.linalg.norm(full_spin_vectors, axis=2)
        all_control_spin_rates = np.linalg.norm(control_spin_vectors, axis=2)
        full_spin_rates = all_full_spin_rates[:, earth_spin_index]
        control_spin_rates = all_control_spin_rates[:, earth_spin_index]
        earth_full_spin_vectors = full_spin_vectors[:, earth_spin_index, :]
        earth_control_spin_vectors = control_spin_vectors[:, earth_spin_index, :]
        delta_spin_rate = full_spin_rates - control_spin_rates
        delta_phase = _cumulative_trapezoid(
            delta_spin_rate,
            self.seconds,
        )
        longitude_shift = -np.degrees(delta_phase)
        initial_spin_rate = float(np.linalg.norm(initial_spin_vector))
        self._initial_earth_spin_rate_rad_s = initial_spin_rate
        full_lod = EARTH_MEAN_SOLAR_DAY_S * initial_spin_rate / full_spin_rates
        control_lod = EARTH_MEAN_SOLAR_DAY_S * initial_spin_rate / control_spin_rates
        delta_lod_ms = (full_lod - control_lod) * 1_000.0
        delta_ut1_s = delta_phase / initial_spin_rate
        full_spin_axes = earth_full_spin_vectors / full_spin_rates[:, None]
        control_spin_axes = earth_control_spin_vectors / control_spin_rates[:, None]
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
        rotational_histories = {}
        mutual_mean_motion = (
            None
            if binary_architecture is None
            else math.sqrt(
                simulation.G
                * (
                    float(simulation.particles["real_moon"].m)
                    + float(simulation.particles["second_moon"].m)
                )
                / binary_architecture.mutual_orbit.semimajor_axis_km**3
            )
        )
        for spin_index, name in enumerate(spin_body_names):
            rates = all_full_spin_rates[:, spin_index]
            history = {
                "initial_spin_vector_rad_s": full_spin_vectors[
                    0, spin_index
                ].tolist(),
                "final_spin_vector_rad_s": full_spin_vectors[
                    -1, spin_index
                ].tolist(),
                "initial_spin_rate_rad_s": float(rates[0]),
                "final_spin_rate_rad_s": float(rates[-1]),
                "fractional_spin_rate_change": float(
                    (rates[-1] - rates[0]) / rates[0]
                ),
            }
            if mutual_mean_motion is not None and name != "earth":
                history.update(
                    {
                        "mutual_mean_motion_rad_s": mutual_mean_motion,
                        "initial_spin_to_mutual_mean_motion": float(
                            rates[0] / mutual_mean_motion
                        ),
                        "final_spin_to_mutual_mean_motion": float(
                            rates[-1] / mutual_mean_motion
                        ),
                    }
                )
            rotational_histories[name] = history

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
                "rotational_tides_and_spin": tide_metadata,
                # Retained as an alias for readers of the previous audit schema.
                "earth_tides_and_spin": tide_metadata,
                "earth_j2_and_first_post_newtonian": correction_metadata,
            },
            "rotational_state_histories": rotational_histories,
            "rotational_conservation_diagnostic": {
                "initial": initial_rotational_diagnostic,
                "final": final_rotational_diagnostic,
                "relative_total_angular_momentum_vector_change": (
                    relative_vector_change(
                        initial_rotational_diagnostic[
                            "total_angular_momentum_kg_km2_s"
                        ],
                        final_rotational_diagnostic[
                            "total_angular_momentum_kg_km2_s"
                        ],
                    )
                ),
                "mechanical_energy_change_kg_km2_s2": (
                    final_rotational_diagnostic[
                        "mechanical_energy_kg_km2_s2"
                    ]
                    - initial_rotational_diagnostic[
                        "mechanical_energy_kg_km2_s2"
                    ]
                ),
                "interpretation": (
                    "Orbital plus configured spin angular momentum. Strictly "
                    "conserved for isolated reaction-complete limits; with Earth "
                    "gravitational_harmonics enabled, its omitted source-spin "
                    "reaction makes this a bounded diagnostic rather than a "
                    "conservation claim. Positive-lag CTL tides dissipate the "
                    "reported mechanical energy."
                ),
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
                "reaction-aware lunar and second-moon permanent J2/C22 figures and physical libration",
                "post-1PN relativity, solar frame dragging, and solar quadrupole",
                "frequency-dependent ocean, mantle, and solid-body tidal response",
                "J2 reaction torque on Earth's spin, free nutation, and atmosphere/ocean angular momentum",
                "a refitted alternate-planet ephemeris and observational Earth-orientation series",
            ],
            "limitations": [
                "DE440s supplies real-body states only at the 2026 epoch; the alternate system then evolves freely.",
                "Major planets are point masses; Mars through Neptune use system barycentres and system GMs.",
                "Relativity uses REBOUNDx gr_full first-order post-Newtonian dynamics; higher orders and spin terms are omitted.",
                "The constant-time-lag Earth tide is calibrated to present lunar recession and is not an ocean model; lunar and giant-moon lags are labeled scenarios.",
                "REBOUNDx tides_spin applies each structured body's tide to every other active massive body; it has no pair filter.",
                "The empirical Earth J2 and tides_spin equilibrium quadrupoles may overlap; strict angular-momentum tests disable gravitational_harmonics because it omits the J2 source-spin reaction.",
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

    def velocity(self, name: str, tt_jd: float | np.ndarray) -> np.ndarray:
        """Return the integrator-sampled inertial velocity in km/s."""

        return np.asarray(
            self._velocity_splines[name](self._seconds(tt_jd)),
            dtype=float,
        )

    def spin_vector_rad_s(
        self,
        body: str,
        tt_jd: float | np.ndarray,
    ) -> np.ndarray:
        """Return one modeled body's inertial spin vector in rad/s."""

        if body not in self._spin_splines:
            raise KeyError(f"no modeled spin history for {body!r}")
        return np.asarray(
            self._spin_splines[body](self._seconds(tt_jd)),
            dtype=float,
        )

    def earth_rotation_diagnostics(
        self,
        tt_jd: float,
    ) -> dict[str, float]:
        """Return full-minus-control Earth rotation diagnostics at one epoch."""

        seconds = float(self._seconds(tt_jd))
        full_vector = np.asarray(self._spin_splines["earth"](seconds), dtype=float)
        control_vector = np.asarray(
            self._control_spin_splines["earth"](seconds),
            dtype=float,
        )
        full_rate = float(np.linalg.norm(full_vector))
        control_rate = float(np.linalg.norm(control_vector))
        initial_rate = self._initial_earth_spin_rate_rad_s
        full_axis = full_vector / full_rate
        control_axis = control_vector / control_rate
        pole_separation = math.acos(
            float(np.clip(np.dot(full_axis, control_axis), -1.0, 1.0))
        )
        delta_phase = float(self._phase_offset_spline(seconds))
        return {
            "delta_mean_solar_lod_ms": (
                EARTH_MEAN_SOLAR_DAY_S
                * initial_rate
                * (1.0 / full_rate - 1.0 / control_rate)
                * 1_000.0
            ),
            "delta_ut1_s": delta_phase / initial_rate,
            "longitude_shift_deg": float(self._longitude_offset_spline(seconds)),
            "pole_separation_arcsec": math.degrees(pole_separation) * 3_600.0,
        }

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
    try:
        import rebound
        import reboundx

        numerical_runtime = {
            "rebound": rebound.__version__,
            "reboundx": reboundx.__version__,
        }
    except ImportError:  # pragma: no cover - production path already requires both
        numerical_runtime = {"rebound": "unavailable", "reboundx": "unavailable"}
    payload = {
        "schema": 6,
        "force_model_revision": "three-structured-body-tides-spin-v2",
        "elements": elements.to_dict(),
        "binary_architecture": (
            None if binary_architecture is None else binary_architecture.to_dict()
        ),
        "end_utc": end_utc,
        "sample_step_seconds": sample_step_seconds,
        "ias15_epsilon": ias15_epsilon,
        "dynamics": dynamics_config.to_dict(),
        "rotational_tides": tide_config.to_dict(),
        "include_pluto": include_pluto,
        "numerical_runtime": numerical_runtime,
        "kernel": {
            "name": kernel.name,
            "size": kernel_stat.st_size,
            "mtime_ns": kernel_stat.st_mtime_ns,
            "sha256": _file_sha256(kernel),
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cumulative_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Integrate sampled scalar values without importing all of scipy.integrate."""

    values = np.asarray(values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or coordinates.shape != values.shape:
        raise ValueError("cumulative trapezoid inputs must be equal-length vectors")
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(coordinates)
    return np.concatenate((np.zeros(1, dtype=float), np.cumsum(increments)))


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
    "attach_reboundx_rotational_tides",
    "earth_tidal_accelerations",
]
