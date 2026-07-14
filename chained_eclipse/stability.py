"""Coupled N-body stability checks for the two-moon Earth system.

The eclipse search can use DE440s as a prescribed background, but a massive
second moon cannot be added to that background without perturbing the real
Moon.  This module therefore uses DE440s only to initialize the Sun, Earth,
and real Moon at the configured epoch.  From that instant onward all four
bodies are propagated self-consistently by REBOUND.

The reference integration is deliberately a Newtonian point-mass model.
Earth's J2, tides, relativity, major planets, and lunar figure terms are not
silently approximated.  They are listed in every result so that a successful
run is interpreted as a gross dynamical-stability test, not a 100,000-year
physical ephemeris.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    JULIAN_YEAR_DAYS,
    MU_EARTH_KM3_S2,
    MU_MOON_KM3_S2,
    MU_SUN_KM3_S2,
    OBLIQUITY_J2000_DEG,
    REAL_MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SUN_RADIUS_KM,
    WGS84_A_KM,
)
from .ephemeris import EphemerisContext
from .models import OrbitalElements

try:  # Permit importing reports/configuration code before optional deps install.
    import rebound as _rebound
except ImportError:  # pragma: no cover - exercised only in an incomplete environment
    _rebound = None

try:  # This module can still stand alone while orbital_dynamics is being scaffolded.
    from .orbital_dynamics import elements_to_state as _project_elements_to_state
except ImportError:  # pragma: no cover - fallback for partial installations
    _project_elements_to_state = None


_BODY_NAMES = ("sun", "earth", "real_moon", "second_moon")
_MOON_NAMES = ("real_moon", "second_moon")
_SERIES_KEYS = (
    "semimajor_axis_km",
    "eccentricity",
    "inclination_deg",
    "periapsis_km",
    "apoapsis_km",
    "earth_distance_km",
)


@dataclass(slots=True)
class StabilityConfig:
    """Numerical and diagnostic settings for a stability integration."""

    years: float = 1_000.0
    sample_interval_days: float = 30.0
    ias15_epsilon: float = 1.0e-10
    ejection_distance_km: float | None = None
    stop_on_ejection: bool = True
    monitor_internal_steps: bool = True
    severe_eccentricity: float = 0.8
    severe_semimajor_axis_fraction: float = 0.5
    max_relative_energy_error: float = 1.0e-8

    def validate(self) -> None:
        if not math.isfinite(self.years) or self.years <= 0.0:
            raise ValueError("years must be finite and positive")
        if not math.isfinite(self.sample_interval_days) or self.sample_interval_days <= 0.0:
            raise ValueError("sample_interval_days must be finite and positive")
        if not 0.0 < self.ias15_epsilon < 1.0:
            raise ValueError("ias15_epsilon must lie between zero and one")
        if self.ejection_distance_km is not None and self.ejection_distance_km <= 0.0:
            raise ValueError("ejection_distance_km must be positive when supplied")
        if not 0.0 < self.severe_eccentricity < 1.0:
            raise ValueError("severe_eccentricity must lie between zero and one")
        if self.severe_semimajor_axis_fraction <= 0.0:
            raise ValueError("severe_semimajor_axis_fraction must be positive")
        if self.max_relative_energy_error <= 0.0:
            raise ValueError("max_relative_energy_error must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(slots=True)
class StabilityResult:
    """JSON-serializable diagnostics from one coupled integration."""

    stable: bool
    completed: bool
    termination_reason: str | None
    epoch_utc: str
    requested_years: float
    integrated_years: float
    integrator: str
    ias15_epsilon: float
    sample_interval_days: float
    element_reference_frame: str
    ejection_distance_km: float
    time_years: list[float]
    real_moon: dict[str, list[float]]
    second_moon: dict[str, list[float]]
    moon_moon_distance_km: list[float]
    relative_energy_error: list[float]
    min_moon_moon_distance_km: float
    min_moon_moon_distance_time_years: float
    collision_detected: bool
    collision_pair: tuple[str, str] | None
    ejection_detected: bool
    ejected_bodies: list[str]
    unbound_detected: bool
    unbound_bodies: list[str]
    orbit_crossing_detected: bool
    severe_growth_detected: bool
    energy_error_exceeded: bool
    final_relative_energy_error: float
    max_abs_relative_energy_error: float
    initial_energy: float
    final_energy: float
    initial_state: dict[str, Any] = field(default_factory=dict)
    force_model: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-ready dictionary (no NumPy or NaN values)."""

        return _json_ready(asdict(self))


def orbital_elements_to_icrf_state(
    elements: OrbitalElements,
    gravitational_parameter_km3_s2: float,
) -> np.ndarray:
    """Convert mean-ecliptic J2000 elements to an Earth-relative ICRF state.

    The output order is ``x, y, z, vx, vy, vz`` in km and km/s.  The
    gravitational parameter should include Earth and second-moon masses,
    since these are relative two-body osculating elements.
    """

    frame = elements.frame.strip().lower().replace("-", "_")
    accepted_frames = {
        "mean_ecliptic_j2000",
        "j2000_mean_ecliptic",
        "ecliptic_j2000",
    }
    if frame not in accepted_frames:
        raise ValueError(
            f"Unsupported element frame {elements.frame!r}; expected mean ecliptic J2000"
        )
    a = float(elements.semimajor_axis_km)
    eccentricity = float(elements.eccentricity)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("semimajor axis must be finite and positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("the stability initializer currently requires 0 <= e < 1")
    if gravitational_parameter_km3_s2 <= 0.0:
        raise ValueError("gravitational parameter must be positive")

    inclination = math.radians(elements.inclination_deg)
    ascending_node = math.radians(elements.longitude_ascending_node_deg)
    periapsis = math.radians(elements.argument_periapsis_deg)
    mean_anomaly = math.radians(elements.mean_anomaly_deg)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)

    cos_e = math.cos(eccentric_anomaly)
    sin_e = math.sin(eccentric_anomaly)
    root = math.sqrt(1.0 - eccentricity * eccentricity)
    denominator = 1.0 - eccentricity * cos_e
    mean_motion = math.sqrt(gravitational_parameter_km3_s2 / a**3)
    position_perifocal = np.asarray(
        (a * (cos_e - eccentricity), a * root * sin_e, 0.0), dtype=float
    )
    velocity_perifocal = np.asarray(
        (
            -a * mean_motion * sin_e / denominator,
            a * mean_motion * root * cos_e / denominator,
            0.0,
        ),
        dtype=float,
    )

    perifocal_to_ecliptic = (
        _rotation_z(ascending_node) @ _rotation_x(inclination) @ _rotation_z(periapsis)
    )
    ecliptic_to_icrf = _rotation_x(math.radians(OBLIQUITY_J2000_DEG))
    rotation = ecliptic_to_icrf @ perifocal_to_ecliptic
    position_icrf = rotation @ position_perifocal
    velocity_icrf = rotation @ velocity_perifocal
    return np.concatenate((position_icrf, velocity_icrf))


def build_coupled_simulation(
    context: EphemerisContext,
    elements: OrbitalElements,
    *,
    second_moon_state_icrf: Any | None = None,
    ias15_epsilon: float = 1.0e-10,
) -> tuple[Any, dict[str, Any]]:
    """Build a four-body REBOUND simulation at the elements' epoch.

    ``second_moon_state_icrf`` may be a six-vector or an object/tuple holding
    Earth-relative ICRF position (km) and velocity (km/s).  If omitted, the
    state is calculated from ``OrbitalElements`` by this module.  Accepting an
    explicit state lets this function consume a refined state from
    :mod:`orbital_dynamics` without depending on that module's concrete API.
    """

    rebound = _require_rebound()
    if not 0.0 < ias15_epsilon < 1.0:
        raise ValueError("ias15_epsilon must lie between zero and one")

    epoch = context.time_utc(elements.epoch_utc)
    states = {
        "sun": _skyfield_state(context.sun, epoch),
        "earth": _skyfield_state(context.earth, epoch),
        "real_moon": _skyfield_state(context.moon, epoch),
    }

    simulation = rebound.Simulation()
    simulation.units = ("s", "km", "kg")
    gravitational_constant = float(simulation.G)

    # Particle masses derived from conventional GMs keep the dynamical model
    # internally consistent with the project's high-precision constants.
    masses = {
        "sun": MU_SUN_KM3_S2 / gravitational_constant,
        "earth": MU_EARTH_KM3_S2 / gravitational_constant,
        "real_moon": MU_MOON_KM3_S2 / gravitational_constant,
        "second_moon": float(elements.mass_kg),
    }
    radii = {
        "sun": SUN_RADIUS_KM,
        "earth": WGS84_A_KM,
        "real_moon": REAL_MOON_RADIUS_KM,
        "second_moon": float(elements.radius_km),
    }

    if second_moon_state_icrf is None:
        if _project_elements_to_state is not None:
            relative_second_state = _coerce_state(_project_elements_to_state(elements))
            state_source = "orbital_dynamics.elements_to_state"
        else:
            relative_second_state = orbital_elements_to_icrf_state(
                elements,
                MU_EARTH_KM3_S2,
            )
            state_source = "local OrbitalElements fallback using the project Earth GM"
    else:
        relative_second_state = _coerce_state(second_moon_state_icrf)
        state_source = "explicit Earth-relative ICRF Cartesian state"
    states["second_moon"] = states["earth"] + relative_second_state

    for name in _BODY_NAMES:
        state = states[name]
        _add_named_particle(
            simulation,
            name,
            mass_kg=masses[name],
            radius_km=radii[name],
            state=state,
        )

    # Translation to the modeled system's COM improves conditioning without
    # changing any relative state or osculating element.
    simulation.move_to_com()
    simulation.integrator = "ias15"
    if hasattr(simulation, "ri_ias15"):  # REBOUND 4.x
        simulation.ri_ias15.epsilon = ias15_epsilon
    else:  # REBOUND 5 exposes integrator fields through its configuration object.
        simulation.integrator.epsilon = ias15_epsilon
    simulation.collision = "direct"
    simulation.collision_resolve = "halt"

    earth_sun_distance = _distance(simulation.particles["earth"], simulation.particles["sun"])
    earth_system_mass = masses["earth"] + masses["real_moon"] + masses["second_moon"]
    hill_radius = earth_sun_distance * (earth_system_mass / (3.0 * masses["sun"])) ** (1.0 / 3.0)
    metadata = {
        "epoch_utc": elements.epoch_utc,
        "state_source": state_source,
        "relative_second_moon_state_icrf": relative_second_state.tolist(),
        "masses_kg": masses,
        "radii_km": radii,
        "gravitational_constant_km3_kg_s2": gravitational_constant,
        "earth_system_hill_radius_km": float(hill_radius),
        "ephemeris_kernel": str(getattr(context, "kernel_path", "unknown")),
    }
    return simulation, metadata


def run_stability_check(
    context: EphemerisContext,
    elements: OrbitalElements,
    *,
    config: StabilityConfig | None = None,
    years: float | None = None,
    sample_interval_days: float | None = None,
    second_moon_state_icrf: Any | None = None,
) -> StabilityResult:
    """Integrate and diagnose the coupled Sun-Earth-two-moon system.

    The default duration is 1,000 Julian years.  ``years`` and
    ``sample_interval_days`` are convenience overrides for smoke tests and
    command-line use; all other settings belong in ``StabilityConfig``.
    """

    settings = StabilityConfig() if config is None else StabilityConfig(**asdict(config))
    if years is not None:
        settings.years = float(years)
    if sample_interval_days is not None:
        settings.sample_interval_days = float(sample_interval_days)
    settings.validate()

    simulation, initial_state = build_coupled_simulation(
        context,
        elements,
        second_moon_state_icrf=second_moon_state_icrf,
        ias15_epsilon=settings.ias15_epsilon,
    )
    ejection_distance = (
        float(settings.ejection_distance_km)
        if settings.ejection_distance_km is not None
        else float(initial_state["earth_system_hill_radius_km"])
    )

    initial_energy = float(simulation.energy())
    time_years: list[float] = []
    moon_moon_distance: list[float] = []
    energy_error: list[float] = []
    series = {name: {key: [] for key in _SERIES_KEYS} for name in _MOON_NAMES}
    flags: dict[str, Any] = {
        "collision": False,
        "collision_pair": None,
        "ejected": set(),
        "unbound": set(),
        "orbit_crossing": False,
        "termination_reason": None,
        "failure": False,
    }
    monitor: dict[str, Any] = {
        "minimum_distance_km": math.inf,
        "minimum_distance_time_s": 0.0,
        "ejected": flags["ejected"],
        "stopped_for_ejection": False,
    }
    heartbeat_reference = None
    heartbeat_mode = "saved output samples"
    if settings.monitor_internal_steps:
        heartbeat_reference, heartbeat_mode = _install_internal_monitor(
            simulation,
            monitor,
            ejection_distance_km=ejection_distance,
            stop_on_ejection=settings.stop_on_ejection,
        )

    def record_sample() -> None:
        current_years = float(simulation.t / (JULIAN_YEAR_DAYS * SECONDS_PER_DAY))
        if time_years and abs(current_years - time_years[-1]) < 1.0e-14:
            return
        real = _osculating_sample(simulation, "real_moon")
        second = _osculating_sample(simulation, "second_moon")
        for name, sample in (("real_moon", real), ("second_moon", second)):
            for key in _SERIES_KEYS:
                series[name][key].append(float(sample[key]))
        current_distance = _distance(
            simulation.particles["real_moon"],
            simulation.particles["second_moon"],
        )
        if current_distance < monitor["minimum_distance_km"]:
            monitor["minimum_distance_km"] = current_distance
            monitor["minimum_distance_time_s"] = float(simulation.t)
        current_energy = float(simulation.energy())
        relative_error = (current_energy - initial_energy) / abs(initial_energy)
        time_years.append(current_years)
        moon_moon_distance.append(float(current_distance))
        energy_error.append(float(relative_error))

        if _radial_ranges_overlap(real, second):
            flags["orbit_crossing"] = True
        for name, sample in (("real_moon", real), ("second_moon", second)):
            if sample["earth_distance_km"] >= ejection_distance:
                flags["ejected"].add(name)
            if sample["eccentricity"] >= 1.0 or sample["semimajor_axis_km"] <= 0.0:
                flags["unbound"].add(name)

    record_sample()
    duration_s = settings.years * JULIAN_YEAR_DAYS * SECONDS_PER_DAY
    interval_s = settings.sample_interval_days * SECONDS_PER_DAY
    targets = np.arange(interval_s, duration_s, interval_s, dtype=float).tolist()
    targets.append(float(duration_s))

    for target in targets:
        try:
            simulation.integrate(target, exact_finish_time=1)
        except _rebound.Collision:
            flags["collision"] = True
            flags["collision_pair"] = _closest_surface_pair(simulation)
            flags["termination_reason"] = "physical collision"
            record_sample()
            break
        except Exception as exc:  # retain diagnostics from a failed long run
            flags["failure"] = True
            flags["termination_reason"] = f"integration failure: {type(exc).__name__}: {exc}"
            record_sample()
            break
        record_sample()
        if monitor["stopped_for_ejection"] or flags["ejected"]:
            flags["termination_reason"] = "Earth-system ejection boundary crossed"
            if settings.stop_on_ejection:
                break

    # Keep the callback alive through the final C call.  The explicit read
    # prevents linters from treating this important reference as dead state.
    _ = heartbeat_reference
    integrated_years = float(simulation.t / (JULIAN_YEAR_DAYS * SECONDS_PER_DAY))
    completed = (
        integrated_years >= settings.years * (1.0 - 1.0e-12) and flags["termination_reason"] is None
    )
    final_energy = float(simulation.energy())
    final_relative_error = (final_energy - initial_energy) / abs(initial_energy)
    max_abs_energy_error = max((abs(value) for value in energy_error), default=math.inf)
    energy_error_exceeded = max_abs_energy_error > settings.max_relative_energy_error
    severe_growth = _detect_severe_growth(series, settings)
    stable = bool(
        completed
        and not flags["collision"]
        and not flags["ejected"]
        and not flags["unbound"]
        and not flags["orbit_crossing"]
        and not severe_growth
        and not energy_error_exceeded
        and not flags["failure"]
    )

    warnings = [
        "DE440s supplies epoch states only; it is not forced after the epoch.",
        "Force model is Newtonian point masses: Earth J2, tides, relativity, lunar "
        "figure terms, and major planets are omitted.",
        "A 100,000-year point-mass result is not a tidal-evolution prediction.",
        "Input elements use the project's Earth-monopole state convention; reported "
        "two-body osculating elements use G times (Earth mass + satellite mass).",
        "Orbit crossing means overlap of sampled osculating radial ranges; inclined "
        "orbits need not intersect geometrically.",
        f"Minimum moon-moon distance was monitored at {heartbeat_mode}.",
    ]
    if not completed:
        warnings.append("Integration ended before the requested duration.")
    if energy_error_exceeded:
        warnings.append("Relative point-mass energy error exceeded the configured limit.")

    return StabilityResult(
        stable=stable,
        completed=completed,
        termination_reason=flags["termination_reason"],
        epoch_utc=elements.epoch_utc,
        requested_years=float(settings.years),
        integrated_years=integrated_years,
        integrator="IAS15",
        ias15_epsilon=float(settings.ias15_epsilon),
        sample_interval_days=float(settings.sample_interval_days),
        element_reference_frame="Earth-relative mean ecliptic J2000",
        ejection_distance_km=ejection_distance,
        time_years=time_years,
        real_moon=series["real_moon"],
        second_moon=series["second_moon"],
        moon_moon_distance_km=moon_moon_distance,
        relative_energy_error=energy_error,
        min_moon_moon_distance_km=float(monitor["minimum_distance_km"]),
        min_moon_moon_distance_time_years=float(
            monitor["minimum_distance_time_s"] / (JULIAN_YEAR_DAYS * SECONDS_PER_DAY)
        ),
        collision_detected=bool(flags["collision"]),
        collision_pair=flags["collision_pair"],
        ejection_detected=bool(flags["ejected"]),
        ejected_bodies=sorted(flags["ejected"]),
        unbound_detected=bool(flags["unbound"]),
        unbound_bodies=sorted(flags["unbound"]),
        orbit_crossing_detected=bool(flags["orbit_crossing"]),
        severe_growth_detected=severe_growth,
        energy_error_exceeded=energy_error_exceeded,
        final_relative_energy_error=float(final_relative_error),
        max_abs_relative_energy_error=float(max_abs_energy_error),
        initial_energy=initial_energy,
        final_energy=final_energy,
        initial_state=initial_state,
        force_model={
            "gravity": "fully coupled Newtonian four-body point masses",
            "bodies": list(_BODY_NAMES),
            "collision_detection": "REBOUND direct physical-radius collision halt",
            "ejection_boundary": "initial Earth-system Hill radius"
            if settings.ejection_distance_km is None
            else "user supplied geocentric radius",
            "element_diagnostics": "instantaneous two-body osculating about Earth",
        },
        warnings=warnings,
    )


def run_stability(
    context: EphemerisContext,
    elements: OrbitalElements,
    **kwargs: Any,
) -> StabilityResult:
    """Compatibility alias for :func:`run_stability_check`."""

    return run_stability_check(context, elements, **kwargs)


def plot_stability(
    result: StabilityResult | Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    title: str = "Coupled two-moon stability",
):
    """Plot sampled elements, moon separation, and point-mass energy error."""

    import matplotlib.pyplot as plt

    data = result.to_dict() if isinstance(result, StabilityResult) else dict(result)
    time = _float_array(data["time_years"])
    real = data["real_moon"]
    second = data["second_moon"]
    fig, axes = plt.subplots(5, 1, figsize=(11.0, 13.5), sharex=True, constrained_layout=True)

    axes[0].plot(time, _float_array(real["semimajor_axis_km"]), label="Real Moon")
    axes[0].plot(time, _float_array(second["semimajor_axis_km"]), label="Second moon")
    axes[0].set_ylabel("a (km)")
    axes[0].legend(loc="best")

    axes[1].plot(time, _float_array(real["eccentricity"]))
    axes[1].plot(time, _float_array(second["eccentricity"]))
    axes[1].set_ylabel("e")

    axes[2].plot(time, _float_array(real["inclination_deg"]))
    axes[2].plot(time, _float_array(second["inclination_deg"]))
    axes[2].set_ylabel("i (deg, ecl. J2000)")

    separation = _float_array(data["moon_moon_distance_km"])
    axes[3].plot(time, separation, color="tab:purple")
    axes[3].axhline(
        REAL_MOON_RADIUS_KM
        + float(data.get("initial_state", {}).get("radii_km", {}).get("second_moon", 0.0)),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Physical-radius sum",
    )
    axes[3].set_ylabel("Moon separation (km)")
    axes[3].legend(loc="best")

    absolute_error = np.abs(_float_array(data["relative_energy_error"]))
    positive_error = np.where(absolute_error > 0.0, absolute_error, np.nan)
    axes[4].semilogy(time, positive_error, color="tab:gray")
    axes[4].set_ylabel("|Delta E / E0|")
    axes[4].set_xlabel("Years after epoch")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.25)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    return fig


def _require_rebound():
    if _rebound is None:
        raise RuntimeError(
            "REBOUND is required for stability integration; install the project dependencies"
        )
    return _rebound


def _skyfield_state(body: Any, time: Any) -> np.ndarray:
    state = body.at(time)
    return np.concatenate(
        (
            np.asarray(state.position.km, dtype=float),
            np.asarray(state.velocity.km_per_s, dtype=float),
        )
    )


def _coerce_state(value: Any) -> np.ndarray:
    if isinstance(value, Sequence) and len(value) == 2:
        first = np.asarray(value[0], dtype=float)
        second = np.asarray(value[1], dtype=float)
        if first.shape == (3,) and second.shape == (3,):
            state = np.concatenate((first, second))
        else:
            state = np.asarray(value, dtype=float)
    else:
        position = None
        velocity = None
        for name in ("position_km", "r_km", "position"):
            if hasattr(value, name):
                position = np.asarray(getattr(value, name), dtype=float)
                break
        for name in ("velocity_km_s", "v_km_s", "velocity"):
            if hasattr(value, name):
                velocity = np.asarray(getattr(value, name), dtype=float)
                break
        if position is not None and velocity is not None:
            state = np.concatenate((position, velocity))
        else:
            state = np.asarray(value, dtype=float)
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("second_moon_state_icrf must be a finite Earth-relative six-vector")
    return state


def _add_named_particle(
    simulation: Any,
    name: str,
    *,
    mass_kg: float,
    radius_km: float,
    state: np.ndarray,
) -> None:
    kwargs = {
        "m": float(mass_kg),
        "r": float(radius_km),
        "x": float(state[0]),
        "y": float(state[1]),
        "z": float(state[2]),
        "vx": float(state[3]),
        "vy": float(state[4]),
        "vz": float(state[5]),
    }
    try:  # REBOUND 5 names particles; REBOUND 4 used hashes.
        simulation.add(name=name, **kwargs)
    except TypeError:  # pragma: no cover - version compatibility
        simulation.add(hash=name, **kwargs)


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    mean_anomaly = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi
    eccentric_anomaly = mean_anomaly if eccentricity < 0.8 else math.copysign(math.pi, mean_anomaly)
    for _ in range(30):
        residual = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly
        derivative = 1.0 - eccentricity * math.cos(eccentric_anomaly)
        correction = residual / derivative
        eccentric_anomaly -= correction
        if abs(correction) < 1.0e-14:
            return eccentric_anomaly
    raise RuntimeError("Kepler equation did not converge")


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=float,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _distance(first: Any, second: Any) -> float:
    dx = float(first.x - second.x)
    dy = float(first.y - second.y)
    dz = float(first.z - second.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _relative_vectors(simulation: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    earth = simulation.particles["earth"]
    body = simulation.particles[body_name]
    position = np.asarray((body.x - earth.x, body.y - earth.y, body.z - earth.z), dtype=float)
    velocity = np.asarray((body.vx - earth.vx, body.vy - earth.vy, body.vz - earth.vz), dtype=float)
    return position, velocity


def _osculating_sample(simulation: Any, body_name: str) -> dict[str, float]:
    position_icrf, velocity_icrf = _relative_vectors(simulation, body_name)
    icrf_to_ecliptic = _rotation_x(-math.radians(OBLIQUITY_J2000_DEG))
    position = icrf_to_ecliptic @ position_icrf
    velocity = icrf_to_ecliptic @ velocity_icrf
    distance = float(np.linalg.norm(position))
    speed_squared = float(np.dot(velocity, velocity))
    angular_momentum = np.cross(position, velocity)
    momentum_norm = float(np.linalg.norm(angular_momentum))
    earth = simulation.particles["earth"]
    body = simulation.particles[body_name]
    mu = float(simulation.G * (earth.m + body.m))
    specific_energy = 0.5 * speed_squared - mu / distance
    semimajor_axis = -mu / (2.0 * specific_energy) if specific_energy != 0.0 else math.inf
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu - position / distance
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    inclination = math.degrees(
        math.acos(float(np.clip(angular_momentum[2] / momentum_norm, -1.0, 1.0)))
    )
    if semimajor_axis > 0.0 and eccentricity < 1.0:
        periapsis = semimajor_axis * (1.0 - eccentricity)
        apoapsis = semimajor_axis * (1.0 + eccentricity)
    else:
        periapsis = math.nan
        apoapsis = math.nan
    return {
        "semimajor_axis_km": float(semimajor_axis),
        "eccentricity": eccentricity,
        "inclination_deg": float(inclination),
        "periapsis_km": float(periapsis),
        "apoapsis_km": float(apoapsis),
        "earth_distance_km": distance,
    }


def _radial_ranges_overlap(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    values = (
        first["periapsis_km"],
        first["apoapsis_km"],
        second["periapsis_km"],
        second["apoapsis_km"],
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return max(values[0], values[2]) <= min(values[1], values[3])


def _install_internal_monitor(
    simulation: Any,
    monitor: dict[str, Any],
    *,
    ejection_distance_km: float,
    stop_on_ejection: bool,
) -> tuple[Any | None, str]:
    def heartbeat(simulation_pointer: Any) -> None:
        try:
            current = simulation_pointer.contents
            real = current.particles["real_moon"]
            second = current.particles["second_moon"]
            earth = current.particles["earth"]
            separation = _distance(real, second)
            if separation < monitor["minimum_distance_km"]:
                monitor["minimum_distance_km"] = separation
                monitor["minimum_distance_time_s"] = float(current.t)
            crossed = False
            for name, moon in (("real_moon", real), ("second_moon", second)):
                if _distance(moon, earth) >= ejection_distance_km:
                    monitor["ejected"].add(name)
                    crossed = True
            if crossed and stop_on_ejection:
                monitor["stopped_for_ejection"] = True
                current.stop()
        except Exception:
            # A diagnostic callback must never corrupt or abort the integrator.
            return

    try:
        simulation.heartbeat = heartbeat
        return heartbeat, "IAS15 internal-step endpoints"
    except AttributeError:
        # REBOUND 5.0.1's public setter cannot retain its callback because its
        # slotted Simulation omits `_hb`; assign the documented C field while
        # returning the CFUNCTYPE object to keep it alive.
        try:
            from rebound.simulation import AFF

            callback_reference = AFF(heartbeat)
            simulation._heartbeat = callback_reference
            return callback_reference, "IAS15 internal-step endpoints"
        except Exception:
            return None, "saved output samples (heartbeat unavailable)"


def _closest_surface_pair(simulation: Any) -> tuple[str, str] | None:
    closest: tuple[str, str] | None = None
    minimum_gap = math.inf
    for index, first_name in enumerate(_BODY_NAMES):
        first = simulation.particles[first_name]
        for second_name in _BODY_NAMES[index + 1 :]:
            second = simulation.particles[second_name]
            gap = _distance(first, second) - float(first.r + second.r)
            if gap < minimum_gap:
                minimum_gap = gap
                closest = (first_name, second_name)
    return closest


def _detect_severe_growth(
    series: Mapping[str, Mapping[str, list[float]]], settings: StabilityConfig
) -> bool:
    for body in _MOON_NAMES:
        eccentricities = np.asarray(series[body]["eccentricity"], dtype=float)
        semimajor_axes = np.asarray(series[body]["semimajor_axis_km"], dtype=float)
        if eccentricities.size and np.nanmax(eccentricities) >= settings.severe_eccentricity:
            return True
        if semimajor_axes.size and math.isfinite(semimajor_axes[0]) and semimajor_axes[0] > 0.0:
            fractional_change = np.abs(semimajor_axes / semimajor_axes[0] - 1.0)
            if np.nanmax(fractional_change) >= settings.severe_semimajor_axis_fraction:
                return True
    return False


def _float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([math.nan if value is None else float(value) for value in values])


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


__all__ = [
    "StabilityConfig",
    "StabilityResult",
    "build_coupled_simulation",
    "orbital_elements_to_icrf_state",
    "plot_stability",
    "run_stability",
    "run_stability_check",
]
