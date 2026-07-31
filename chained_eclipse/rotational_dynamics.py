"""Configuration for coupled spin and constant-time-lag tidal dynamics.

The production integrator uses REBOUNDx ``tides_spin``.  Its structured-body
parameters are expressed in the simulator's inertial ICRF/J2000 frame and in
the repository's native km--s--kg units:

* angular velocity: rad/s;
* polar moment of inertia: kg km^2;
* Love number ``k2``: dimensionless; and
* constant time lag ``tau``: seconds.

REBOUNDx applies each configured structured body's tide against every other
massive particle.  It does not expose a source--target pair filter.  That
interaction scope is therefore explicit configuration and metadata rather
than being inferred from a list of satellites.

Permanent figures are represented here so scenario assumptions, cache
identity, and future backends have a stable interface.  They intentionally
cannot be enabled yet: the current REBOUNDx gravitational-harmonics effect
omits the equal-and-opposite source-spin reaction, and C22 additionally needs
a body attitude state.  Silently enabling either would violate the simulator's
angular-momentum contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from .constants import WGS84_A_KM
from .tides_spin import calibrate_earth_k2_delta_t_s


STRUCTURED_BODY_NAMES: tuple[str, ...] = (
    "earth",
    "real_moon",
    "second_moon",
)
ALL_ACTIVE_MASSIVE_BODIES = "all_active_massive_bodies"
EARTH_SIDEREAL_DAY_S = 86_164.0905
EARTH_POLAR_MOMENT_FACTOR = 0.3307


def _unknown_fields(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(unknown)}")


def _finite_vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return tuple(float(component) for component in vector)


@dataclass(frozen=True, slots=True)
class PermanentFigureConfig:
    """A deferred, reaction-aware permanent-figure interface.

    ``j2`` and ``c22`` are unnormalized dimensionless coefficients.  Values
    may be carried while ``enabled`` is false for sensitivity bookkeeping.
    """

    enabled: bool = False
    j2: float = 0.0
    c22: float = 0.0
    backend: str = "reaction_aware_attitude_backend_required"
    scenario_label: str = "spherical"
    provenance: str = "explicit scenario assumption"

    def validate(self) -> None:
        for name, value in (("j2", self.j2), ("c22", self.c22)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.backend.strip():
            raise ValueError("permanent-figure backend must not be empty")
        if not self.scenario_label.strip() or not self.provenance.strip():
            raise ValueError("permanent-figure scenario and provenance must not be empty")
        if self.enabled:
            raise ValueError(
                "permanent figures require a reaction-aware attitude backend; "
                "REBOUNDx gravitational_harmonics omits the source-spin reaction"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PermanentFigureConfig:
        _unknown_fields(
            payload,
            {
                "enabled",
                "j2",
                "c22",
                "backend",
                "scenario_label",
                "provenance",
            },
            context="permanent_figure",
        )
        result = cls(
            enabled=bool(payload.get("enabled", False)),
            j2=float(payload.get("j2", 0.0)),
            c22=float(payload.get("c22", 0.0)),
            backend=str(
                payload.get(
                    "backend",
                    "reaction_aware_attitude_backend_required",
                )
            ),
            scenario_label=str(payload.get("scenario_label", "spherical")),
            provenance=str(payload.get("provenance", "explicit scenario assumption")),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class RotationalBodyConfig:
    """One body's axisymmetric spin/inertia and dissipative tide parameters."""

    name: str
    radius_km: float
    polar_moment_factor: float
    initial_spin_vector_rad_s: tuple[float, float, float]
    love_number_k2: float
    constant_time_lag_s: float | None
    scenario_label: str
    provenance: str
    permanent_figure: PermanentFigureConfig = PermanentFigureConfig()

    def validate(self) -> None:
        if self.name not in STRUCTURED_BODY_NAMES:
            raise ValueError(f"unsupported structured body {self.name!r}")
        if not math.isfinite(self.radius_km) or self.radius_km <= 0.0:
            raise ValueError(f"{self.name}.radius_km must be finite and positive")
        if (
            not math.isfinite(self.polar_moment_factor)
            or self.polar_moment_factor <= 0.0
            or self.polar_moment_factor > 1.0
        ):
            raise ValueError(
                f"{self.name}.polar_moment_factor must satisfy 0 < factor <= 1"
            )
        spin = np.asarray(self.initial_spin_vector_rad_s, dtype=float)
        if spin.shape != (3,) or not np.all(np.isfinite(spin)):
            raise ValueError(
                f"{self.name}.initial_spin_vector_rad_s must be a finite three-vector"
            )
        if np.linalg.norm(spin) == 0.0:
            raise ValueError(
                f"{self.name}.initial_spin_vector_rad_s must be non-zero"
            )
        if not math.isfinite(self.love_number_k2) or self.love_number_k2 < 0.0:
            raise ValueError(f"{self.name}.love_number_k2 must be non-negative")
        if self.constant_time_lag_s is not None and (
            not math.isfinite(self.constant_time_lag_s)
            or self.constant_time_lag_s < 0.0
        ):
            raise ValueError(
                f"{self.name}.constant_time_lag_s must be finite and non-negative"
            )
        if (
            self.name != "earth"
            and self.love_number_k2 > 0.0
            and self.constant_time_lag_s is None
        ):
            raise ValueError(
                f"{self.name}.constant_time_lag_s is required when its tide is active"
            )
        if not self.scenario_label.strip() or not self.provenance.strip():
            raise ValueError(f"{self.name} scenario and provenance must not be empty")
        self.permanent_figure.validate()

    @property
    def initial_spin_rate_rad_s(self) -> float:
        return float(np.linalg.norm(self.initial_spin_vector_rad_s))

    @classmethod
    def from_mapping(
        cls,
        name: str,
        payload: Mapping[str, Any],
    ) -> RotationalBodyConfig:
        _unknown_fields(
            payload,
            {
                "radius_km",
                "polar_moment_factor",
                "initial_spin_vector_rad_s",
                "love_number_k2",
                "constant_time_lag_s",
                "scenario_label",
                "provenance",
                "permanent_figure",
            },
            context=f"rotational_tides body {name}",
        )
        figure_payload = payload.get("permanent_figure", {})
        if not isinstance(figure_payload, Mapping):
            raise ValueError(f"{name}.permanent_figure must be a mapping")
        lag = payload.get("constant_time_lag_s")
        result = cls(
            name=name,
            radius_km=float(payload["radius_km"]),
            polar_moment_factor=float(payload["polar_moment_factor"]),
            initial_spin_vector_rad_s=_finite_vector3(
                payload["initial_spin_vector_rad_s"],
                name=f"{name}.initial_spin_vector_rad_s",
            ),
            love_number_k2=float(payload["love_number_k2"]),
            constant_time_lag_s=None if lag is None else float(lag),
            scenario_label=str(payload["scenario_label"]),
            provenance=str(payload["provenance"]),
            permanent_figure=PermanentFigureConfig.from_mapping(figure_payload),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NBodyTideConfig:
    """Settings for the coupled structured-body tide and vector-spin model.

    The legacy Earth fields retain source compatibility.  When
    ``structured_bodies`` is empty they resolve to the exact previous
    Earth-only model.  Parsed scenario configurations populate
    ``structured_bodies`` explicitly.
    """

    enabled: bool = True
    satellite_names: tuple[str, ...] = ("real_moon", "second_moon")
    earth_spin_axis_icrf: tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_sidereal_day_s: float = EARTH_SIDEREAL_DAY_S
    earth_love_number_k2: float = 0.3
    earth_polar_moment_factor: float = EARTH_POLAR_MOMENT_FACTOR
    evolve_spin: bool = True
    k2_delta_t_s: float | None = None
    interaction_scope: str = ALL_ACTIVE_MASSIVE_BODIES
    active_scenario: str = "legacy_earth_only"
    structured_bodies: tuple[RotationalBodyConfig, ...] = ()

    def validate(self) -> None:
        if self.enabled and not self.evolve_spin:
            raise ValueError(
                "active tides require evolve_spin=true so every orbital torque "
                "has its equal-and-opposite spin reaction; disable the tide "
                "entirely for the backward-compatible fixed-spin limit"
            )
        if not self.satellite_names:
            raise ValueError("satellite_names must not be empty")
        if len(set(self.satellite_names)) != len(self.satellite_names):
            raise ValueError("satellite_names must be unique")
        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("earth_spin_axis_icrf must be a finite three-vector")
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("earth_spin_axis_icrf must be non-zero")
        if (
            not math.isfinite(self.initial_sidereal_day_s)
            or self.initial_sidereal_day_s <= 0.0
        ):
            raise ValueError("initial_sidereal_day_s must be finite and positive")
        if (
            not math.isfinite(self.earth_love_number_k2)
            or self.earth_love_number_k2 <= 0.0
        ):
            raise ValueError("earth_love_number_k2 must be finite and positive")
        if (
            not math.isfinite(self.earth_polar_moment_factor)
            or self.earth_polar_moment_factor <= 0.0
        ):
            raise ValueError("earth_polar_moment_factor must be finite and positive")
        if self.k2_delta_t_s is not None and (
            not math.isfinite(self.k2_delta_t_s) or self.k2_delta_t_s < 0.0
        ):
            raise ValueError(
                "k2_delta_t_s must be finite and non-negative when supplied"
            )
        if self.interaction_scope != ALL_ACTIVE_MASSIVE_BODIES:
            raise ValueError(
                "REBOUNDx tides_spin supports only all_active_massive_bodies"
            )
        if not self.active_scenario.strip():
            raise ValueError("active_scenario must not be empty")
        names = [body.name for body in self.structured_bodies]
        if len(set(names)) != len(names):
            raise ValueError("structured body names must be unique")
        for body in self.structured_bodies:
            body.validate()

    def normalized_spin_axis(self) -> np.ndarray:
        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        return axis / np.linalg.norm(axis)

    def resolved_bodies(self) -> tuple[RotationalBodyConfig, ...]:
        """Return explicit bodies, or the backward-compatible Earth model."""

        if self.structured_bodies:
            return self.structured_bodies
        spin = self.normalized_spin_axis() * (
            2.0 * math.pi / self.initial_sidereal_day_s
        )
        return (
            RotationalBodyConfig(
                name="earth",
                radius_km=WGS84_A_KM,
                polar_moment_factor=self.earth_polar_moment_factor,
                initial_spin_vector_rad_s=tuple(float(value) for value in spin),
                love_number_k2=self.earth_love_number_k2,
                constant_time_lag_s=None,
                scenario_label="legacy_calibrated_earth",
                provenance=(
                    "calibrated to 38.2 mm/year circular lunar recession; "
                    "bounded CTL approximation, not a global ocean model"
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rotational_tide_config_from_mapping(
    payload: Mapping[str, Any],
) -> NBodyTideConfig | None:
    """Resolve the active named rotational/tidal scenario from configuration."""

    section = payload.get("rotational_tides")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("rotational_tides must be a mapping")
    _unknown_fields(
        section,
        {
            "enabled",
            "evolve_spin",
            "interaction_scope",
            "active_scenario",
            "scenarios",
        },
        context="rotational_tides",
    )
    scenarios = section.get("scenarios")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("rotational_tides.scenarios must be a non-empty mapping")
    active = str(section.get("active_scenario", "nominal"))
    if active not in scenarios:
        raise ValueError(f"unknown rotational-tide scenario {active!r}")
    selected = scenarios[active]
    if not isinstance(selected, Mapping):
        raise ValueError(f"rotational-tide scenario {active!r} must be a mapping")
    _unknown_fields(
        selected,
        {"description", "bodies"},
        context=f"rotational-tide scenario {active}",
    )
    bodies_payload = selected.get("bodies")
    if not isinstance(bodies_payload, Mapping):
        raise ValueError(f"rotational-tide scenario {active!r} needs bodies")
    unknown_bodies = sorted(set(bodies_payload) - set(STRUCTURED_BODY_NAMES))
    if unknown_bodies:
        raise ValueError(
            f"unknown rotational-tide bodies: {', '.join(unknown_bodies)}"
        )
    missing_bodies = sorted(set(STRUCTURED_BODY_NAMES) - set(bodies_payload))
    if missing_bodies:
        raise ValueError(
            f"rotational-tide scenario is missing bodies: {', '.join(missing_bodies)}"
        )
    bodies = tuple(
        RotationalBodyConfig.from_mapping(name, bodies_payload[name])
        for name in STRUCTURED_BODY_NAMES
    )
    result = NBodyTideConfig(
        enabled=bool(section.get("enabled", True)),
        evolve_spin=bool(section.get("evolve_spin", True)),
        interaction_scope=str(
            section.get("interaction_scope", ALL_ACTIVE_MASSIVE_BODIES)
        ),
        active_scenario=active,
        structured_bodies=bodies,
    )
    result.validate()
    return result


def attach_reboundx_rotational_tides(
    simulation: Any,
    config: NBodyTideConfig | None = None,
) -> dict[str, Any]:
    """Attach one compiled CTL force for all configured structured bodies.

    Frame and signs are inherited from REBOUNDx ``tides_spin``: positions,
    velocities, and spins are inertial; each internal orbital force has an
    equal-and-opposite reaction; and the deformed source receives the balancing
    spin torque. Positive ``tau`` dissipates mechanical energy.
    """

    try:
        import reboundx
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("The coupled tidal-spin model requires reboundx") from exc

    settings = NBodyTideConfig() if config is None else config
    settings.validate()
    configured_bodies = settings.resolved_bodies()
    applied: list[dict[str, Any]] = []
    skipped_massless: list[str] = []
    if settings.enabled:
        extras = getattr(simulation, "_extras_ref", None)
        if extras is None:
            extras = reboundx.Extras(simulation)
            simulation._extras_ref = extras
        tides = extras.load_force("tides_spin")
        extras.add_force(tides)
        for body_config in configured_bodies:
            particle = simulation.particles[body_config.name]
            mass = float(particle.m)
            if mass <= 0.0:
                skipped_massless.append(body_config.name)
                continue
            particle.r = body_config.radius_km
            physical_lag = body_config.constant_time_lag_s
            calibration_target = None
            if physical_lag is None:
                if body_config.name != "earth":
                    raise ValueError(
                        "only Earth's legacy recession calibration may omit "
                        "constant_time_lag_s"
                    )
                k2_lag = (
                    calibrate_earth_k2_delta_t_s(
                        moon_mass_kg=float(simulation.particles["real_moon"].m),
                        earth_mass_kg=mass,
                        earth_spin_rad_s=body_config.initial_spin_rate_rad_s,
                    )
                    if settings.k2_delta_t_s is None
                    else float(settings.k2_delta_t_s)
                )
                # REBOUNDx's Eggleton/Hut normalization produces twice the
                # circular recession of the Mignard oracle for the same k2*tau.
                physical_lag = k2_lag / (2.0 * body_config.love_number_k2)
                calibration_target = (
                    "38.2 mm/year circular real-Moon recession at 384,400 km"
                )
            particle.params["k2"] = body_config.love_number_k2
            particle.params["I"] = (
                body_config.polar_moment_factor
                * mass
                * body_config.radius_km**2
            )
            particle.params["Omega"] = np.asarray(
                body_config.initial_spin_vector_rad_s,
                dtype=float,
            )
            particle.params["tau"] = physical_lag
            applied.append(
                {
                    **body_config.to_dict(),
                    "mass_kg": mass,
                    "polar_moment_kg_km2": float(particle.params["I"]),
                    "constant_time_lag_s": physical_lag,
                    "mignard_equivalent_k2_delta_t_s": (
                        2.0 * body_config.love_number_k2 * physical_lag
                    ),
                    "calibration_target": calibration_target,
                }
            )
        if settings.evolve_spin and applied:
            extras.initialize_spin_ode(tides)
        simulation._tides_spin_force_ref = tides

    return {
        "enabled": settings.enabled,
        "model": "REBOUNDx tides_spin vector constant-time-lag equilibrium tide",
        "implementation": "one compiled C force plus one coupled vector-spin ODE",
        "active_scenario": settings.active_scenario,
        "interaction_scope": settings.interaction_scope,
        "interaction_scope_explanation": (
            "Every configured structured source interacts with every other "
            "massive active particle; REBOUNDx exposes no source-target pair filter."
        ),
        "structured_bodies": applied,
        "skipped_massless_structured_bodies": skipped_massless,
        "reboundx_normalization_factor": 0.5,
        "spin_evolved_inside_nbody": bool(settings.enabled and settings.evolve_spin),
        "force_and_torque_contract": (
            "Each CTL channel applies equal-and-opposite orbital force and the "
            "balancing torque to the deformed body's spin."
        ),
        "conservative_limit": (
            "k2=0 or tau=0 removes that body's dissipative CTL contribution"
        ),
        "omissions": [
            "frequency-dependent ocean, mantle, and solid-body response",
            "time-varying Earth inertia and atmosphere/ocean angular momentum",
            "permanent J2/C22 figures with reaction-aware attitude dynamics",
        ],
    }


__all__ = [
    "ALL_ACTIVE_MASSIVE_BODIES",
    "NBodyTideConfig",
    "PermanentFigureConfig",
    "RotationalBodyConfig",
    "STRUCTURED_BODY_NAMES",
    "attach_reboundx_rotational_tides",
    "rotational_tide_config_from_mapping",
]
