"""Optional non-Newtonian forces for the coupled two-moon simulation.

The baseline coupled model intentionally uses Newtonian point masses.  This
module adds two small, explicitly bounded corrections without requiring
REBOUNDx:

* Earth's axisymmetric :math:`J_2` quadrupole, referred to a fixed ICRF/J2000
  spin axis; and
* the first post-Newtonian (1PN) relative acceleration for every Sun--body
  pair.

The 1PN term is the standard isolated two-body relative equation including the
symmetric mass ratio.  Applying it pairwise to a many-body system is a
solar-monopole approximation, not the full Einstein--Infeld--Hoffmann N-body
equations.  Each relative correction is distributed between the two members
of its pair so the added forces conserve total linear momentum.

Distances are kilometres, times seconds, and masses kilograms throughout.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .constants import (
    EARTH_J2,
    MU_EARTH_KM3_S2,
    SPEED_OF_LIGHT_KM_S,
    WGS84_A_KM,
)
from .models import OrbitalElements
from .moon_architecture import BinaryMoonArchitecture


@dataclass(frozen=True, slots=True)
class EnhancedDynamicsConfig:
    """Configuration for the optional coupled-force corrections.

    ``earth_spin_axis_icrf`` is deliberately explicit.  Its default is the
    mean north pole of the J2000 equator, which is the +z axis of ICRF to the
    accuracy relevant here.  Precession, nutation, polar motion, tides, and
    lunar figure forces remain separate model choices.
    """

    earth_j2_enabled: bool = True
    solar_1pn_enabled: bool = True
    earth_j2: float = EARTH_J2
    earth_equatorial_radius_km: float = WGS84_A_KM
    earth_mu_km3_s2: float = MU_EARTH_KM3_S2
    earth_spin_axis_icrf: tuple[float, float, float] = (0.0, 0.0, 1.0)
    speed_of_light_km_s: float = SPEED_OF_LIGHT_KM_S

    def validate(self) -> None:
        """Reject non-physical or numerically undefined settings."""

        if not math.isfinite(self.earth_j2) or self.earth_j2 < 0.0:
            raise ValueError("earth_j2 must be finite and non-negative")
        if (
            not math.isfinite(self.earth_equatorial_radius_km)
            or self.earth_equatorial_radius_km <= 0.0
        ):
            raise ValueError("earth_equatorial_radius_km must be finite and positive")
        if not math.isfinite(self.earth_mu_km3_s2) or self.earth_mu_km3_s2 <= 0.0:
            raise ValueError("earth_mu_km3_s2 must be finite and positive")
        if not math.isfinite(self.speed_of_light_km_s) or self.speed_of_light_km_s <= 0.0:
            raise ValueError("speed_of_light_km_s must be finite and positive")
        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
            raise ValueError("earth_spin_axis_icrf must be a finite, non-zero three-vector")

    def normalized_spin_axis(self) -> np.ndarray:
        """Return a unit vector along the configured terrestrial north pole."""

        axis = np.asarray(self.earth_spin_axis_icrf, dtype=float)
        return axis / np.linalg.norm(axis)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready settings."""

        data = asdict(self)
        data["earth_spin_axis_icrf"] = list(self.earth_spin_axis_icrf)
        return data


def earth_j2_acceleration(
    position_from_earth_km: np.ndarray,
    *,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    equatorial_radius_km: float = WGS84_A_KM,
    j2: float = EARTH_J2,
    spin_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    r"""Return the acceleration from Earth's :math:`J_2` term only.

    With :math:`\mathbf{k}` the terrestrial spin unit vector,
    :math:`z=\mathbf{r}\!\cdot\!\mathbf{k}`, and :math:`r=|\mathbf{r}|`,

    .. math::

       \mathbf{a}_{J_2}=\frac{3J_2\mu R_e^2}{2r^5}
       \left[\left(5\frac{z^2}{r^2}-1\right)\mathbf{r}
       -2z\mathbf{k}\right].

    The Newtonian monopole is not included.  ``position_from_earth_km`` is
    the target's inertial position relative to Earth's centre.
    """

    position = _finite_vector(position_from_earth_km, "position_from_earth_km")
    axis = _finite_vector(spin_axis, "spin_axis")
    axis_norm = float(np.linalg.norm(axis))
    radius_squared = float(np.dot(position, position))
    if radius_squared == 0.0:
        raise ValueError("J2 acceleration is undefined at Earth's centre")
    if axis_norm == 0.0:
        raise ValueError("spin_axis must be non-zero")
    if (
        not math.isfinite(mu_km3_s2)
        or not math.isfinite(equatorial_radius_km)
        or not math.isfinite(j2)
        or mu_km3_s2 <= 0.0
        or equatorial_radius_km <= 0.0
        or j2 < 0.0
    ):
        raise ValueError("J2 constants require mu > 0, radius > 0, and j2 >= 0")

    axis = axis / axis_norm
    axial_distance = float(np.dot(position, axis))
    radius = math.sqrt(radius_squared)
    prefactor = 1.5 * j2 * mu_km3_s2 * equatorial_radius_km**2 / radius**5
    return prefactor * (
        (5.0 * axial_distance**2 / radius_squared - 1.0) * position
        - 2.0 * axial_distance * axis
    )


def solar_1pn_relative_acceleration(
    relative_position_km: np.ndarray,
    relative_velocity_km_s: np.ndarray,
    *,
    total_mu_km3_s2: float,
    symmetric_mass_ratio: float = 0.0,
    speed_of_light_km_s: float = SPEED_OF_LIGHT_KM_S,
) -> np.ndarray:
    r"""Return the first post-Newtonian Sun--body relative acceleration.

    This is the 1PN correction (the Newtonian term is excluded) in harmonic
    coordinates for an isolated, non-spinning two-body pair:

    .. math::

       \mathbf{a}_{1PN}=\frac{\mu}{c^2r^3}\left\{
       \left[(4+2\eta)\frac{\mu}{r}-(1+3\eta)v^2
       +\frac{3}{2}\eta\dot r^2\right]\mathbf{r}
       +(4-2\eta)(\mathbf{r}\!\cdot\!\mathbf{v})\mathbf{v}\right\}.

    Here :math:`\eta=m_1m_2/(m_1+m_2)^2`.  The test-particle Schwarzschild
    expression is recovered when ``symmetric_mass_ratio`` is zero.
    """

    position = _finite_vector(relative_position_km, "relative_position_km")
    velocity = _finite_vector(relative_velocity_km_s, "relative_velocity_km_s")
    radius_squared = float(np.dot(position, position))
    if radius_squared == 0.0:
        raise ValueError("1PN acceleration is undefined at zero separation")
    if not math.isfinite(total_mu_km3_s2) or total_mu_km3_s2 <= 0.0:
        raise ValueError("total_mu_km3_s2 must be finite and positive")
    if not 0.0 <= symmetric_mass_ratio <= 0.25:
        raise ValueError("symmetric_mass_ratio must lie between zero and 0.25")
    if not math.isfinite(speed_of_light_km_s) or speed_of_light_km_s <= 0.0:
        raise ValueError("speed_of_light_km_s must be finite and positive")

    radius = math.sqrt(radius_squared)
    velocity_squared = float(np.dot(velocity, velocity))
    radial_velocity = float(np.dot(position, velocity)) / radius
    radial_coefficient = (
        (4.0 + 2.0 * symmetric_mass_ratio) * total_mu_km3_s2 / radius
        - (1.0 + 3.0 * symmetric_mass_ratio) * velocity_squared
        + 1.5 * symmetric_mass_ratio * radial_velocity**2
    )
    velocity_coefficient = (4.0 - 2.0 * symmetric_mass_ratio) * float(
        np.dot(position, velocity)
    )
    prefactor = total_mu_km3_s2 / (
        speed_of_light_km_s**2 * radius**3
    )
    return prefactor * (radial_coefficient * position + velocity_coefficient * velocity)


def enhanced_accelerations(
    simulation: Any,
    config: EnhancedDynamicsConfig | None = None,
) -> np.ndarray:
    """Evaluate all configured corrections without mutating ``simulation``.

    The result has shape ``(simulation.N, 3)`` and follows REBOUND particle
    order.  J2 is applied between Earth and every other particle; solar 1PN is
    applied between the Sun and every other particle.  Equal and opposite
    pair forces are included, so ``sum(m_i * a_i)`` is zero to roundoff.
    """

    settings = EnhancedDynamicsConfig() if config is None else config
    settings.validate()
    sun_index = _particle_index(simulation, "sun") if settings.solar_1pn_enabled else -1
    earth_index = (
        _particle_index(simulation, "earth")
        if settings.earth_j2_enabled and settings.earth_j2 != 0.0
        else -1
    )
    return _enhanced_accelerations_unchecked(
        simulation,
        settings,
        sun_index=sun_index,
        earth_index=earth_index,
        spin_axis=settings.normalized_spin_axis(),
    )


def _enhanced_accelerations_unchecked(
    simulation: Any,
    settings: EnhancedDynamicsConfig,
    *,
    sun_index: int,
    earth_index: int,
    spin_axis: np.ndarray,
) -> np.ndarray:
    """Fast force evaluation after attachment-time validation."""

    particles = simulation.particles
    accelerations = np.zeros((simulation.N, 3), dtype=float)

    if settings.earth_j2_enabled and settings.earth_j2 != 0.0:
        earth = particles[earth_index]
        earth_mass = float(earth.m)
        if earth_mass <= 0.0:
            raise ValueError("Earth must have positive mass for the J2 recoil")
        earth_position = _position(earth)
        for index, body in enumerate(particles):
            if index == earth_index:
                continue
            relative_position = _position(body) - earth_position
            correction = _earth_j2_acceleration_unchecked(
                relative_position,
                coefficient=(
                    1.5
                    * settings.earth_j2
                    * settings.earth_mu_km3_s2
                    * settings.earth_equatorial_radius_km**2
                ),
                spin_axis=spin_axis,
            )
            accelerations[index] += correction
            accelerations[earth_index] -= float(body.m) / earth_mass * correction

    if settings.solar_1pn_enabled:
        sun = particles[sun_index]
        sun_mass = float(sun.m)
        if sun_mass <= 0.0:
            raise ValueError("Sun must have positive mass for the 1PN correction")
        sun_position = _position(sun)
        sun_velocity = _velocity(sun)
        gravitational_constant = float(simulation.G)
        for index, body in enumerate(particles):
            if index == sun_index:
                continue
            body_mass = float(body.m)
            total_mass = sun_mass + body_mass
            mass_product = sun_mass * body_mass
            symmetric_mass_ratio = mass_product / total_mass**2
            relative_correction = _solar_1pn_acceleration_unchecked(
                _position(body) - sun_position,
                _velocity(body) - sun_velocity,
                total_mu_km3_s2=gravitational_constant * total_mass,
                symmetric_mass_ratio=symmetric_mass_ratio,
                speed_of_light_squared=settings.speed_of_light_km_s**2,
            )
            # Split the relative acceleration so a_body - a_sun equals the
            # 1PN equation while the pair's added momentum derivative is zero.
            accelerations[index] += sun_mass / total_mass * relative_correction
            accelerations[sun_index] -= body_mass / total_mass * relative_correction

    return accelerations


def attach_enhanced_forces(
    simulation: Any,
    config: EnhancedDynamicsConfig | None = None,
) -> dict[str, Any]:
    """Attach J2/1PN forces to a REBOUND simulation and return metadata.

    Existing additional forces are composed rather than discarded.  Because
    the 1PN term depends on velocity, the REBOUND velocity-dependent-force
    flag is enabled whenever that term is active.
    """

    settings = EnhancedDynamicsConfig() if config is None else config
    settings.validate()
    # Fail before installing a callback if the expected named particles are
    # absent or the state is singular.
    enhanced_accelerations(simulation, settings)
    sun_index = _particle_index(simulation, "sun") if settings.solar_1pn_enabled else -1
    earth_index = (
        _particle_index(simulation, "earth")
        if settings.earth_j2_enabled and settings.earth_j2 != 0.0
        else -1
    )
    spin_axis = settings.normalized_spin_axis()

    existing_field = simulation._additional_forces
    existing_force = None
    if bool(existing_field):
        # REBOUND's `_additional_forces` is a live ctypes field.  Holding that
        # object directly and then assigning a new callback makes it point back
        # at the new callback, causing recursion.  Python-installed callbacks
        # retain an independent owner in `_afp`; C/REBOUNDx callbacks need an
        # independent function pointer copied from the current address.
        existing_force = getattr(simulation, "_afp", None)
        if existing_force is None:  # pragma: no cover - exercised by C plugins
            address = ctypes.cast(existing_field, ctypes.c_void_p).value
            existing_force = type(existing_field)(address)
    if settings.earth_j2_enabled or settings.solar_1pn_enabled:

        def combined_force(simulation_pointer: Any) -> None:
            if existing_force is not None:
                existing_force(simulation_pointer)
            active_simulation = simulation_pointer.contents
            corrections = _enhanced_accelerations_unchecked(
                active_simulation,
                settings,
                sun_index=sun_index,
                earth_index=earth_index,
                spin_axis=spin_axis,
            )
            for particle, correction in zip(active_simulation.particles, corrections, strict=True):
                particle.ax += float(correction[0])
                particle.ay += float(correction[1])
                particle.az += float(correction[2])

        simulation.additional_forces = combined_force

    if settings.solar_1pn_enabled:
        simulation.force_is_velocity_dependent = 1

    return force_model_metadata(settings, composed_with_existing=existing_force is not None)


def attach_reboundx_enhanced_forces(
    simulation: Any,
    config: EnhancedDynamicsConfig | None = None,
) -> dict[str, Any]:
    """Attach compiled Earth-J2 and full first-order N-body GR effects.

    The pure-Python callback above remains useful for equation-level tests and
    installations without REBOUNDx.  Multi-decade production integrations use
    this compiled path: REBOUNDx's ``gravitational_harmonics`` force supplies
    Earth's arbitrarily oriented J2 term, while ``gr_full`` supplies the full
    first-order post-Newtonian interaction among all active bodies.
    """

    try:
        import rebound
        import reboundx
    except ImportError as exc:  # pragma: no cover - dependency is required in production
        raise RuntimeError(
            "The enhanced production ephemeris requires the reboundx package"
        ) from exc

    settings = EnhancedDynamicsConfig() if config is None else config
    settings.validate()
    extras = getattr(simulation, "_extras_ref", None)
    if extras is None:
        extras = reboundx.Extras(simulation)
        simulation._extras_ref = extras
    attached: list[str] = []

    if settings.earth_j2_enabled and settings.earth_j2 != 0.0:
        harmonics = extras.load_force("gravitational_harmonics")
        extras.add_force(harmonics)
        earth = simulation.particles["earth"]
        earth.params["J2"] = settings.earth_j2
        earth.params["R_eq"] = settings.earth_equatorial_radius_km
        earth.params["Omega"] = rebound.Vec3d(settings.normalized_spin_axis())
        simulation._gravitational_harmonics_force_ref = harmonics
        attached.append("gravitational_harmonics")

    if settings.solar_1pn_enabled:
        relativity = extras.load_force("gr_full")
        relativity.params["c"] = settings.speed_of_light_km_s
        extras.add_force(relativity)
        simulation._gr_full_force_ref = relativity
        attached.append("gr_full")

    return {
        "gravity": "fully coupled Newtonian point masses plus compiled corrections",
        "implementation": "REBOUNDx compiled C force modules",
        "rebound_version": rebound.__version__,
        "reboundx_version": reboundx.__version__,
        "attached_effects": attached,
        "earth_j2": {
            "enabled": settings.earth_j2_enabled,
            "effect": "gravitational_harmonics",
            "j2": settings.earth_j2,
            "equatorial_radius_km": settings.earth_equatorial_radius_km,
            "spin_axis_icrf": list(settings.earth_spin_axis_icrf),
            "targets": "all other active bodies",
            "axis_semantics": (
                "reads the shared REBOUNDx Omega vector; tides_spin may evolve its "
                "direction when coupled spin evolution is enabled"
            ),
            "angular_momentum_limit": (
                "gravitational_harmonics omits the equal-and-opposite source-spin "
                "reaction torque; strict orbital-plus-spin conservation requires J2 off"
            ),
        },
        "first_post_newtonian": {
            "enabled": settings.solar_1pn_enabled,
            "effect": "gr_full",
            "scope": "full first-order post-Newtonian interactions among all active bodies",
            "speed_of_light_km_s": settings.speed_of_light_km_s,
        },
        "energy_diagnostic": (
            "Use the REBOUNDx gr_full Hamiltonian plus the gravitational-harmonics "
            "potential for conservative-only audits; the separately attached tide is dissipative."
        ),
        "remaining_omissions": [
            "Earth J2 source-spin reaction torque, free nutation, and polar motion",
            "lunar and second-moon permanent-figure terms",
            "post-1PN relativity, solar frame dragging, and solar quadrupole",
        ],
    }


def build_enhanced_coupled_simulation(
    context: Any,
    elements: OrbitalElements,
    *,
    config: EnhancedDynamicsConfig | None = None,
    second_moon_state_icrf: Any | None = None,
    binary_architecture: BinaryMoonArchitecture | None = None,
    ias15_epsilon: float = 1.0e-10,
) -> tuple[Any, dict[str, Any]]:
    """Build the existing coupled simulation and attach enhanced forces."""

    # The local import keeps the pure force equations usable even when the
    # optional ephemeris/REBOUND stack has not yet been initialized.
    from .stability import build_coupled_simulation

    simulation, metadata = build_coupled_simulation(
        context,
        elements,
        second_moon_state_icrf=second_moon_state_icrf,
        binary_architecture=binary_architecture,
        ias15_epsilon=ias15_epsilon,
    )
    force_metadata = attach_enhanced_forces(simulation, config)
    return simulation, {**metadata, "force_model": force_metadata}


def force_model_metadata(
    config: EnhancedDynamicsConfig,
    *,
    composed_with_existing: bool = False,
) -> dict[str, Any]:
    """Describe the exact force model and its interpretation boundaries."""

    return {
        "gravity": "fully coupled Newtonian point masses plus optional corrections",
        "implementation": "native REBOUND additional_forces callback",
        "earth_j2": {
            "enabled": config.earth_j2_enabled,
            "j2": config.earth_j2,
            "equatorial_radius_km": config.earth_equatorial_radius_km,
            "mu_km3_s2": config.earth_mu_km3_s2,
            "spin_axis_icrf": list(config.earth_spin_axis_icrf),
            "targets": "every non-Earth particle with Earth recoil",
        },
        "solar_1pn": {
            "enabled": config.solar_1pn_enabled,
            "form": "pairwise harmonic-coordinate 1PN relative equation",
            "targets": "every non-Sun particle with barycentric pair splitting",
            "speed_of_light_km_s": config.speed_of_light_km_s,
            "scope": "solar-monopole approximation; not full EIH N-body 1PN",
        },
        "composed_with_existing_additional_forces": composed_with_existing,
        "energy_diagnostic": (
            "REBOUND Simulation.energy() contains the Newtonian point-mass Hamiltonian only; "
            "its drift is not a conserved-energy error once J2 or 1PN forces are enabled."
        ),
        "remaining_omissions": [
            "time-dependent terrestrial pole precession, nutation, and polar motion",
            "solid-Earth and ocean tides",
            "tidal dissipation and spin evolution",
            "lunar and second-moon figure terms",
            "full Einstein-Infeld-Hoffmann N-body relativity",
            "solar frame dragging and quadrupole",
        ],
    }


def _earth_j2_acceleration_unchecked(
    position: np.ndarray,
    *,
    coefficient: float,
    spin_axis: np.ndarray,
) -> np.ndarray:
    radius_squared = float(np.dot(position, position))
    radius = math.sqrt(radius_squared)
    axial_distance = float(np.dot(position, spin_axis))
    return coefficient / radius**5 * (
        (5.0 * axial_distance**2 / radius_squared - 1.0) * position
        - 2.0 * axial_distance * spin_axis
    )


def _solar_1pn_acceleration_unchecked(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    total_mu_km3_s2: float,
    symmetric_mass_ratio: float,
    speed_of_light_squared: float,
) -> np.ndarray:
    radius_squared = float(np.dot(position, position))
    radius = math.sqrt(radius_squared)
    position_velocity = float(np.dot(position, velocity))
    velocity_squared = float(np.dot(velocity, velocity))
    radial_velocity = position_velocity / radius
    radial_coefficient = (
        (4.0 + 2.0 * symmetric_mass_ratio) * total_mu_km3_s2 / radius
        - (1.0 + 3.0 * symmetric_mass_ratio) * velocity_squared
        + 1.5 * symmetric_mass_ratio * radial_velocity**2
    )
    velocity_coefficient = (4.0 - 2.0 * symmetric_mass_ratio) * position_velocity
    return total_mu_km3_s2 / (speed_of_light_squared * radius**3) * (
        radial_coefficient * position + velocity_coefficient * velocity
    )


def _particle_index(simulation: Any, name: str) -> int:
    try:
        particle = simulation.particles[name]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"simulation requires a named {name!r} particle") from exc
    index = getattr(particle, "index", None)
    if index is None:  # pragma: no cover - compatibility with older REBOUND
        for candidate_index, candidate in enumerate(simulation.particles):
            if candidate == particle:
                return candidate_index
        raise ValueError(f"could not determine the particle index for {name!r}")
    return int(index)


def _position(particle: Any) -> np.ndarray:
    return np.asarray((particle.x, particle.y, particle.z), dtype=float)


def _velocity(particle: Any) -> np.ndarray:
    return np.asarray((particle.vx, particle.vy, particle.vz), dtype=float)


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector
