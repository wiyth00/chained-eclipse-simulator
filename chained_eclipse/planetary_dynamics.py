"""DE440-initialized major-planet perturbations for the coupled moon model.

The project's original coupled simulation contains the Sun, Earth, real Moon,
and hypothetical second moon.  This module adds the seven other major planets
as *active* REBOUND point masses.  Their ICRF/BCRS Cartesian states are sampled
from DE440s at the orbital epoch; after that instant the complete system is
integrated self-consistently rather than being forced to follow DE440s.

Mercury and Venus use their planet-centre states.  Mars through Neptune use
planetary-system barycentres and matching system gravitational parameters, so
their unmodeled natural satellites are not silently discarded or double
counted.  Pluto is available as an explicitly optional dwarf-planet system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .ephemeris import EphemerisContext
from .moon_architecture import BinaryMoonArchitecture
from .models import OrbitalElements
from .stability import build_coupled_simulation


JPL_ASTRODYNAMIC_PARAMETERS_URL = "https://ssd.jpl.nasa.gov/astro_par.html"


@dataclass(frozen=True, slots=True)
class PlanetaryPointMass:
    """A DE440 state target paired with its DE440-compatible system GM."""

    name: str
    ephemeris_target: str
    gm_km3_s2: float
    representation: str
    major_planet: bool = True


# These gravitational parameters are the values JPL publishes for DE440.  The
# outer-planet and Mars values include each planet's satellite system, matching
# the barycentric states carried by the compact DE440s kernel.
PLANETARY_POINT_MASSES: dict[str, PlanetaryPointMass] = {
    "mercury": PlanetaryPointMass(
        "mercury", "mercury", 22_031.868_551, "planet centre"
    ),
    "venus": PlanetaryPointMass(
        "venus", "venus", 324_858.592_000, "planet centre"
    ),
    "mars": PlanetaryPointMass(
        "mars", "mars barycenter", 42_828.375_816, "planetary-system barycentre"
    ),
    "jupiter": PlanetaryPointMass(
        "jupiter",
        "jupiter barycenter",
        126_712_764.100_000,
        "planetary-system barycentre",
    ),
    "saturn": PlanetaryPointMass(
        "saturn",
        "saturn barycenter",
        37_940_584.841_800,
        "planetary-system barycentre",
    ),
    "uranus": PlanetaryPointMass(
        "uranus",
        "uranus barycenter",
        5_794_556.400_000,
        "planetary-system barycentre",
    ),
    "neptune": PlanetaryPointMass(
        "neptune",
        "neptune barycenter",
        6_836_527.100_580,
        "planetary-system barycentre",
    ),
    "pluto": PlanetaryPointMass(
        "pluto",
        "pluto barycenter",
        975.500_000,
        "dwarf-planet-system barycentre",
        major_planet=False,
    ),
}

DEFAULT_MAJOR_PLANETS: tuple[str, ...] = tuple(
    name for name, body in PLANETARY_POINT_MASSES.items() if body.major_planet
)


def add_planetary_perturbers(
    simulation: Any,
    context: EphemerisContext,
    epoch_utc: str,
    *,
    body_names: Sequence[str] = DEFAULT_MAJOR_PLANETS,
) -> dict[str, Any]:
    """Add selected DE440-initialized point masses to a REBOUND simulation.

    The input simulation must still be at its epoch and must already contain a
    particle named ``sun``.  It may already have been translated to its centre
    of mass: the translation is recovered by comparing that particle with the
    DE440 Sun state, applied to every new state, and then recomputed after all
    requested bodies have been added.  Thus all barycentric relative states are
    retained to floating-point precision.
    """

    if abs(float(simulation.t)) > 1.0e-12:
        raise ValueError("planetary perturbers must be added at the simulation epoch")

    requested = tuple(str(name).lower() for name in body_names)
    if len(set(requested)) != len(requested):
        raise ValueError("planetary body names must be unique")
    unknown = sorted(set(requested) - PLANETARY_POINT_MASSES.keys())
    if unknown:
        raise ValueError(f"unknown planetary point masses: {', '.join(unknown)}")

    epoch = context.time_utc(epoch_utc)
    sun_de440 = _skyfield_state(context.sun, epoch)
    sun_particle = simulation.particles["sun"]
    com_translation = _particle_state(sun_particle) - sun_de440

    masses_kg: dict[str, float] = {}
    initial_states_bcrs: dict[str, list[float]] = {}
    targets: dict[str, str] = {}
    representations: dict[str, str] = {}
    for name in requested:
        spec = PLANETARY_POINT_MASSES[name]
        state_bcrs = _skyfield_state(context.ephemeris[spec.ephemeris_target], epoch)
        state_simulation = state_bcrs + com_translation
        mass_kg = float(spec.gm_km3_s2 / simulation.G)
        _add_named_point_mass(simulation, name, mass_kg, state_simulation)
        masses_kg[name] = mass_kg
        initial_states_bcrs[name] = state_bcrs.tolist()
        targets[name] = spec.ephemeris_target
        representations[name] = spec.representation

    # New active masses change the full system's barycentre.  This translation
    # leaves all relative positions and velocities invariant.
    simulation.move_to_com()

    return {
        "active": True,
        "epoch_utc": epoch_utc,
        "frame": "DE440 BCRS/ICRF barycentric Cartesian states",
        "state_source": str(getattr(context, "kernel_path", "DE440s")),
        "mass_source": JPL_ASTRODYNAMIC_PARAMETERS_URL,
        "bodies": list(requested),
        "ephemeris_targets": targets,
        "representations": representations,
        "gm_km3_s2": {
            name: PLANETARY_POINT_MASSES[name].gm_km3_s2 for name in requested
        },
        "masses_kg": masses_kg,
        "initial_states_bcrs_km_km_s": initial_states_bcrs,
        "particle_radius_km": 0.0,
        "notes": [
            "All added bodies gravitationally perturb every other active particle.",
            "Planetary states are initialized from DE440s only at the epoch; no later "
            "ephemeris forcing is applied.",
            "System barycentre particles use system GMs that include unmodeled satellites.",
            "Zero collision radii avoid treating a planetary-system barycentre as a solid body.",
        ],
    }


def build_planetary_simulation(
    context: EphemerisContext,
    elements: OrbitalElements,
    *,
    real_moon_state_icrf: Any | None = None,
    second_moon_state_icrf: Any | None = None,
    binary_architecture: BinaryMoonArchitecture | None = None,
    ias15_epsilon: float = 1.0e-10,
    include_pluto: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build the coupled two-moon model with all other major planets active."""

    simulation, metadata = build_coupled_simulation(
        context,
        elements,
        real_moon_state_icrf=real_moon_state_icrf,
        second_moon_state_icrf=second_moon_state_icrf,
        binary_architecture=binary_architecture,
        ias15_epsilon=ias15_epsilon,
    )
    names = DEFAULT_MAJOR_PLANETS + (("pluto",) if include_pluto else ())
    planetary_metadata = add_planetary_perturbers(
        simulation,
        context,
        elements.epoch_utc,
        body_names=names,
    )
    metadata = {
        **metadata,
        "planetary_perturbers": planetary_metadata,
        "force_model": (
            "Fully coupled Newtonian point-mass Sun/Earth/real Moon/second moon "
            "+ seven other major planets"
            + (" + Pluto system" if include_pluto else "")
        ),
    }
    return simulation, metadata


def _skyfield_state(body: Any, time: Any) -> np.ndarray:
    state = body.at(time)
    return np.concatenate(
        (
            np.asarray(state.position.km, dtype=float),
            np.asarray(state.velocity.km_per_s, dtype=float),
        )
    )


def _particle_state(particle: Any) -> np.ndarray:
    return np.asarray(
        (particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz),
        dtype=float,
    )


def _add_named_point_mass(
    simulation: Any,
    name: str,
    mass_kg: float,
    state: np.ndarray,
) -> None:
    kwargs = {
        "m": float(mass_kg),
        "r": 0.0,
        "x": float(state[0]),
        "y": float(state[1]),
        "z": float(state[2]),
        "vx": float(state[3]),
        "vy": float(state[4]),
        "vz": float(state[5]),
    }
    try:
        simulation.add(name=name, **kwargs)
    except TypeError:  # pragma: no cover - REBOUND 4 compatibility
        simulation.add(hash=name, **kwargs)
