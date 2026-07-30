"""Checks for DE440-initialized active planetary perturbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.models import OrbitalElements
from chained_eclipse.moon_architecture import architecture_from_config, elements_from_config
from chained_eclipse.planetary_dynamics import (
    DEFAULT_MAJOR_PLANETS,
    PLANETARY_POINT_MASSES,
    add_planetary_perturbers,
    build_planetary_simulation,
)
from chained_eclipse.stability import build_coupled_simulation


ROOT = Path(__file__).resolve().parents[1]


def _elements() -> OrbitalElements:
    payload = yaml.safe_load((ROOT / "config" / "optimized_system.yaml").read_text())
    return OrbitalElements(**payload["orbital_elements"])


def _particle_state(particle) -> np.ndarray:
    return np.asarray(
        (particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz),
        dtype=float,
    )


def test_major_planet_states_and_masses_match_de440() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    elements = _elements()
    simulation, metadata = build_planetary_simulation(context, elements)
    epoch = context.time_utc(elements.epoch_utc)

    assert simulation.N == 11
    assert metadata["planetary_perturbers"]["bodies"] == list(DEFAULT_MAJOR_PLANETS)
    assert "pluto" not in metadata["planetary_perturbers"]["bodies"]

    simulated_sun = _particle_state(simulation.particles["sun"])
    de440_sun = np.concatenate(
        (context.sun.at(epoch).position.km, context.sun.at(epoch).velocity.km_per_s)
    )
    for name in DEFAULT_MAJOR_PLANETS:
        spec = PLANETARY_POINT_MASSES[name]
        target = context.ephemeris[spec.ephemeris_target].at(epoch)
        de440_state = np.concatenate((target.position.km, target.velocity.km_per_s))
        simulated_relative = _particle_state(simulation.particles[name]) - simulated_sun
        de440_relative = de440_state - de440_sun
        assert simulated_relative[:3] == pytest.approx(de440_relative[:3], abs=2.0e-5)
        assert simulated_relative[3:] == pytest.approx(de440_relative[3:], abs=2.0e-11)
        assert simulation.G * simulation.particles[name].m == pytest.approx(
            spec.gm_km3_s2, rel=2.0e-15
        )


def test_planetary_system_integrates_cleanly_for_thirty_days() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    simulation, _ = build_planetary_simulation(context, _elements())
    initial_energy = float(simulation.energy())

    simulation.integrate(30.0 * 86_400.0, exact_finish_time=1)

    final_energy = float(simulation.energy())
    assert abs((final_energy - initial_energy) / initial_energy) < 1.0e-11
    for name in ("sun", "earth", "real_moon", "second_moon", *DEFAULT_MAJOR_PLANETS):
        assert np.all(np.isfinite(_particle_state(simulation.particles[name])))


def test_planetary_system_accepts_bound_binary_initial_conditions() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    payload = yaml.safe_load((ROOT / "config" / "bound_binary_giant.yaml").read_text())
    elements = elements_from_config(payload)
    architecture = architecture_from_config(payload)
    assert architecture is not None

    simulation, metadata = build_planetary_simulation(
        context,
        elements,
        binary_architecture=architecture,
    )
    real = simulation.particles["real_moon"]
    second = simulation.particles["second_moon"]
    separation = np.linalg.norm(_particle_state(second)[:3] - _particle_state(real)[:3])

    assert metadata["architecture"] == "hierarchical binary moons"
    periapsis = architecture.mutual_orbit.semimajor_axis_km * (
        1.0 - architecture.mutual_orbit.eccentricity
    )
    apoapsis = architecture.mutual_orbit.semimajor_axis_km * (
        1.0 + architecture.mutual_orbit.eccentricity
    )
    assert periapsis <= separation <= apoapsis


def test_pluto_is_opt_in_and_late_addition_is_rejected() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    elements = _elements()
    with_pluto, metadata = build_planetary_simulation(
        context, elements, include_pluto=True
    )
    assert with_pluto.N == 12
    assert metadata["planetary_perturbers"]["bodies"][-1] == "pluto"

    base, _ = build_coupled_simulation(context, elements)
    base.integrate(1.0)
    with pytest.raises(ValueError, match="at the simulation epoch"):
        add_planetary_perturbers(base, context, elements.epoch_utc)
