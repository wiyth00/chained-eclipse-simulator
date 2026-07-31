from __future__ import annotations

import math

import numpy as np
import pytest

from chained_eclipse.constants import EARTH_MASS_KG, REAL_MOON_MASS_KG
from chained_eclipse.rotational_diagnostics import (
    relative_vector_change,
    rotational_diagnostic_snapshot,
)
from chained_eclipse.rotational_dynamics import (
    NBodyTideConfig,
    RotationalBodyConfig,
    attach_reboundx_rotational_tides,
)
from chained_eclipse.tides_spin import G_KM3_KG_S2


def _isolated_earth_moon(*, lag_s: float):
    rebound = pytest.importorskip("rebound")
    simulation = rebound.Simulation()
    simulation.G = G_KM3_KG_S2
    semimajor_axis_km = 100_000.0
    mean_motion = math.sqrt(
        simulation.G
        * (EARTH_MASS_KG + REAL_MOON_MASS_KG)
        / semimajor_axis_km**3
    )
    simulation.add(m=EARTH_MASS_KG, r=6_378.137, hash="earth")
    simulation.add(
        m=REAL_MOON_MASS_KG,
        r=1_737.4,
        a=semimajor_axis_km,
        e=0.04,
        primary=simulation.particles["earth"],
        hash="real_moon",
    )
    simulation.move_to_com()
    simulation.integrator = "ias15"
    simulation.ri_ias15.epsilon = 1.0e-12
    config = NBodyTideConfig(
        active_scenario=f"isolated_lag_{lag_s:g}",
        structured_bodies=(
            RotationalBodyConfig(
                name="earth",
                radius_km=6_378.137,
                polar_moment_factor=0.3307,
                initial_spin_vector_rad_s=(0.0, 0.0, 7.2921150e-5),
                love_number_k2=0.3,
                constant_time_lag_s=lag_s,
                scenario_label="isolated_test",
                provenance="deterministic conservation test",
            ),
            RotationalBodyConfig(
                name="real_moon",
                radius_km=1_737.4,
                polar_moment_factor=0.3931,
                initial_spin_vector_rad_s=(0.0, 0.0, mean_motion),
                love_number_k2=0.024,
                constant_time_lag_s=lag_s,
                scenario_label="isolated_test",
                provenance="deterministic conservation test",
            ),
        ),
    )
    attach_reboundx_rotational_tides(simulation, config)
    return simulation


def _linear_momentum(simulation) -> np.ndarray:
    return np.sum(
        [
            particle.m
            * np.asarray((particle.vx, particle.vy, particle.vz), dtype=float)
            for particle in simulation.particles
        ],
        axis=0,
    )


def test_compiled_tides_conserve_internal_momentum_and_dissipate_energy() -> None:
    simulation = _isolated_earth_moon(lag_s=10_000.0)
    initial = rotational_diagnostic_snapshot(simulation)
    initial_momentum = _linear_momentum(simulation)

    simulation.integrate(40.0 * 86_400.0, exact_finish_time=1)
    final = rotational_diagnostic_snapshot(simulation)
    final_momentum = _linear_momentum(simulation)

    assert relative_vector_change(
        initial.total_angular_momentum_kg_km2_s,
        final.total_angular_momentum_kg_km2_s,
    ) < 2.0e-11
    momentum_scale = EARTH_MASS_KG * 1.0
    assert np.linalg.norm(final_momentum - initial_momentum) / momentum_scale < 1.0e-12
    assert final.mechanical_energy_kg_km2_s2 < initial.mechanical_energy_kg_km2_s2


def test_zero_lag_recovers_conservative_compiled_limit() -> None:
    simulation = _isolated_earth_moon(lag_s=0.0)
    initial = rotational_diagnostic_snapshot(simulation)

    simulation.integrate(40.0 * 86_400.0, exact_finish_time=1)
    final = rotational_diagnostic_snapshot(simulation)

    assert relative_vector_change(
        initial.total_angular_momentum_kg_km2_s,
        final.total_angular_momentum_kg_km2_s,
    ) < 2.0e-11
    fractional_energy_change = abs(
        (
            final.mechanical_energy_kg_km2_s2
            - initial.mechanical_energy_kg_km2_s2
        )
        / initial.mechanical_energy_kg_km2_s2
    )
    assert fractional_energy_change < 2.0e-11
