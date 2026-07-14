"""Integration checks for the enhanced planetary/tidal eclipse trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rebound
import yaml

from chained_eclipse.constants import (
    EARTH_MASS_KG,
    REAL_MOON_MASS_KG,
    WGS84_A_KM,
)
from chained_eclipse.enhanced_ephemeris import (
    EnhancedEphemeris,
    NBodyTideConfig,
    attach_reboundx_earth_tides,
    earth_tidal_accelerations,
)
from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.models import OrbitalElements
from chained_eclipse.planetary_dynamics import build_planetary_simulation
from chained_eclipse.tides_spin import calibrate_earth_k2_delta_t_s


ROOT = Path(__file__).resolve().parents[1]


def _elements() -> OrbitalElements:
    payload = yaml.safe_load((ROOT / "config" / "optimized_system.yaml").read_text())
    return OrbitalElements(**payload["orbital_elements"])


def test_tidal_callback_accelerations_balance_force_and_spin_torque() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    simulation, _ = build_planetary_simulation(context, _elements())
    settings = NBodyTideConfig()
    k2_lag = calibrate_earth_k2_delta_t_s(
        moon_mass_kg=float(simulation.particles["real_moon"].m),
        earth_mass_kg=float(simulation.particles["earth"].m),
    )
    accelerations, spin_torque = earth_tidal_accelerations(
        simulation,
        settings,
        k2_delta_t_s=k2_lag,
    )
    forces = np.asarray(
        [particle.m * accelerations[index] for index, particle in enumerate(simulation.particles)]
    )
    force_scale = float(np.sum(np.linalg.norm(forces, axis=1)))
    net_force = np.sum(forces, axis=0)
    earth = simulation.particles["earth"]
    earth_position = np.asarray((earth.x, earth.y, earth.z))
    orbital_torque = np.zeros(3)
    for name in settings.satellite_names:
        satellite = simulation.particles[name]
        relative_position = np.asarray((satellite.x, satellite.y, satellite.z)) - earth_position
        orbital_torque += np.cross(
            relative_position,
            satellite.m * accelerations[int(satellite.index)],
        )

    assert force_scale > 0.0
    assert np.linalg.norm(net_force) / force_scale < 2.0e-15
    assert orbital_torque == pytest.approx(-spin_torque, rel=2.0e-14)


def test_enhanced_ephemeris_builds_rotation_correction_and_strict_metadata() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    ephemeris = EnhancedEphemeris(
        context,
        _elements(),
        "2026-07-12T00:00:00Z",
        sample_step_seconds=3_600.0,
    )
    rotation = ephemeris.metadata["alternate_earth_rotation"]
    final_offset = float(ephemeris.longitude_offset_deg(ephemeris.end_tt_jd))

    assert ephemeris.metadata["dynamics_model"] == "enhanced"
    assert ephemeris.metadata["planetary_perturbers"]["bodies"] == [
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ]
    assert rotation["final_delta_mean_solar_lod_ms"] > 0.0
    assert 0.0 < final_offset < 1.0e-5
    assert final_offset == pytest.approx(rotation["final_longitude_shift_deg"])
    assert ephemeris.longitude_offset_deg(ephemeris.epoch_tt_jd) == pytest.approx(0.0)
    assert np.all(np.isfinite(ephemeris.relative("second_moon", ephemeris.end_tt_jd)))
    rotation = ephemeris.rotation_matrix_icrf_to_itrs(ephemeris.end_tt_jd)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=2.0e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2.0e-14)
    json.dumps(ephemeris.metadata, allow_nan=False)


def test_enhanced_ephemeris_reuses_content_addressed_trajectory_cache(
    tmp_path: Path,
) -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    first = EnhancedEphemeris(
        context,
        _elements(),
        "2026-07-10T03:00:00Z",
        sample_step_seconds=3_600.0,
        trajectory_cache_dir=tmp_path,
    )
    second = EnhancedEphemeris(
        context,
        _elements(),
        "2026-07-10T03:00:00Z",
        sample_step_seconds=3_600.0,
        trajectory_cache_dir=tmp_path,
    )

    assert first.metadata["trajectory_cache"]["hit"] is False
    assert second.metadata["trajectory_cache"]["hit"] is True
    assert first.metadata["trajectory_cache"]["path"] == second.metadata[
        "trajectory_cache"
    ]["path"]
    assert Path(first.metadata["trajectory_cache"]["path"]).exists()
    assert second.relative("real_moon", second.end_tt_jd) == pytest.approx(
        first.relative("real_moon", first.end_tt_jd)
    )


def test_reboundx_tide_normalization_recovers_lunar_recession() -> None:
    simulation = rebound.Simulation()
    simulation.units = ("s", "km", "kg")
    try:
        simulation.add(m=EARTH_MASS_KG, r=WGS84_A_KM, name="earth")
        simulation.add(
            m=REAL_MOON_MASS_KG,
            a=384_400.0,
            primary=simulation.particles["earth"],
            name="real_moon",
        )
    except TypeError:
        simulation.add(m=EARTH_MASS_KG, r=WGS84_A_KM, hash="earth")
        simulation.add(
            m=REAL_MOON_MASS_KG,
            a=384_400.0,
            primary=simulation.particles["earth"],
            hash="real_moon",
        )
    simulation.move_to_com()
    simulation.integrator = "ias15"
    simulation.ri_ias15.epsilon = 1.0e-11
    metadata = attach_reboundx_earth_tides(simulation)
    initial_axis = simulation.particles["real_moon"].orbit(
        primary=simulation.particles["earth"]
    ).a

    simulation.integrate(10.0 * 365.25 * 86_400.0, exact_finish_time=1)

    final_axis = simulation.particles["real_moon"].orbit(
        primary=simulation.particles["earth"]
    ).a
    recession_m_per_year = (final_axis - initial_axis) * 1_000.0 / 10.0
    assert recession_m_per_year == pytest.approx(0.0382, abs=2.0e-5)
    assert metadata["reboundx_normalization_factor"] == 0.5
