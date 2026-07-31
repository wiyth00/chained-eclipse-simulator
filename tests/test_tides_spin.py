"""Tests for the explicitly optional constant-time-lag tide/spin model."""

import json
import math

import numpy as np
import pytest

from chained_eclipse.constants import (
    EARTH_MASS_KG,
    REAL_MOON_MASS_KG,
    SECOND_MOON_MASS_KG,
)
from chained_eclipse.tides_spin import (
    EARTH_SIDEREAL_DAY_S,
    NOMINAL_LUNAR_RECESSION_M_PER_YEAR,
    NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM,
    TidalBody,
    calibrate_earth_k2_delta_t_s,
    circular_recession_rate_km_s,
    compare_rotation_histories,
    integrate_sampled_earth_rotation,
    main,
    mean_motion_rad_s,
    mignard_pair_effect,
    simulate_secular_tidal_spin,
)


def test_calibration_recovers_nominal_lunar_recession() -> None:
    spin = 2.0 * math.pi / EARTH_SIDEREAL_DAY_S
    k2_lag = calibrate_earth_k2_delta_t_s(earth_spin_rad_s=spin)
    moon = TidalBody(
        "real_moon", REAL_MOON_MASS_KG, NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM
    )
    recession_m_per_year = (
        circular_recession_rate_km_s(moon, spin, k2_lag)
        * 365.25
        * 86_400.0
        * 1_000.0
    )

    assert k2_lag == pytest.approx(181.2844847, rel=1.0e-8)
    assert recession_m_per_year == pytest.approx(
        NOMINAL_LUNAR_RECESSION_M_PER_YEAR, rel=1.0e-13
    )


def test_mignard_pair_force_balances_linear_and_angular_momentum() -> None:
    position = np.asarray((180_000.0, 4_000.0, -2_000.0))
    velocity = np.asarray((-0.03, 1.45, 0.08))
    effect = mignard_pair_effect(
        position,
        velocity,
        satellite_mass_kg=SECOND_MOON_MASS_KG,
        primary_spin_vector_rad_s=(0.0, 0.0, 2.0 * math.pi / EARTH_SIDEREAL_DAY_S),
        k2_delta_t_s=calibrate_earth_k2_delta_t_s(),
    )

    net_force = (
        EARTH_MASS_KG * effect.primary_acceleration_km_s2
        + SECOND_MOON_MASS_KG * effect.satellite_acceleration_km_s2
    )
    orbital_torque = np.cross(position, effect.force_on_satellite_kg_km_s2)

    np.testing.assert_allclose(net_force, np.zeros(3), rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(
        orbital_torque + effect.primary_spin_torque_kg_km2_s2,
        np.zeros(3),
        rtol=0.0,
        atol=2.0,
    )


@pytest.mark.parametrize(
    ("position", "velocity", "spin"),
    [
        (
            (40_000.0, 0.0, 0.0),
            (0.02, 0.70, 0.04),
            (0.0, 0.0, 1.8e-5),
        ),
        (
            (32_000.0, -18_000.0, 5_000.0),
            (-0.12, -0.4, 0.2),
            (2.0e-6, -4.0e-6, 1.0e-5),
        ),
    ],
)
def test_mignard_tide_dissipates_or_reaches_zero_limit(
    position,
    velocity,
    spin,
) -> None:
    position_vector = np.asarray(position)
    velocity_vector = np.asarray(velocity)
    spin_vector = np.asarray(spin)
    effect = mignard_pair_effect(
        position_vector,
        velocity_vector,
        satellite_mass_kg=REAL_MOON_MASS_KG,
        primary_mass_kg=SECOND_MOON_MASS_KG,
        primary_radius_km=2_514.0,
        primary_spin_vector_rad_s=spin_vector,
        k2_delta_t_s=100.0,
    )
    mechanical_power = float(
        np.dot(effect.force_on_satellite_kg_km_s2, velocity_vector)
        + np.dot(effect.primary_spin_torque_kg_km2_s2, spin_vector)
    )

    assert mechanical_power < 0.0

    disabled = mignard_pair_effect(
        position_vector,
        velocity_vector,
        satellite_mass_kg=REAL_MOON_MASS_KG,
        primary_mass_kg=SECOND_MOON_MASS_KG,
        primary_radius_km=2_514.0,
        primary_spin_vector_rad_s=spin_vector,
        k2_delta_t_s=0.0,
    )
    np.testing.assert_array_equal(
        disabled.force_on_satellite_kg_km_s2,
        np.zeros(3),
    )
    np.testing.assert_array_equal(
        disabled.primary_spin_torque_kg_km2_s2,
        np.zeros(3),
    )


def test_sampled_spin_history_produces_differential_ground_track_correction() -> None:
    # Circular synthetic trajectories exercise the sampled force/torque API,
    # independently of the secular da/dt integrator.
    times = np.arange(0.0, 40.0 * 86_400.0 + 3_600.0, 3_600.0)
    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    masses = {
        "real_moon": REAL_MOON_MASS_KG,
        "second_moon": SECOND_MOON_MASS_KG,
    }
    for name, mass, axis in (
        ("real_moon", REAL_MOON_MASS_KG, NOMINAL_LUNAR_SEMIMAJOR_AXIS_KM),
        ("second_moon", SECOND_MOON_MASS_KG, 180_000.0),
    ):
        motion = mean_motion_rad_s(axis, mass)
        phase = motion * times
        positions[name] = np.column_stack(
            (axis * np.cos(phase), axis * np.sin(phase), np.zeros_like(phase))
        )
        velocities[name] = np.column_stack(
            (
                -axis * motion * np.sin(phase),
                axis * motion * np.cos(phase),
                np.zeros_like(phase),
            )
        )

    two_moon = integrate_sampled_earth_rotation(times, positions, velocities, masses)
    baseline = integrate_sampled_earth_rotation(
        times,
        positions,
        velocities,
        masses,
        included_bodies=("real_moon",),
    )
    correction = compare_rotation_histories(two_moon, baseline)

    assert two_moon.spin_rate_rad_s[-1] < baseline.spin_rate_rad_s[-1]
    assert correction.delta_mean_solar_lod_ms[-1] > 0.0
    assert correction.delta_ut1_s[-1] < 0.0
    assert correction.earth_fixed_longitude_shift_deg[-1] > 0.0


def test_secular_30_year_audit_conserves_axial_angular_momentum() -> None:
    result = simulate_secular_tidal_spin()
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

    assert max(map(abs, full.relative_axial_angular_momentum_error)) < 1.0e-15
    assert max(map(abs, baseline.relative_axial_angular_momentum_error)) < 1.0e-15
    assert real_recession_m == pytest.approx(1.146, abs=0.002)
    assert second_recession_m == pytest.approx(7.643, abs=0.01)
    assert correction.delta_mean_solar_lod_ms[-1] == pytest.approx(0.691, abs=0.005)
    assert correction.delta_ut1_s[-1] == pytest.approx(-3.785, abs=0.02)
    assert correction.earth_fixed_longitude_shift_deg[-1] == pytest.approx(
        0.01581, abs=5.0e-5
    )
    assert full.orbital_timing_offset_s["second_moon"][-1] == pytest.approx(
        -30.15, abs=0.1
    )
    assert full.mechanical_energy_change_j[-1] < 0.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_cli_writes_strict_json_and_png(tmp_path, capsys) -> None:
    output_dir = tmp_path / "audit"

    exit_code = main(
        (
            "--output-dir",
            str(output_dir),
            "--years",
            "0.2",
            "--sample-days",
            "10",
            "--dpi",
            "72",
        )
    )

    json_path = output_dir / "tidal_spin_audit.json"
    png_path = output_dir / "tidal_spin_audit.png"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["period_years"] == pytest.approx(0.2)
    assert payload["summary"]["second_moon_recession_m"] > 0.0
    assert payload["summary"]["maximum_abs_relative_axial_angular_momentum_error"] < 1e-15
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000
    assert "Tidal audit complete" in output
