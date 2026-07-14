"""Physics and REBOUND-callback checks for the enhanced force model."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

rebound = pytest.importorskip("rebound")

from chained_eclipse.constants import (  # noqa: E402
    EARTH_J2,
    EARTH_MASS_KG,
    MU_EARTH_KM3_S2,
    REAL_MOON_MASS_KG,
    SPEED_OF_LIGHT_KM_S,
    SUN_MASS_KG,
    WGS84_A_KM,
)
from chained_eclipse.enhanced_dynamics import (  # noqa: E402
    EnhancedDynamicsConfig,
    attach_enhanced_forces,
    earth_j2_acceleration,
    enhanced_accelerations,
    solar_1pn_relative_acceleration,
)


def _add_named(simulation, name: str, **kwargs) -> None:
    try:
        simulation.add(name=name, **kwargs)
    except TypeError:
        simulation.add(hash=name, **kwargs)


def _simulation():
    simulation = rebound.Simulation()
    simulation.units = ("s", "km", "kg")
    _add_named(simulation, "sun", m=SUN_MASS_KG)
    _add_named(
        simulation,
        "earth",
        m=EARTH_MASS_KG,
        x=149_597_870.7,
        vy=29.78,
    )
    _add_named(
        simulation,
        "real_moon",
        m=REAL_MOON_MASS_KG,
        x=149_597_870.7 + 384_400.0,
        z=12_000.0,
        vy=29.78 + 1.02,
    )
    _add_named(
        simulation,
        "second_moon",
        m=2.0e22,
        x=149_597_870.7 + 180_000.0,
        y=10_000.0,
        vy=29.78 + 1.48,
        vz=0.08,
    )
    simulation.move_to_com()
    simulation.integrator = "ias15"
    return simulation


def test_j2_equatorial_and_polar_limits() -> None:
    distance = 20_000.0
    scale = EARTH_J2 * MU_EARTH_KM3_S2 * WGS84_A_KM**2 / distance**4

    equatorial = earth_j2_acceleration(np.asarray((distance, 0.0, 0.0)))
    polar = earth_j2_acceleration(np.asarray((0.0, 0.0, distance)))

    assert equatorial == pytest.approx((-1.5 * scale, 0.0, 0.0), rel=1.0e-14)
    assert polar == pytest.approx((0.0, 0.0, 3.0 * scale), rel=1.0e-14)


def test_j2_follows_configured_spin_axis() -> None:
    distance = 25_000.0
    along_x_pole = earth_j2_acceleration(
        np.asarray((distance, 0.0, 0.0)),
        spin_axis=np.asarray((2.0, 0.0, 0.0)),
    )
    expected = 3.0 * EARTH_J2 * MU_EARTH_KM3_S2 * WGS84_A_KM**2 / distance**4

    assert along_x_pole == pytest.approx((expected, 0.0, 0.0), rel=1.0e-14)


def test_solar_1pn_circular_test_particle_limit() -> None:
    radius = 149_597_870.7
    mu = 132_712_440_041.939_38
    speed = np.sqrt(mu / radius)
    correction = solar_1pn_relative_acceleration(
        np.asarray((radius, 0.0, 0.0)),
        np.asarray((0.0, speed, 0.0)),
        total_mu_km3_s2=mu,
    )
    expected = 3.0 * mu**2 / (SPEED_OF_LIGHT_KM_S**2 * radius**3)

    assert correction == pytest.approx((expected, 0.0, 0.0), rel=2.0e-15)


def test_added_pair_forces_conserve_linear_momentum() -> None:
    simulation = _simulation()
    accelerations = enhanced_accelerations(simulation)
    forces = np.asarray(
        [particle.m * accelerations[index] for index, particle in enumerate(simulation.particles)]
    )
    residual = float(np.linalg.norm(np.sum(forces, axis=0)))
    force_scale = float(np.sum(np.linalg.norm(forces, axis=1)))

    assert force_scale > 0.0
    assert residual / force_scale < 2.0e-15


def test_rebound_callback_adds_expected_accelerations() -> None:
    simulation = _simulation()
    expected = enhanced_accelerations(simulation)
    metadata = attach_enhanced_forces(simulation)
    for particle in simulation.particles:
        particle.ax = particle.ay = particle.az = 0.0

    simulation._additional_forces(ctypes.pointer(simulation))
    actual = np.asarray(
        [(particle.ax, particle.ay, particle.az) for particle in simulation.particles]
    )

    assert actual == pytest.approx(expected, rel=1.0e-14, abs=1.0e-30)
    assert simulation.force_is_velocity_dependent == 1
    assert metadata["solar_1pn"]["scope"].startswith("solar-monopole")


def test_solar_1pn_reproduces_mercury_precession() -> None:
    simulation = rebound.Simulation()
    simulation.units = ("s", "km", "kg")
    _add_named(simulation, "sun", m=SUN_MASS_KG)
    semimajor_axis = 57_909_050.0
    eccentricity = 0.205_630
    _add_named(
        simulation,
        "mercury",
        m=3.3011e23,
        a=semimajor_axis,
        e=eccentricity,
        primary=simulation.particles["sun"],
    )
    simulation.move_to_com()
    simulation.integrator = "ias15"
    if hasattr(simulation, "ri_ias15"):
        simulation.ri_ias15.epsilon = 1.0e-11
    else:
        simulation.integrator.epsilon = 1.0e-11
    attach_enhanced_forces(
        simulation,
        EnhancedDynamicsConfig(earth_j2_enabled=False),
    )
    total_mu = simulation.G * sum(particle.m for particle in simulation.particles)
    period = 2.0 * np.pi * np.sqrt(semimajor_axis**3 / total_mu)
    periapsis_longitudes = []
    orbit_numbers = np.arange(11)
    for orbit_number in orbit_numbers:
        simulation.integrate(float(orbit_number * period), exact_finish_time=1)
        orbit = simulation.particles["mercury"].orbit(primary=simulation.particles["sun"])
        periapsis_longitudes.append(orbit.pomega)

    measured_per_orbit = np.polyfit(
        orbit_numbers,
        np.unwrap(periapsis_longitudes),
        1,
    )[0]
    analytic_per_orbit = (
        6.0
        * np.pi
        * total_mu
        / (
            semimajor_axis
            * (1.0 - eccentricity**2)
            * SPEED_OF_LIGHT_KM_S**2
        )
    )

    assert measured_per_orbit == pytest.approx(analytic_per_orbit, rel=3.0e-5)


def test_existing_rebound_force_is_composed() -> None:
    simulation = _simulation()
    prior_acceleration = 2.5e-9

    def prior_force(simulation_pointer) -> None:
        simulation_pointer.contents.particles["second_moon"].ay += prior_acceleration

    simulation.additional_forces = prior_force
    settings = EnhancedDynamicsConfig(solar_1pn_enabled=False)
    expected = enhanced_accelerations(simulation, settings)
    metadata = attach_enhanced_forces(simulation, settings)
    for particle in simulation.particles:
        particle.ax = particle.ay = particle.az = 0.0

    simulation._additional_forces(ctypes.pointer(simulation))
    actual = np.asarray(
        [(particle.ax, particle.ay, particle.az) for particle in simulation.particles]
    )
    expected[simulation.particles["second_moon"].index, 1] += prior_acceleration

    assert actual == pytest.approx(expected, rel=1.0e-14, abs=1.0e-30)
    assert metadata["composed_with_existing_additional_forces"] is True


@pytest.mark.parametrize(
    "settings, message",
    [
        (EnhancedDynamicsConfig(earth_j2=-1.0), "earth_j2"),
        (EnhancedDynamicsConfig(earth_spin_axis_icrf=(0.0, 0.0, 0.0)), "spin_axis"),
        (EnhancedDynamicsConfig(speed_of_light_km_s=0.0), "speed_of_light"),
    ],
)
def test_configuration_rejects_non_physical_values(settings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings.validate()
