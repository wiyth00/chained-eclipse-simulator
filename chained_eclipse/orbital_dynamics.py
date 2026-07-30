"""Numerical and analytic dynamics for the hypothetical second moon."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, least_squares
from skyfield.api import wgs84

from .constants import (
    EARTH_J2,
    MU_EARTH_KM3_S2,
    MU_MOON_KM3_S2,
    MU_SUN_KM3_S2,
    OBLIQUITY_J2000_DEG,
    SECONDS_PER_DAY,
    WGS84_A_KM,
)
from .ephemeris import EphemerisContext
from .models import OrbitalElements, Trajectory


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


ECLIPTIC_TO_ICRF = _rotation_x(np.radians(OBLIQUITY_J2000_DEG))
ICRF_TO_ECLIPTIC = ECLIPTIC_TO_ICRF.T


def solve_kepler(mean_anomaly_rad: float | np.ndarray, eccentricity: float) -> np.ndarray:
    """Solve Kepler's equation with a vectorized Newton iteration."""

    mean = np.mod(np.asarray(mean_anomaly_rad, dtype=float), 2.0 * np.pi)
    # NumPy ufuncs collapse 0-d array inputs to scalars, so a scalar mean
    # anomaly reaches this point as np.float64; the previous in-place item
    # assignment of the high-eccentricity starting guess therefore raised
    # TypeError for scalar inputs with eccentricity > 0.8.
    if eccentricity > 0.8:
        eccentric = np.full_like(np.asarray(mean), np.pi)
    else:
        eccentric = np.asarray(mean).copy()
    for _ in range(20):
        residual = eccentric - eccentricity * np.sin(eccentric) - mean
        delta = residual / (1.0 - eccentricity * np.cos(eccentric))
        eccentric -= delta
        if np.max(np.abs(delta)) < 2e-14:
            break
    return eccentric


def mean_motion_rad_s(semimajor_axis_km: float) -> float:
    return float(np.sqrt(MU_EARTH_KM3_S2 / semimajor_axis_km**3))


def elements_to_state(
    elements: OrbitalElements,
    gravitational_parameter_km3_s2: float = MU_EARTH_KM3_S2,
) -> np.ndarray:
    """Convert mean J2000 ecliptic elements to an ICRF relative state.

    The default gravitational parameter preserves the original Earth-centred
    convention.  Supplying another positive value makes the same conversion
    usable for Jacobi orbits such as a binary-moon mutual orbit.
    """

    a = elements.semimajor_axis_km
    e = elements.eccentricity
    if not np.isfinite(gravitational_parameter_km3_s2) or gravitational_parameter_km3_s2 <= 0.0:
        raise ValueError("gravitational_parameter_km3_s2 must be finite and positive")
    inc = np.radians(elements.inclination_deg)
    node = np.radians(elements.longitude_ascending_node_deg)
    argp = np.radians(elements.argument_periapsis_deg)
    mean = np.radians(elements.mean_anomaly_deg)
    ecc = float(solve_kepler(mean, e))
    radius = a * (1.0 - e * np.cos(ecc))
    position_pf = np.asarray((a * (np.cos(ecc) - e), a * np.sqrt(1.0 - e * e) * np.sin(ecc), 0.0))
    velocity_pf = (
        np.sqrt(gravitational_parameter_km3_s2 * a)
        / radius
        * np.asarray((-np.sin(ecc), np.sqrt(1.0 - e * e) * np.cos(ecc), 0.0))
    )

    c_o, s_o = np.cos(node), np.sin(node)
    c_i, s_i = np.cos(inc), np.sin(inc)
    c_w, s_w = np.cos(argp), np.sin(argp)
    rotation = np.asarray(
        (
            (c_o * c_w - s_o * s_w * c_i, -c_o * s_w - s_o * c_w * c_i, s_o * s_i),
            (s_o * c_w + c_o * s_w * c_i, -s_o * s_w + c_o * c_w * c_i, -c_o * s_i),
            (s_w * s_i, c_w * s_i, c_i),
        )
    )
    position = ECLIPTIC_TO_ICRF @ (rotation @ position_pf)
    velocity = ECLIPTIC_TO_ICRF @ (rotation @ velocity_pf)
    return np.concatenate((position, velocity))


def state_to_elements(state_icrf: np.ndarray, *, epoch_utc: str) -> OrbitalElements:
    """Convert an Earth-centred ICRF state to osculating ecliptic elements."""

    state = np.asarray(state_icrf, dtype=float)
    r = ICRF_TO_ECLIPTIC @ state[:3]
    v = ICRF_TO_ECLIPTIC @ state[3:]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    k = np.asarray((0.0, 0.0, 1.0))
    node_vector = np.cross(k, h)
    node_norm = np.linalg.norm(node_vector)
    eccentricity_vector = np.cross(v, h) / MU_EARTH_KM3_S2 - r / r_norm
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    energy = v_norm * v_norm / 2.0 - MU_EARTH_KM3_S2 / r_norm
    semimajor_axis = float(-MU_EARTH_KM3_S2 / (2.0 * energy))
    inclination = float(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0)))
    if node_norm > 1e-12:
        node = float(np.mod(np.arctan2(node_vector[1], node_vector[0]), 2.0 * np.pi))
    else:
        node = 0.0
    if eccentricity > 1e-12 and node_norm > 1e-12:
        argp = float(
            np.mod(
                np.arctan2(
                    np.dot(np.cross(node_vector, eccentricity_vector), h)
                    / (node_norm * eccentricity * h_norm),
                    np.dot(node_vector, eccentricity_vector) / (node_norm * eccentricity),
                ),
                2.0 * np.pi,
            )
        )
    elif eccentricity > 1e-12:
        argp = float(
            np.mod(np.arctan2(eccentricity_vector[1], eccentricity_vector[0]), 2.0 * np.pi)
        )
    else:
        argp = 0.0

    if eccentricity > 1e-12:
        cos_true = np.clip(np.dot(eccentricity_vector, r) / (eccentricity * r_norm), -1.0, 1.0)
        sin_true = np.dot(np.cross(eccentricity_vector, r), h) / (eccentricity * r_norm * h_norm)
        true_anomaly = float(np.mod(np.arctan2(sin_true, cos_true), 2.0 * np.pi))
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - eccentricity) * np.sin(true_anomaly / 2.0),
            np.sqrt(1.0 + eccentricity) * np.cos(true_anomaly / 2.0),
        )
        mean_anomaly = float(
            np.mod(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly), 2.0 * np.pi)
        )
    else:
        mean_anomaly = 0.0
    return OrbitalElements(
        semimajor_axis_km=semimajor_axis,
        eccentricity=eccentricity,
        inclination_deg=float(np.degrees(inclination)),
        longitude_ascending_node_deg=float(np.degrees(node)),
        argument_periapsis_deg=float(np.degrees(argp)),
        mean_anomaly_deg=float(np.degrees(mean_anomaly)),
        epoch_utc=epoch_utc,
    )


def propagate_two_body(elements: OrbitalElements, delta_seconds: float | np.ndarray) -> np.ndarray:
    """Fast analytic propagation used only by the coarse design stage."""

    delta = np.asarray(delta_seconds, dtype=float)
    mean_deg = elements.mean_anomaly_deg + np.degrees(
        mean_motion_rad_s(elements.semimajor_axis_km) * delta
    )
    if delta.ndim == 0:
        return elements_to_state(replace(elements, mean_anomaly_deg=float(mean_deg % 360.0)))
    return np.vstack(
        [
            elements_to_state(replace(elements, mean_anomaly_deg=float(value % 360.0)))
            for value in mean_deg
        ]
    )


class EphemerisInterpolator:
    """Cubic interpolation of prescribed Sun/Moon positions for ODE forcing."""

    def __init__(
        self,
        context: EphemerisContext,
        start_tt_jd: float,
        end_tt_jd: float,
        *,
        step_hours: float = 3.0,
    ) -> None:
        self.context = context
        self.start_tt_jd = float(min(start_tt_jd, end_tt_jd))
        self.end_tt_jd = float(max(start_tt_jd, end_tt_jd))
        count = max(5, int(np.ceil((self.end_tt_jd - self.start_tt_jd) * 24.0 / step_hours)) + 1)
        self.tt_jd = np.linspace(self.start_tt_jd, self.end_tt_jd, count)
        times = context.tt_jd(self.tt_jd)
        earth_position = context.earth.at(times).position.km
        self.sun_position = (context.sun.at(times).position.km - earth_position).T
        self.moon_position = (context.moon.at(times).position.km - earth_position).T
        seconds = (self.tt_jd - self.start_tt_jd) * SECONDS_PER_DAY
        self.sun_spline = CubicSpline(seconds, self.sun_position, axis=0)
        self.moon_spline = CubicSpline(seconds, self.moon_position, axis=0)

    def positions(self, tt_jd: float) -> tuple[np.ndarray, np.ndarray]:
        seconds = (float(tt_jd) - self.start_tt_jd) * SECONDS_PER_DAY
        return np.asarray(self.sun_spline(seconds)), np.asarray(self.moon_spline(seconds))


def _restricted_acceleration(
    t_seconds: float,
    state: np.ndarray,
    *,
    epoch_tt_jd: float,
    perturbers: EphemerisInterpolator,
    include_j2: bool,
) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    norm_r = np.linalg.norm(r)
    acceleration = -MU_EARTH_KM3_S2 * r / norm_r**3
    tt_jd = epoch_tt_jd + t_seconds / SECONDS_PER_DAY
    sun, moon = perturbers.positions(tt_jd)
    for mu, body_position in ((MU_SUN_KM3_S2, sun), (MU_MOON_KM3_S2, moon)):
        delta = body_position - r
        acceleration += mu * (
            delta / np.linalg.norm(delta) ** 3 - body_position / np.linalg.norm(body_position) ** 3
        )
    if include_j2:
        x, y, z = r
        r2 = norm_r * norm_r
        factor = 1.5 * EARTH_J2 * MU_EARTH_KM3_S2 * WGS84_A_KM**2 / norm_r**5
        z_ratio = 5.0 * z * z / r2
        acceleration += factor * np.asarray(
            (x * (z_ratio - 1.0), y * (z_ratio - 1.0), z * (z_ratio - 3.0))
        )
    return np.concatenate((v, acceleration))


def integrate_restricted(
    context: EphemerisContext,
    elements: OrbitalElements,
    end_tt_jd: float,
    *,
    start_tt_jd: float | None = None,
    perturbers: EphemerisInterpolator | None = None,
    include_j2: bool = True,
    rtol: float = 3e-11,
    max_step_seconds: float = 7_200.0,
) -> Trajectory:
    """Numerically integrate the moon in the prescribed DE440s force field."""

    epoch_time = context.time_utc(elements.epoch_utc)
    epoch_tt_jd = float(epoch_time.tt)
    if start_tt_jd is None:
        start_tt_jd = epoch_tt_jd
    if abs(start_tt_jd - epoch_tt_jd) > 1e-12:
        raise ValueError(
            "elements are defined at the epoch; non-epoch integration starts are unsupported"
        )
    end_tt_jd = float(end_tt_jd)
    if perturbers is None:
        perturbers = EphemerisInterpolator(context, epoch_tt_jd, end_tt_jd)
    end_seconds = (end_tt_jd - epoch_tt_jd) * SECONDS_PER_DAY
    solution = solve_ivp(
        lambda seconds, state: _restricted_acceleration(
            seconds,
            state,
            epoch_tt_jd=epoch_tt_jd,
            perturbers=perturbers,
            include_j2=include_j2,
        ),
        (0.0, end_seconds),
        elements_to_state(elements),
        method="DOP853",
        rtol=rtol,
        atol=np.asarray((1e-5, 1e-5, 1e-5, 1e-10, 1e-10, 1e-10)),
        max_step=max_step_seconds,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"restricted integration failed: {solution.message}")

    def evaluate(jd: float | np.ndarray) -> np.ndarray:
        values = np.asarray(jd, dtype=float)
        seconds = (values - epoch_tt_jd) * SECONDS_PER_DAY
        result = np.asarray(solution.sol(seconds))
        return result if values.ndim == 0 else result.T

    return Trajectory(
        epoch_jd=epoch_tt_jd,
        start_jd=min(epoch_tt_jd, end_tt_jd),
        end_jd=max(epoch_tt_jd, end_tt_jd),
        evaluator=evaluate,
        metadata={
            "model": "DE440s prescribed Sun/Moon + Earth monopole/J2",
            "integrator": "scipy DOP853",
            "rtol": rtol,
            "max_step_seconds": max_step_seconds,
            "ephemeris_interpolation_hours": float(np.diff(perturbers.tt_jd[:2])[0] * 24.0),
            "nfev": int(solution.nfev),
        },
        second_moon_radius_km=elements.radius_km,
        second_moon_mass_kg=elements.mass_kg,
    )


def target_position_for_observer(
    context: EphemerisContext,
    target_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
    geocentric_radius_km: float,
) -> np.ndarray:
    """Position a moon exactly on the observer's apparent Sun ray."""

    time = context.tt_jd(target_tt_jd)
    location = wgs84.latlon(latitude_deg, longitude_deg)
    observer_position = np.asarray(location.at(time).position.km, dtype=float)
    observer = context.earth + location
    sun_vector = np.asarray(
        observer.at(time).observe(context.sun).apparent().position.km, dtype=float
    )
    direction = sun_vector / np.linalg.norm(sun_vector)
    dot = float(np.dot(observer_position, direction))
    discriminant = (
        dot * dot + geocentric_radius_km**2 - float(np.dot(observer_position, observer_position))
    )
    if discriminant <= 0.0:
        raise ValueError("target Sun ray does not intersect requested geocentric radius")
    distance = -dot + np.sqrt(discriminant)
    return observer_position + distance * direction


def perigee_state_for_position(
    position_icrf_km: np.ndarray,
    semimajor_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
) -> np.ndarray:
    """Construct a prograde perigee state through a requested position vector."""

    position_ecl = ICRF_TO_ECLIPTIC @ np.asarray(position_icrf_km, dtype=float)
    radius = np.linalg.norm(position_ecl)
    expected = semimajor_axis_km * (1.0 - eccentricity)
    if abs(radius - expected) > 1e-3:
        position_ecl *= expected / radius
        radius = expected
    longitude = np.arctan2(position_ecl[1], position_ecl[0])
    latitude = np.arcsin(position_ecl[2] / radius)
    inclination = np.radians(inclination_deg)
    ratio = -np.tan(latitude) / np.tan(inclination)
    if abs(ratio) > 1.0:
        raise ValueError(
            f"target ecliptic latitude {np.degrees(latitude):.3f} exceeds orbit inclination"
        )
    candidates = (
        longitude + np.arcsin(ratio),
        longitude + (np.pi - np.arcsin(ratio)),
    )
    best_state = None
    for node in candidates:
        normal = np.asarray(
            (
                np.sin(inclination) * np.sin(node),
                -np.sin(inclination) * np.cos(node),
                np.cos(inclination),
            )
        )
        r_hat = position_ecl / radius
        velocity_hat = np.cross(normal, r_hat)
        velocity_hat /= np.linalg.norm(velocity_hat)
        speed = np.sqrt(
            MU_EARTH_KM3_S2 * (1.0 + eccentricity) / (semimajor_axis_km * (1.0 - eccentricity))
        )
        state_ecl = np.concatenate((position_ecl, speed * velocity_hat))
        state_icrf = np.concatenate(
            (ECLIPTIC_TO_ICRF @ state_ecl[:3], ECLIPTIC_TO_ICRF @ state_ecl[3:])
        )
        if best_state is None:
            best_state = state_icrf
    assert best_state is not None
    return best_state


def analytic_design_seed(
    context: EphemerisContext,
    baseline: OrbitalElements,
    target_tt_jd: float,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[OrbitalElements, np.ndarray]:
    """Construct an exact two-body seed for an observer-aligned perigee."""

    radius = baseline.semimajor_axis_km * (1.0 - baseline.eccentricity)
    target_position = target_position_for_observer(
        context, target_tt_jd, latitude_deg, longitude_deg, radius
    )
    target_state = perigee_state_for_position(
        target_position,
        baseline.semimajor_axis_km,
        baseline.eccentricity,
        baseline.inclination_deg,
    )
    target_elements = state_to_elements(target_state, epoch_utc=baseline.epoch_utc)
    epoch_tt_jd = float(context.time_utc(baseline.epoch_utc).tt)
    delta_seconds = (target_tt_jd - epoch_tt_jd) * SECONDS_PER_DAY
    mean_at_epoch = target_elements.mean_anomaly_deg - np.degrees(
        mean_motion_rad_s(baseline.semimajor_axis_km) * delta_seconds
    )
    seed = replace(
        baseline,
        longitude_ascending_node_deg=target_elements.longitude_ascending_node_deg,
        argument_periapsis_deg=target_elements.argument_periapsis_deg,
        mean_anomaly_deg=float(mean_at_epoch % 360.0),
    )
    return seed, target_position


def coarse_global_design(
    baseline: OrbitalElements,
    target_position_icrf_km: np.ndarray,
    delta_seconds: float,
    *,
    seed: int = 20260710,
    maxiter: int = 70,
    popsize: int = 12,
) -> tuple[OrbitalElements, dict[str, float]]:
    """Differential-evolution search over Ω, ω, and M0 in a two-body model."""

    target = np.asarray(target_position_icrf_km, dtype=float)

    def objective(angles_deg: np.ndarray) -> float:
        candidate = replace(
            baseline,
            longitude_ascending_node_deg=float(angles_deg[0]),
            argument_periapsis_deg=float(angles_deg[1]),
            mean_anomaly_deg=float(angles_deg[2]),
        )
        position = propagate_two_body(candidate, delta_seconds)[:3]
        radial_error = np.linalg.norm(position - target)
        angular_error = angular_between(position, target) * baseline.semimajor_axis_km
        return float(radial_error + angular_error)

    result = differential_evolution(
        objective,
        bounds=((0.0, 360.0), (0.0, 360.0), (0.0, 360.0)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-8,
        polish=True,
        updating="immediate",
    )
    candidate = replace(
        baseline,
        longitude_ascending_node_deg=float(result.x[0] % 360.0),
        argument_periapsis_deg=float(result.x[1] % 360.0),
        mean_anomaly_deg=float(result.x[2] % 360.0),
    )
    return candidate, {
        "objective_km": float(result.fun),
        "evaluations": float(result.nfev),
        "success": bool(result.success),
    }


def angular_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = np.asarray(a, dtype=float) / np.linalg.norm(a)
    ub = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.arctan2(np.linalg.norm(np.cross(ua, ub)), np.dot(ua, ub)))


def refine_design_shooting(
    context: EphemerisContext,
    seed_elements: OrbitalElements,
    target_tt_jd: float,
    target_position_icrf_km: np.ndarray,
    *,
    include_j2: bool = True,
) -> tuple[OrbitalElements, Trajectory, dict[str, float]]:
    """Refine the epoch angles by numerically shooting to the desired position."""

    epoch_tt_jd = float(context.time_utc(seed_elements.epoch_utc).tt)
    perturbers = EphemerisInterpolator(context, epoch_tt_jd, target_tt_jd, step_hours=1.0)
    x0 = np.asarray(
        (
            seed_elements.longitude_ascending_node_deg,
            seed_elements.argument_periapsis_deg,
            seed_elements.mean_anomaly_deg,
        )
    )
    target = np.asarray(target_position_icrf_km, dtype=float)
    calls = 0

    def residual(angles_deg: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        candidate = replace(
            seed_elements,
            longitude_ascending_node_deg=float(angles_deg[0] % 360.0),
            argument_periapsis_deg=float(angles_deg[1] % 360.0),
            mean_anomaly_deg=float(angles_deg[2] % 360.0),
        )
        trajectory = integrate_restricted(
            context,
            candidate,
            target_tt_jd,
            perturbers=perturbers,
            include_j2=include_j2,
            rtol=1e-11,
            max_step_seconds=3_600.0,
        )
        return (trajectory.position(target_tt_jd) - target) / 100.0

    fit = least_squares(
        residual,
        x0,
        method="trf",
        x_scale=np.asarray((5.0, 5.0, 5.0)),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=40,
    )
    refined = replace(
        seed_elements,
        longitude_ascending_node_deg=float(fit.x[0] % 360.0),
        argument_periapsis_deg=float(fit.x[1] % 360.0),
        mean_anomaly_deg=float(fit.x[2] % 360.0),
    )
    trajectory = integrate_restricted(
        context,
        refined,
        target_tt_jd,
        perturbers=perturbers,
        include_j2=include_j2,
        rtol=3e-12,
        max_step_seconds=1_800.0,
    )
    error = np.linalg.norm(trajectory.position(target_tt_jd) - target)
    return (
        refined,
        trajectory,
        {
            "success": bool(fit.success),
            "shooting_calls": float(calls),
            "target_position_error_km": float(error),
            "cost": float(fit.cost),
            "optimality": float(fit.optimality),
        },
    )
