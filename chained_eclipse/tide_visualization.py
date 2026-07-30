"""Animate the exact equilibrium tide-generating potential raised by both moons.

This is a tide-generating-potential visualization, not a hydrodynamic ocean
forecast.  It intentionally excludes bathymetry, continents, resonances,
friction, loading, and the solar tide so the interference of the two lunar
bulges can be seen directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

from .constants import (
    REAL_MOON_MASS_KG,
    SECONDS_PER_DAY,
    WGS84_A_KM,
    WGS84_B_KM,
)
from .enhanced_ephemeris import EnhancedEphemeris
from .ephemeris import load_ephemeris, time_iso_utc
from .models import OrbitalElements
from .tides_spin import G_KM3_KG_S2


STANDARD_GRAVITY_KM_S2 = 9.80665e-3
EARTH_MEAN_RADIUS_KM = (2.0 * WGS84_A_KM + WGS84_B_KM) / 3.0
REAL_BLUE = "#32A9E8"
SECOND_ORANGE = "#F28E2B"


@dataclass(frozen=True, slots=True)
class TideBodyState:
    """Earth-fixed tide-raiser state and its subpoint equilibrium amplitude."""

    name: str
    mass_kg: float
    distance_km: float
    subpoint_latitude_deg: float
    subpoint_longitude_deg: float
    subpoint_amplitude_m: float
    unit_itrs: np.ndarray


def surface_unit_vectors(
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return regular longitude/latitude grids and geocentric unit vectors."""

    if not math.isfinite(step_deg) or step_deg <= 0.0 or step_deg > 10.0:
        raise ValueError("step_deg must be finite and in (0, 10]")
    longitude = np.arange(-180.0, 180.0 + 0.5 * step_deg, step_deg)
    latitude = np.arange(-90.0, 90.0 + 0.5 * step_deg, step_deg)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    longitude_rad = np.radians(lon_grid)
    latitude_rad = np.radians(lat_grid)
    cosine_latitude = np.cos(latitude_rad)
    unit = np.stack(
        (
            cosine_latitude * np.cos(longitude_rad),
            cosine_latitude * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ),
        axis=-1,
    )
    return lon_grid, lat_grid, unit


def body_tide_state(
    ephemeris: EnhancedEphemeris,
    body: str,
    mass_kg: float,
    tt_jd: float,
) -> TideBodyState:
    """Evaluate a moon's direction, range, and equilibrium-tide amplitude."""

    relative_icrf = np.asarray(ephemeris.relative(body, tt_jd), dtype=float)
    distance_km = float(np.linalg.norm(relative_icrf))
    if not math.isfinite(distance_km) or distance_km <= 0.0:
        raise ValueError("tide raiser must have a finite positive geocentric range")
    unit_itrs = ephemeris.rotation_matrix_icrf_to_itrs(tt_jd) @ (
        relative_icrf / distance_km
    )
    unit_itrs /= np.linalg.norm(unit_itrs)
    longitude = math.degrees(math.atan2(unit_itrs[1], unit_itrs[0]))
    latitude = math.degrees(math.asin(float(np.clip(unit_itrs[2], -1.0, 1.0))))
    amplitude_m = (
        G_KM3_KG_S2
        * mass_kg
        * EARTH_MEAN_RADIUS_KM**2
        / (STANDARD_GRAVITY_KM_S2 * distance_km**3)
        * 1_000.0
    )
    return TideBodyState(
        name=body,
        mass_kg=mass_kg,
        distance_km=distance_km,
        subpoint_latitude_deg=latitude,
        subpoint_longitude_deg=longitude,
        subpoint_amplitude_m=amplitude_m,
        unit_itrs=unit_itrs,
    )


def equilibrium_tide_height_m(
    surface_units: np.ndarray,
    states: tuple[TideBodyState, ...],
) -> np.ndarray:
    """Return the summed degree-2 equilibrium height from supplied bodies.

    For each body, ``zeta = (G M R^2 / g D^3) P2(cos psi)``.  The degree-zero
    and degree-one terms are absent, so the field is the actual tide-generating
    part of the external potential rather than raw gravitational potential.
    """

    field = np.zeros(surface_units.shape[:-1], dtype=float)
    for state in states:
        cosine_zenith = np.einsum("...i,i->...", surface_units, state.unit_itrs)
        legendre_2 = 0.5 * (3.0 * cosine_zenith**2 - 1.0)
        field += state.subpoint_amplitude_m * legendre_2
    return field


def exact_equilibrium_tide_height_m(
    surface_positions_km: np.ndarray,
    area_weights: np.ndarray,
    states: tuple[TideBodyState, ...],
) -> np.ndarray:
    """Return the exact equilibrium tide after removing degrees zero and one.

    Unlike the usual degree-2 approximation, this retains the small near/far
    asymmetry produced by the unusually close second moon.  A weighted mean is
    removed after summation so the plotted anomaly has zero global volume on
    the discretized ellipsoid.
    """

    field_km = np.zeros(surface_positions_km.shape[:-1], dtype=float)
    for state in states:
        raiser = state.unit_itrs * state.distance_km
        separation = np.linalg.norm(raiser - surface_positions_km, axis=-1)
        direct = 1.0 / separation
        constant = 1.0 / state.distance_km
        uniform_acceleration = (
            np.einsum("...i,i->...", surface_positions_km, raiser)
            / state.distance_km**3
        )
        field_km += (
            G_KM3_KG_S2
            * state.mass_kg
            / STANDARD_GRAVITY_KM_S2
            * (direct - constant - uniform_acceleration)
        )
    field_m = field_km * 1_000.0
    field_m -= np.average(field_m, weights=area_weights)
    return field_m


def wgs84_surface_positions(
    longitude_deg: np.ndarray, latitude_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 surface positions and approximate cell-area weights."""

    longitude = np.radians(longitude_deg)
    latitude = np.radians(latitude_deg)
    eccentricity_squared = 1.0 - (WGS84_B_KM / WGS84_A_KM) ** 2
    prime_vertical = WGS84_A_KM / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude) ** 2
    )
    positions = np.stack(
        (
            prime_vertical * np.cos(latitude) * np.cos(longitude),
            prime_vertical * np.cos(latitude) * np.sin(longitude),
            prime_vertical
            * (1.0 - eccentricity_squared)
            * np.sin(latitude),
        ),
        axis=-1,
    )
    # Spherical cos(latitude) weights are ample at this display resolution;
    # keep tiny nonzero polar weights so np.average remains well conditioned.
    weights = np.maximum(np.cos(latitude), 1.0e-12)
    return positions, weights


def _alignment_deg(real: TideBodyState, second: TideBodyState) -> float:
    # A degree-2 bulge is antipodally symmetric, so aligned and anti-aligned
    # moons are equally constructive.
    dot = float(np.clip(abs(np.dot(real.unit_itrs, second.unit_itrs)), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def build_tide_frames(
    ephemeris: EnhancedEphemeris,
    times_tt_jd: np.ndarray,
    surface_positions_km: np.ndarray,
    area_weights: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Evaluate all map frames and compact per-frame diagnostics."""

    frames = np.empty(
        (len(times_tt_jd), *surface_positions_km.shape[:-1]), dtype=np.float32
    )
    diagnostics: list[dict[str, float | str]] = []
    for index, tt_jd in enumerate(times_tt_jd):
        real = body_tide_state(
            ephemeris, "real_moon", REAL_MOON_MASS_KG, float(tt_jd)
        )
        second = body_tide_state(
            ephemeris,
            "second_moon",
            ephemeris.elements.mass_kg,
            float(tt_jd),
        )
        field = exact_equilibrium_tide_height_m(
            surface_positions_km, area_weights, (real, second)
        )
        frames[index] = field
        diagnostics.append(
            {
                "utc": str(ephemeris.time(float(tt_jd)).utc_iso(places=0)),
                "real_moon_distance_km": real.distance_km,
                "real_moon_subpoint_latitude_deg": real.subpoint_latitude_deg,
                "real_moon_subpoint_longitude_deg": real.subpoint_longitude_deg,
                "real_moon_equilibrium_amplitude_m": real.subpoint_amplitude_m,
                "second_moon_distance_km": second.distance_km,
                "second_moon_subpoint_latitude_deg": second.subpoint_latitude_deg,
                "second_moon_subpoint_longitude_deg": second.subpoint_longitude_deg,
                "second_moon_equilibrium_amplitude_m": second.subpoint_amplitude_m,
                "bulge_axis_separation_deg": _alignment_deg(real, second),
                "combined_maximum_m": float(np.max(field)),
                "combined_minimum_m": float(np.min(field)),
                "combined_peak_to_trough_m": float(np.ptp(field)),
            }
        )
    return frames, diagnostics


def render_tide_animation(
    frames: np.ndarray,
    diagnostics: list[dict[str, float | str]],
    output_path: str | Path,
    *,
    fps: int = 24,
    dpi: int = 140,
) -> Path:
    """Render an equirectangular H.264 movie or GIF from computed tide fields."""

    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    if len(frames) != len(diagnostics) or len(frames) < 2:
        raise ValueError("frames and diagnostics must have matching length >= 2")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    projection = ccrs.PlateCarree()
    # Even-inch dimensions guarantee an even H.264 frame size for integer DPI.
    figure = plt.figure(figsize=(14.0, 8.0), facecolor="#07111D")
    axis = figure.add_axes((0.035, 0.12, 0.93, 0.76), projection=projection)
    axis.set_global()
    axis.set_facecolor("#07111D")
    limit = max(abs(float(np.min(frames))), abs(float(np.max(frames))))
    image = axis.imshow(
        frames[0],
        extent=(-180.0, 180.0, -90.0, 90.0),
        origin="lower",
        transform=projection,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="bilinear",
        zorder=0,
    )
    axis.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=(0.12, 0.16, 0.20, 0.30),
        edgecolor="none",
        zorder=1,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="#D7E0EA",
        linewidth=0.45,
        zorder=3,
    )
    gridlines = axis.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="#B7C4D3",
        alpha=0.35,
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"color": "#CFD8E3", "size": 8}
    gridlines.ylabel_style = {"color": "#CFD8E3", "size": 8}
    real_marker, = axis.plot(
        [], [], marker="o", markersize=7, color=REAL_BLUE, markeredgecolor="white",
        markeredgewidth=0.8, linestyle="none", transform=projection, zorder=5,
    )
    second_marker, = axis.plot(
        [], [], marker="D", markersize=6, color=SECOND_ORANGE, markeredgecolor="white",
        markeredgewidth=0.8, linestyle="none", transform=projection, zorder=5,
    )
    title = figure.text(
        0.5, 0.945, "", ha="center", va="center", color="white", fontsize=15,
        fontweight="bold",
    )
    status = figure.text(
        0.5, 0.055, "", ha="center", va="center", color="#D7E0EA", fontsize=10,
    )
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.075, fraction=0.05)
    colorbar.set_label("Two-moon equilibrium tide height relative to mean sea level (m)", color="white")
    colorbar.ax.tick_params(colors="white", labelsize=8)
    colorbar.outline.set_edgecolor("#8493A5")

    def update(index: int):
        row = diagnostics[index]
        image.set_data(frames[index])
        real_marker.set_data(
            [float(row["real_moon_subpoint_longitude_deg"])],
            [float(row["real_moon_subpoint_latitude_deg"])],
        )
        second_marker.set_data(
            [float(row["second_moon_subpoint_longitude_deg"])],
            [float(row["second_moon_subpoint_latitude_deg"])],
        )
        title.set_text(f"Two-moon equilibrium tide · {row['utc']}")
        status.set_text(
            f"● real Moon {float(row['real_moon_equilibrium_amplitude_m']):.3f} m   ·   "
            f"◆ second moon {float(row['second_moon_equilibrium_amplitude_m']):.3f} m   ·   "
            f"bulge-axis separation {float(row['bulge_axis_separation_deg']):.1f}°   ·   "
            f"combined high {float(row['combined_maximum_m']):.3f} m"
        )
        return image, real_marker, second_marker, title, status

    movie = mpl_animation.FuncAnimation(
        figure, update, frames=len(frames), interval=1_000.0 / fps, blit=False
    )
    if output.suffix.lower() == ".mp4":
        writer = mpl_animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=5_000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    elif output.suffix.lower() == ".gif":
        writer = mpl_animation.PillowWriter(fps=fps)
    else:
        raise ValueError("animation output must end in .mp4 or .gif")
    movie.save(output, writer=writer, dpi=dpi)
    plt.close(figure)
    return output


def run_tide_visualization(
    *,
    output_path: str | Path = (
        "outputs/coupled/tides_30d/two_moon_equilibrium_tides.mp4"
    ),
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    days: float = 30.0,
    time_step_minutes: float = 60.0,
    grid_step_deg: float = 1.0,
    trajectory_step_seconds: float = 3_600.0,
    fps: int = 24,
    dpi: int = 140,
) -> dict[str, object]:
    """Compute, render, and save the complete tide-animation data product."""

    if days <= 0.0 or time_step_minutes <= 0.0:
        raise ValueError("days and time_step_minutes must be positive")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    context = load_ephemeris(ephemeris_cache)
    start = context.time_utc(elements.epoch_utc)
    end_tt = float(start.tt) + days
    end_utc = time_iso_utc(context.tt_jd(end_tt), places=0)
    ephemeris = EnhancedEphemeris(
        context,
        elements,
        end_utc,
        sample_step_seconds=trajectory_step_seconds,
    )
    step_days = time_step_minutes * 60.0 / SECONDS_PER_DAY
    times = np.arange(float(start.tt), end_tt, step_days, dtype=float)
    if not math.isclose(float(times[-1]), end_tt, abs_tol=1.0e-10):
        times = np.append(times, end_tt)
    longitude, latitude, _ = surface_unit_vectors(grid_step_deg)
    surface_positions, area_weights = wgs84_surface_positions(longitude, latitude)
    frames, diagnostics = build_tide_frames(
        ephemeris, times, surface_positions, area_weights
    )
    output = render_tide_animation(frames, diagnostics, output_path, fps=fps, dpi=dpi)
    table_path = output.with_suffix(".csv")
    pd.DataFrame(diagnostics).to_csv(table_path, index=False)
    maximum_index = int(np.argmax([float(row["combined_maximum_m"]) for row in diagnostics]))
    minimum_alignment_index = int(
        np.argmin([float(row["bulge_axis_separation_deg"]) for row in diagnostics])
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "output": str(output.resolve()),
        "data_csv": str(table_path.resolve()),
        "start_utc": diagnostics[0]["utc"],
        "end_utc": diagnostics[-1]["utc"],
        "days": days,
        "frames": len(frames),
        "fps": fps,
        "screen_duration_s": len(frames) / fps,
        "physical_time_step_minutes": time_step_minutes,
        "grid_step_deg": grid_step_deg,
        "projection": "Plate Carree (equirectangular)",
        "trajectory": ephemeris.metadata,
        "model": {
            "quantity": "exact equilibrium tide height from both moons",
            "equation": (
                "zeta=sum[(G*M/g)*(1/|r-x|-1/|r|-(r dot x)/|r|^3)], "
                "with the discrete area-weighted mean removed"
            ),
            "diagnostic_amplitudes": (
                "displayed per-moon amplitudes are the corresponding degree-2 "
                "subpoint coefficients G*M*R^2/(g*D^3)"
            ),
            "reference_radius_km": EARTH_MEAN_RADIUS_KM,
            "gravity_m_s2": STANDARD_GRAVITY_KM_S2 * 1_000.0,
            "included": ["real Moon", "hypothetical second moon"],
            "excluded": [
                "solar tide",
                "ocean dynamics and inertia",
                "bathymetry and coast geometry",
                "resonance, friction, loading, self-attraction, and deformation",
            ],
            "interpretation": (
                "An instantaneous equilibrium open-ocean proxy showing the forcing pattern; "
                "it is not a prediction of coastal water level or flooding."
            ),
        },
        "strongest_sample": diagnostics[maximum_index],
        "closest_bulge_alignment_sample": diagnostics[minimum_alignment_index],
        "field_range_m": [float(np.min(frames)), float(np.max(frames))],
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/coupled/tides_30d/two_moon_equilibrium_tides.mp4",
    )
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--time-step-minutes", type=float, default=60.0)
    parser.add_argument("--grid-step-deg", type=float, default=1.0)
    parser.add_argument("--trajectory-step-seconds", type=float, default=3_600.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    manifest = run_tide_visualization(
        output_path=args.output,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        days=args.days,
        time_step_minutes=args.time_step_minutes,
        grid_step_deg=args.grid_step_deg,
        trajectory_step_seconds=args.trajectory_step_seconds,
        fps=args.fps,
        dpi=args.dpi,
    )
    print(f"Two-moon tide animation: {manifest['output']}")
    print(
        "Strongest sampled equilibrium high: "
        f"{float(manifest['strongest_sample']['combined_maximum_m']):.3f} m at "
        f"{manifest['strongest_sample']['utc']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
