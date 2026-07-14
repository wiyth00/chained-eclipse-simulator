"""Static ground-track map for a standalone second-moon eclipse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
from skyfield.framelib import itrs
import yaml

from .animation_2d import SurfaceGrid
from .constants import SECOND_MOON_RADIUS_KM, SECONDS_PER_DAY, SPEED_OF_LIGHT_KM_S
from .eclipse_geometry import generate_central_track
from .ephemeris import load_ephemeris, time_iso_utc
from .models import OrbitalElements
from .orbital_dynamics import integrate_restricted


PACIFIC_EXTENT = (-180.0, -60.0, -5.0, 40.0)


def _instantaneous_fields(context, trajectory, grid: SurfaceGrid, tt_jd: float):
    time = context.tt_jd(tt_jd)
    rotation = np.asarray(itrs.rotation_at(time), dtype=float)
    sun_icrf = np.asarray(
        context.earth.at(time).observe(context.sun).apparent().position.km,
        dtype=float,
    )
    moon_now = np.asarray(trajectory.position(tt_jd), dtype=float)
    light_time_days = np.linalg.norm(moon_now) / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
    moon_icrf = np.asarray(trajectory.position(tt_jd - light_time_days), dtype=float)
    return grid.obscuration(
        rotation @ sun_icrf,
        rotation @ moon_icrf,
        SECOND_MOON_RADIUS_KM,
    )


def build_ground_track_map(
    *,
    event_json: str | Path = "outputs/next_second_moon_eclipses.json",
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    output_dir: str | Path = "outputs/figures/20261225_second_moon_annular",
    grid_step_deg: float = 0.25,
    time_step_seconds: float = 60.0,
) -> dict[str, str]:
    payload = json.loads(Path(event_json).read_text(encoding="utf-8"))
    event = payload["first_central_eclipse"]
    if event is None:
        raise ValueError("standalone search contains no central eclipse")
    context = load_ephemeris(ephemeris_cache)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    end = context.time_utc(event["global_end_utc"])
    trajectory = integrate_restricted(
        context,
        elements,
        float(end.tt) + 0.02,
        rtol=3e-11,
        max_step_seconds=7_200.0,
    )
    maximum_tt = float(context.time_utc(event["maximum_utc"]).tt)
    start_tt = float(context.time_utc(event["global_start_utc"]).tt)
    end_tt = float(end.tt)
    track = generate_central_track(
        context,
        maximum_tt,
        "second_moon",
        position_provider=trajectory.position,
        half_window_hours=3.0,
        step_seconds=20.0,
    )

    grid = SurfaceGrid(step_deg=grid_step_deg, extent=PACIFIC_EXTENT)
    maximum_obscuration = np.zeros(grid.shape, dtype=float)
    ever_central = np.zeros(grid.shape, dtype=bool)
    sample_times = np.arange(
        start_tt,
        end_tt + time_step_seconds / SECONDS_PER_DAY,
        time_step_seconds / SECONDS_PER_DAY,
    )
    for tt_jd in sample_times:
        obscuration, central = _instantaneous_fields(context, trajectory, grid, float(tt_jd))
        maximum_obscuration = np.maximum(maximum_obscuration, obscuration)
        ever_central |= central

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    projection = ccrs.PlateCarree()
    figure = plt.figure(figsize=(14.0, 7.2), facecolor="white")
    axis = figure.add_axes((0.055, 0.14, 0.90, 0.76), projection=projection)
    axis.set_extent(PACIFIC_EXTENT, crs=projection)
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#D9EAF2", zorder=0)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#EEE9DC", zorder=1)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#D9EAF2", zorder=2)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#6A6A62",
        linewidth=0.45,
        zorder=8,
    )
    axis.coastlines(resolution="50m", color="#27313A", linewidth=0.65, zorder=9)
    gridlines = axis.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.45,
        color="#7B8794",
        alpha=0.55,
        linestyle=":",
        zorder=3,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlocator = FixedLocator((-160, -140, -120, -100, -80))
    gridlines.ylocator = FixedLocator((0, 10, 20, 30, 40))

    lon, lat = np.meshgrid(grid.longitude_deg, grid.latitude_deg)
    levels = (0.001, 0.10, 0.30, 0.50, 0.70, 0.90, 1.001)
    axis.contourf(
        lon,
        lat,
        maximum_obscuration,
        levels=levels,
        colors=("#F6CFAF", "#F2B47E", "#EB9554", "#D96F2A", "#B84A16", "#7D250D"),
        alpha=0.50,
        transform=projection,
        zorder=4,
    )
    axis.contour(
        lon,
        lat,
        maximum_obscuration,
        levels=(0.001,),
        colors=("#A64616",),
        linewidths=0.8,
        transform=projection,
        zorder=5,
    )
    axis.contourf(
        lon,
        lat,
        ever_central.astype(float),
        levels=(0.5, 1.5),
        colors=("#FF7A1A",),
        alpha=0.78,
        transform=projection,
        zorder=6,
    )
    axis.contour(
        lon,
        lat,
        ever_central.astype(float),
        levels=(0.5,),
        colors=("white",),
        linewidths=0.7,
        transform=projection,
        zorder=7,
    )
    track_longitude = np.asarray(track["longitude_deg"], dtype=float)
    # Keep the dateline crossing continuous in the -180..-60 map domain.
    track_longitude = np.where(track_longitude > 0.0, track_longitude - 360.0, track_longitude)
    axis.plot(
        track_longitude,
        track["latitude_deg"],
        color="#541900",
        linewidth=1.5,
        transform=projection,
        zorder=10,
        label="Annular centerline",
    )
    axis.scatter(
        [event["longitude_deg"]],
        [event["latitude_deg"]],
        marker="*",
        s=135,
        facecolor="#FFF4C2",
        edgecolor="#541900",
        linewidth=1.0,
        transform=projection,
        zorder=11,
        label="Greatest eclipse",
    )
    axis.text(
        event["longitude_deg"] + 2.0,
        event["latitude_deg"] + 1.2,
        "Greatest eclipse\n20:24:55 UTC",
        fontsize=8,
        color="#541900",
        transform=projection,
        zorder=12,
    )
    legend = axis.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=9)
    legend.get_frame().set_edgecolor("#8C8C82")
    figure.suptitle(
        "Second moon's next central solar eclipse · 25 December 2026",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.075,
        "Annular · greatest eclipse 20:24:55 UTC at 5.678° N, 124.125° W · "
        "maximum obscuration 94.65% · annularity 50.4 s",
        ha="center",
        fontsize=10,
        color="#30343A",
    )
    figure.text(
        0.5,
        0.035,
        "Orange shading: maximum obscuration during the event · bright band: annularity path "
        f"({grid_step_deg:.2f}° WGS84 grid, {time_step_seconds:.0f} s sampling)",
        ha="center",
        fontsize=8.5,
        color="#646B73",
    )
    map_path = output / "ground_track.png"
    figure.savefig(map_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    track_table = pd.DataFrame(
        {
            "time_utc": [
                time_iso_utc(context.tt_jd(float(value)), places=3)
                for value in track["tt_jd"]
            ],
            "latitude_deg": track["latitude_deg"],
            "longitude_deg": track["longitude_deg"],
            "signed_antumbra_radius_km": track["signed_core_radius_km"],
        }
    )
    csv_path = output / "central_track.csv"
    track_table.to_csv(csv_path, index=False)
    details_path = output / "event_details.json"
    details_path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    return {
        "map": str(map_path.resolve()),
        "track_csv": str(csv_path.resolve()),
        "event_json": str(details_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-json", default="outputs/next_second_moon_eclipses.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output-dir", default="outputs/figures/20261225_second_moon_annular"
    )
    parser.add_argument("--grid-step-deg", type=float, default=0.25)
    parser.add_argument("--time-step-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    result = build_ground_track_map(
        event_json=args.event_json,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_dir=args.output_dir,
        grid_step_deg=args.grid_step_deg,
        time_step_seconds=args.time_step_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
