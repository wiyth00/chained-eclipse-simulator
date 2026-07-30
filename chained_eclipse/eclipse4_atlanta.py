"""Zoomed U.S. ground track for the fourth 2027 second-moon eclipse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod
import yaml

from .animation_2d import SurfaceGrid
from .constants import SECONDS_PER_DAY
from .eclipse_geometry import generate_central_track, solve_local_circumstances
from .ephemeris import load_ephemeris, time_iso_utc
from .models import OrbitalElements
from .orbital_dynamics import integrate_restricted
from .standalone_map import _instantaneous_fields


ATLANTA = (33.7490, -84.3880)
US_EXTENT = (-130.0, -65.0, 20.0, 55.0)


def build_atlanta_zoom(
    *,
    events_path: str | Path = "outputs/figures/2027_second_moon_total_tracks/events.json",
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    output_dir: str | Path = "outputs/figures/20270623_eclipse4_atlanta",
    grid_step_deg: float = 0.10,
    time_step_seconds: float = 30.0,
) -> dict[str, str]:
    payload = json.loads(Path(events_path).read_text(encoding="utf-8"))
    event = payload["events"][3]
    context = load_ephemeris(ephemeris_cache)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    end_tt = float(context.time_utc(event["global_end_utc"]).tt) + 0.02
    trajectory = integrate_restricted(
        context,
        elements,
        end_tt,
        rtol=3e-11,
        max_step_seconds=7_200.0,
    )
    maximum_tt = float(context.time_utc(event["maximum_utc"]).tt)
    track = generate_central_track(
        context,
        maximum_tt,
        "second_moon",
        position_provider=trajectory.position,
        body_radius_km=elements.radius_km,
        half_window_hours=3.0,
        step_seconds=2.0,
    )
    atlanta = solve_local_circumstances(
        context,
        maximum_tt,
        ATLANTA[0],
        ATLANTA[1],
        "second_moon",
        position_provider=trajectory.position,
        body_radius_km=elements.radius_km,
        search_half_window_hours=4.0,
        bracket_step_seconds=10.0,
    )

    geod = Geod(ellps="WGS84")
    _, _, distances_m = geod.inv(
        np.full_like(track["longitude_deg"], ATLANTA[1]),
        np.full_like(track["latitude_deg"], ATLANTA[0]),
        track["longitude_deg"],
        track["latitude_deg"],
    )
    closest_index = int(np.argmin(distances_m))
    closest = {
        "distance_km": float(distances_m[closest_index] / 1_000.0),
        "time_utc": time_iso_utc(
            context.tt_jd(float(track["tt_jd"][closest_index])), places=3
        ),
        "latitude_deg": float(track["latitude_deg"][closest_index]),
        "longitude_deg": float(track["longitude_deg"][closest_index]),
        "umbra_radius_km": float(track["signed_core_radius_km"][closest_index]),
    }

    grid = SurfaceGrid(step_deg=grid_step_deg, extent=US_EXTENT)
    maximum_obscuration = np.zeros(grid.shape, dtype=float)
    ever_total = np.zeros(grid.shape, dtype=bool)
    start_tt = float(context.time_utc(event["global_start_utc"]).tt)
    event_end_tt = float(context.time_utc(event["global_end_utc"]).tt)
    sample_times = np.arange(
        start_tt,
        event_end_tt + time_step_seconds / SECONDS_PER_DAY,
        time_step_seconds / SECONDS_PER_DAY,
    )
    for tt_jd in sample_times:
        obscuration, central = _instantaneous_fields(
            context, trajectory, grid, float(tt_jd)
        )
        maximum_obscuration = np.maximum(maximum_obscuration, obscuration)
        ever_total |= central

    projection = ccrs.PlateCarree()
    figure = plt.figure(figsize=(14.0, 8.0), facecolor="white")
    axis = figure.add_axes((0.055, 0.14, 0.90, 0.76), projection=projection)
    axis.set_extent(US_EXTENT, crs=projection)
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCECF3", zorder=0)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#EEE9DC", zorder=1)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#DCECF3", zorder=2)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#4E555B",
        linewidth=0.55,
        zorder=8,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("50m"),
        edgecolor="#77746B",
        facecolor="none",
        linewidth=0.4,
        zorder=8,
    )
    axis.coastlines(resolution="50m", color="#26343E", linewidth=0.65, zorder=9)
    gridlines = axis.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="#7D8993",
        alpha=0.45,
        linestyle=":",
        zorder=3,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    lon, lat = np.meshgrid(grid.longitude_deg, grid.latitude_deg)
    axis.contourf(
        lon,
        lat,
        maximum_obscuration,
        levels=(0.50, 0.80, 0.90, 0.95, 0.99, 1.001),
        colors=("#F9DEC2", "#F7C18D", "#EF9955", "#D96B2A", "#A83B18"),
        alpha=0.48,
        transform=projection,
        zorder=4,
    )
    axis.contourf(
        lon,
        lat,
        ever_total.astype(float),
        levels=(0.5, 1.5),
        colors=("#D9481A",),
        alpha=0.72,
        transform=projection,
        zorder=5,
    )
    axis.contour(
        lon,
        lat,
        ever_total.astype(float),
        levels=(0.5,),
        colors=("white",),
        linewidths=0.9,
        transform=projection,
        zorder=6,
    )
    track_lon = np.asarray(track["longitude_deg"], dtype=float)
    track_lat = np.asarray(track["latitude_deg"], dtype=float)
    visible = (
        (track_lon >= US_EXTENT[0])
        & (track_lon <= US_EXTENT[1])
        & (track_lat >= US_EXTENT[2])
        & (track_lat <= US_EXTENT[3])
    )
    axis.plot(
        np.where(visible, track_lon, np.nan),
        np.where(visible, track_lat, np.nan),
        color="#4A1609",
        linewidth=1.6,
        transform=projection,
        zorder=10,
        label="Total-eclipse centerline",
    )
    axis.scatter(
        [ATLANTA[1]],
        [ATLANTA[0]],
        marker="*",
        s=170,
        facecolor="#FFF59D",
        edgecolor="#172033",
        linewidth=1.1,
        transform=projection,
        zorder=12,
        label="Atlanta",
    )
    axis.text(
        ATLANTA[1] + 1.0,
        ATLANTA[0] - 2.2,
        "ATL\n86.8 s totality",
        fontsize=9,
        color="#43160A",
        fontweight="bold",
        transform=projection,
        zorder=13,
    )
    axis.scatter(
        [closest["longitude_deg"]],
        [closest["latitude_deg"]],
        s=28,
        facecolor="white",
        edgecolor="#4A1609",
        linewidth=0.8,
        transform=projection,
        zorder=11,
    )
    figure.suptitle(
        "Eclipse #4 crosses Atlanta · 23 June 2027",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.072,
        "Atlanta totality: 19:36:43–19:38:10 EDT · maximum 19:37:26 EDT · "
        "duration 86.8 s · Sun altitude 13.2°",
        ha="center",
        fontsize=10,
        color="#303840",
    )
    figure.text(
        0.5,
        0.037,
        f"Centerline misses downtown by only {closest['distance_km']:.1f} km · "
        f"dark line: centerline · white edges: totality limits · "
        f"{grid_step_deg:.2f}° WGS84 / {time_step_seconds:.0f} s sampling",
        ha="center",
        fontsize=8.8,
        color="#606A73",
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "atlanta_zoom.png"
    figure.savefig(map_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    details_path = output / "atlanta_circumstances.json"
    details_path.write_text(
        json.dumps(
            {
                "event": event,
                "atlanta": atlanta.to_dict(),
                "closest_centerline": closest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"map": str(map_path.resolve()), "details": str(details_path.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="outputs/figures/2027_second_moon_total_tracks/events.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument("--output-dir", default="outputs/figures/20270623_eclipse4_atlanta")
    parser.add_argument("--grid-step-deg", type=float, default=0.10)
    parser.add_argument("--time-step-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = build_atlanta_zoom(
        events_path=args.events,
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
