"""Detailed 2-D map animation of the 2026 chained eclipse.

The map is an equirectangular (Plate Carree) North Atlantic view.  Every frame
recomputes the instantaneous topocentric solar/lunar disk geometry on a WGS84
surface grid.  The translucent footprints therefore represent eclipse
obscuration at that instant, not a precomputed event envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from skyfield.framelib import itrs

from .animation import build_default_scene
from .constants import (
    REAL_MOON_RADIUS_KM,
    SECOND_MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SPEED_OF_LIGHT_KM_S,
    SUN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_E2,
)
from .eclipse_geometry import central_point_hypothetical
from .ephemeris import central_point_real, time_iso_utc


REAL_BLUE = "#33B5FF"
SECOND_ORANGE = "#FF8A32"
MAP_EXTENT = (-80.0, 20.0, 40.0, 85.0)


def _overlap_fraction(
    separation: np.ndarray,
    sun_radius: np.ndarray,
    moon_radius: np.ndarray,
) -> np.ndarray:
    """Vectorized fraction of the apparent solar disk covered by a moon."""

    d = np.asarray(separation, dtype=float)
    rs = np.asarray(sun_radius, dtype=float)
    rm = np.asarray(moon_radius, dtype=float)
    result = np.zeros_like(d)
    contained = d <= np.abs(rm - rs)
    result[contained] = np.where(
        rm[contained] >= rs[contained],
        1.0,
        (rm[contained] / rs[contained]) ** 2,
    )
    partial = (d < rs + rm) & ~contained
    if np.any(partial):
        dp = d[partial]
        rsp = rs[partial]
        rmp = rm[partial]
        arg_s = np.clip(
            (dp * dp + rsp * rsp - rmp * rmp) / (2.0 * dp * rsp), -1.0, 1.0
        )
        arg_m = np.clip(
            (dp * dp + rmp * rmp - rsp * rsp) / (2.0 * dp * rmp), -1.0, 1.0
        )
        radicand = np.maximum(
            0.0,
            (-dp + rsp + rmp)
            * (dp + rsp - rmp)
            * (dp - rsp + rmp)
            * (dp + rsp + rmp),
        )
        area = (
            rsp * rsp * np.arccos(arg_s)
            + rmp * rmp * np.arccos(arg_m)
            - 0.5 * np.sqrt(radicand)
        )
        result[partial] = area / (np.pi * rsp * rsp)
    return np.clip(result, 0.0, 1.0)


class SurfaceGrid:
    """Fixed WGS84 observer grid for instantaneous topocentric masks."""

    def __init__(
        self,
        *,
        step_deg: float = 0.25,
        extent: tuple[float, float, float, float] = MAP_EXTENT,
    ) -> None:
        self.extent = extent
        west, east, south, north = extent
        self.longitude_deg = np.arange(west, east + 0.5 * step_deg, step_deg)
        self.latitude_deg = np.arange(south, north + 0.5 * step_deg, step_deg)
        lon, lat = np.meshgrid(self.longitude_deg, self.latitude_deg)
        self.shape = lon.shape
        lon_rad = np.radians(lon.ravel())
        lat_rad = np.radians(lat.ravel())
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        prime_vertical = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        self.observer_ecef = np.vstack(
            (
                prime_vertical * cos_lat * np.cos(lon_rad),
                prime_vertical * cos_lat * np.sin(lon_rad),
                prime_vertical * (1.0 - WGS84_E2) * sin_lat,
            )
        )
        self.normal_ecef = np.vstack(
            (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), sin_lat)
        )

    def obscuration(
        self,
        sun_ecef_km: np.ndarray,
        moon_ecef_km: np.ndarray,
        moon_radius_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return instantaneous obscuration and centrality on the grid."""

        sun_topo = sun_ecef_km[:, None] - self.observer_ecef
        moon_topo = moon_ecef_km[:, None] - self.observer_ecef
        sun_distance = np.linalg.norm(sun_topo, axis=0)
        moon_distance = np.linalg.norm(moon_topo, axis=0)
        sun_unit = sun_topo / sun_distance
        moon_unit = moon_topo / moon_distance
        dot = np.clip(np.sum(sun_unit * moon_unit, axis=0), -1.0, 1.0)
        cross = np.linalg.norm(np.cross(sun_unit.T, moon_unit.T), axis=1)
        separation = np.arctan2(cross, dot)
        sun_radius = np.arcsin(SUN_RADIUS_KM / sun_distance)
        moon_radius = np.arcsin(moon_radius_km / moon_distance)
        sun_above = np.sum(self.normal_ecef * sun_unit, axis=0) > 0.0
        obscuration = _overlap_fraction(separation, sun_radius, moon_radius)
        obscuration = np.where(sun_above, obscuration, 0.0)
        central = sun_above & (separation <= np.abs(moon_radius - sun_radius))
        return obscuration.reshape(self.shape), central.reshape(self.shape)

    def daylight(self, sun_ecef_km: np.ndarray) -> np.ndarray:
        sun_hat = np.asarray(sun_ecef_km, dtype=float)
        sun_hat /= np.linalg.norm(sun_hat)
        cosine = np.sum(self.normal_ecef * sun_hat[:, None], axis=0)
        return np.clip(cosine.reshape(self.shape), -1.0, 1.0)


class EclipseMapAnimator:
    """Plate Carree renderer with physically evaluated moving footprints."""

    def __init__(self, scene, times_tt_jd: np.ndarray, *, grid_step_deg: float = 0.25):
        self.scene = scene
        self.times = np.asarray(times_tt_jd, dtype=float)
        self.grid = SurfaceGrid(step_deg=grid_step_deg)
        self.real_max_tt = float(scene.context.time_utc(scene.event.real_maximum_utc).tt)
        self.second_max_tt = float(scene.context.time_utc(scene.event.second_maximum_utc).tt)
        self.real_trail: list[tuple[float, float]] = []
        self.second_trail: list[tuple[float, float]] = []

    def _frame_fields(self, tt_jd: float):
        time = self.scene.context.tt_jd(float(tt_jd))
        rotation = np.asarray(itrs.rotation_at(time), dtype=float)
        sun_icrf = np.asarray(
            self.scene.context.earth.at(time)
            .observe(self.scene.context.sun)
            .apparent()
            .position.km,
            dtype=float,
        )
        real_icrf = np.asarray(
            self.scene.context.earth.at(time)
            .observe(self.scene.context.moon)
            .apparent()
            .position.km,
            dtype=float,
        )
        second_now = np.asarray(self.scene.trajectory.position(float(tt_jd)), dtype=float)
        light_time_days = (
            np.linalg.norm(second_now) / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
        )
        second_icrf = np.asarray(
            self.scene.trajectory.position(float(tt_jd) - light_time_days), dtype=float
        )
        sun_ecef = rotation @ sun_icrf
        real_ecef = rotation @ real_icrf
        second_ecef = rotation @ second_icrf
        real_obscuration, real_central = self.grid.obscuration(
            sun_ecef, real_ecef, REAL_MOON_RADIUS_KM
        )
        second_obscuration, second_central = self.grid.obscuration(
            sun_ecef, second_ecef, SECOND_MOON_RADIUS_KM
        )
        daylight = self.grid.daylight(sun_ecef)
        real_point = central_point_real(self.scene.context, time)
        second_point = central_point_hypothetical(
            self.scene.context, time, second_now
        )
        return (
            time,
            daylight,
            real_obscuration,
            real_central,
            second_obscuration,
            second_central,
            real_point,
            second_point,
        )

    @staticmethod
    def _shadow_rgba(
        obscuration: np.ndarray,
        central: np.ndarray,
        rgb: tuple[float, float, float],
    ) -> np.ndarray:
        rgba = np.zeros((*obscuration.shape, 4), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(
            obscuration > 0.0,
            0.08 + 0.58 * np.power(obscuration, 0.58),
            0.0,
        )
        rgba[central, :3] = 1.0
        rgba[central, 3] = 0.98
        return rgba

    def render(self, output_path: str | Path, *, fps: int = 12, dpi: int = 140) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        projection = ccrs.PlateCarree()
        figure = plt.figure(figsize=(13.5, 7.2), facecolor="#07111D")
        axis = figure.add_axes((0.035, 0.10, 0.93, 0.80), projection=projection)
        axis.set_extent(MAP_EXTENT, crs=projection)
        axis.set_facecolor("#BBD8E8")
        axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#BBD8E8", zorder=0)
        axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#D6D0B4", zorder=1)
        axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#BBD8E8", zorder=2)
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor="#665F50",
            linewidth=0.45,
            alpha=0.75,
            zorder=8,
        )
        axis.coastlines(resolution="50m", color="#353B42", linewidth=0.65, zorder=9)
        gridlines = axis.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.4,
            color="#52606D",
            alpha=0.45,
            linestyle=":",
            zorder=3,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 8, "color": "#DCE6EF"}
        gridlines.ylabel_style = {"size": 8, "color": "#DCE6EF"}

        empty = np.zeros((*self.grid.shape, 4), dtype=float)
        extent = (
            self.grid.longitude_deg[0],
            self.grid.longitude_deg[-1],
            self.grid.latitude_deg[0],
            self.grid.latitude_deg[-1],
        )
        night_image = axis.imshow(
            empty,
            extent=extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=4,
        )
        real_image = axis.imshow(
            empty,
            extent=extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=5,
        )
        second_image = axis.imshow(
            empty,
            extent=extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=6,
        )
        real_trail_line, = axis.plot(
            [], [], color=REAL_BLUE, linewidth=1.5, transform=projection, zorder=10
        )
        second_trail_line, = axis.plot(
            [], [], color=SECOND_ORANGE, linewidth=1.5, transform=projection, zorder=10
        )
        real_marker = axis.scatter(
            [], [], s=58, facecolor=REAL_BLUE, edgecolor="white", linewidth=1.1,
            transform=projection, zorder=12, label="Real Moon central shadow"
        )
        second_marker = axis.scatter(
            [], [], s=58, facecolor=SECOND_ORANGE, edgecolor="white", linewidth=1.1,
            transform=projection, zorder=12, label="Second moon central shadow"
        )
        axis.scatter(
            [self.scene.event.longitude_deg],
            [self.scene.event.latitude_deg],
            marker="*",
            s=95,
            facecolor="white",
            edgecolor="#172033",
            linewidth=0.8,
            transform=projection,
            zorder=13,
            label="Best common observing site",
        )
        legend = axis.legend(
            loc="lower left",
            frameon=True,
            facecolor="#07111D",
            edgecolor="#718096",
            framealpha=0.88,
            fontsize=8,
        )
        for text in legend.get_texts():
            text.set_color("white")
        title = figure.text(0.5, 0.955, "", color="white", ha="center", fontsize=14)
        status = figure.text(0.5, 0.035, "", color="#C8D5E3", ha="center", fontsize=9)

        def update(index: int):
            tt_jd = float(self.times[index])
            (
                time,
                daylight,
                real_obscuration,
                real_central,
                second_obscuration,
                second_central,
                real_point,
                second_point,
            ) = self._frame_fields(tt_jd)
            night_rgba = np.zeros((*daylight.shape, 4), dtype=float)
            night_rgba[..., :3] = (0.015, 0.035, 0.065)
            night_rgba[..., 3] = 0.62 * np.clip(-daylight, 0.0, 1.0) ** 0.35
            night_image.set_data(night_rgba)
            real_image.set_data(
                self._shadow_rgba(real_obscuration, real_central, (0.08, 0.54, 0.91))
            )
            second_image.set_data(
                self._shadow_rgba(
                    second_obscuration, second_central, (0.95, 0.36, 0.08)
                )
            )

            if real_point is not None:
                real_lat, real_lon = float(real_point[0]), float(real_point[1])
                self.real_trail.append((real_lon, real_lat))
                real_marker.set_offsets([[real_lon, real_lat]])
            else:
                real_marker.set_offsets(np.empty((0, 2)))
            if second_point is not None:
                second_lat, second_lon = float(second_point[0]), float(second_point[1])
                self.second_trail.append((second_lon, second_lat))
                second_marker.set_offsets([[second_lon, second_lat]])
            else:
                second_marker.set_offsets(np.empty((0, 2)))
            trail_frames = max(2, int(5.0 * 60.0 / ((self.times[1] - self.times[0]) * SECONDS_PER_DAY)))
            real_recent = self.real_trail[-trail_frames:]
            second_recent = self.second_trail[-trail_frames:]
            real_trail_line.set_data(
                [point[0] for point in real_recent], [point[1] for point in real_recent]
            )
            second_trail_line.set_data(
                [point[0] for point in second_recent],
                [point[1] for point in second_recent],
            )

            real_delta = (tt_jd - self.real_max_tt) * SECONDS_PER_DAY
            second_delta = (tt_jd - self.second_max_tt) * SECONDS_PER_DAY
            title.set_text(f"{time_iso_utc(time, places=1)} · chained eclipse over Greenland")
            status.set_text(
                f"Real Moon maximum {real_delta:+.0f} s   ·   "
                f"Second-moon maximum {second_delta:+.0f} s   ·   "
                "instantaneous topocentric obscuration on WGS84"
            )
            return (
                night_image,
                real_image,
                second_image,
                real_trail_line,
                second_trail_line,
                real_marker,
                second_marker,
                title,
                status,
            )

        movie = mpl_animation.FuncAnimation(
            figure,
            update,
            frames=len(self.times),
            interval=1000.0 / fps,
            blit=False,
        )
        if path.suffix.lower() == ".gif":
            writer = mpl_animation.PillowWriter(fps=fps)
        elif path.suffix.lower() == ".mp4":
            writer = mpl_animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=5_000,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        else:
            raise ValueError("animation output must end in .mp4 or .gif")
        movie.save(path, writer=writer, dpi=dpi)
        plt.close(figure)
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="outputs/events.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output", default="outputs/animations/chained_eclipse_20260812_2d_map.mp4"
    )
    parser.add_argument("--frames", type=int, default=361)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--lead-minutes", type=float, default=12.0)
    parser.add_argument("--trail-minutes", type=float, default=12.0)
    parser.add_argument("--grid-step-deg", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    scene, _ = build_default_scene(
        events_path=args.events,
        optimized_config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        end_margin_minutes=args.trail_minutes,
    )
    real_tt = float(scene.context.time_utc(scene.event.real_maximum_utc).tt)
    second_tt = float(scene.context.time_utc(scene.event.second_maximum_utc).tt)
    times = np.linspace(
        real_tt - args.lead_minutes * 60.0 / SECONDS_PER_DAY,
        second_tt + args.trail_minutes * 60.0 / SECONDS_PER_DAY,
        args.frames,
    )
    output = EclipseMapAnimator(
        scene, times, grid_step_deg=args.grid_step_deg
    ).render(args.output, fps=args.fps, dpi=args.dpi)
    manifest = {
        "schema_version": "1.0",
        "event_id": scene.event.event_id,
        "output": str(output.resolve()),
        "projection": "Plate Carree (equirectangular)",
        "extent_deg": list(MAP_EXTENT),
        "grid_step_deg": args.grid_step_deg,
        "frames": args.frames,
        "fps": args.fps,
        "screen_duration_s": args.frames / args.fps,
        "start_utc": time_iso_utc(scene.context.tt_jd(float(times[0])), places=1),
        "end_utc": time_iso_utc(scene.context.tt_jd(float(times[-1])), places=1),
        "footprints": "instantaneous topocentric disk-overlap obscuration on WGS84",
        "second_moon_light_time": "one geocentric retarded-position iteration per frame",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"2-D chained-eclipse map: {output.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
