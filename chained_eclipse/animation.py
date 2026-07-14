"""Scientifically grounded 3-D animation of a chained solar eclipse.

The renderer deliberately separates two spatial scales:

* an inertial, true-centre orbit view containing Earth and both moons; and
* a near-Earth view in which the physically sized core cones sweep across a
  rotating WGS84 ellipsoid.

Moon marker sizes in the orbit view are enlarged for visibility.  Positions,
Earth shape, cone axes, cone opening angles, and time labels are not enlarged
or time-shifted.  The fictional moon uses the same restricted numerical
trajectory as the eclipse search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import shapely
import yaml
from cartopy.io import shapereader
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from skyfield.framelib import itrs

from .constants import (
    REAL_MOON_RADIUS_KM,
    SECOND_MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SUN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_B_KM,
)
from .eclipse_geometry import central_point_hypothetical, hypothetical_shadow_state
from .ephemeris import (
    EphemerisContext,
    central_point_real,
    load_ephemeris,
    real_shadow_state,
    time_iso_utc,
)
from .models import OrbitalElements, Trajectory
from .orbital_dynamics import integrate_restricted


REAL_COLOR = "#5FC8FF"
SECOND_COLOR = "#FF9C4A"
EARTH_DAY = np.asarray((0.20, 0.43, 0.67, 1.0))
EARTH_NIGHT = np.asarray((0.035, 0.065, 0.11, 1.0))
LAND_DAY = np.asarray((0.34, 0.51, 0.32, 1.0))
LAND_NIGHT = np.asarray((0.055, 0.095, 0.07, 1.0))


@dataclass(frozen=True, slots=True)
class AnimationEvent:
    """The two maxima and common observing site used to frame an animation."""

    real_maximum_utc: str
    second_maximum_utc: str
    latitude_deg: float
    longitude_deg: float
    event_id: str = "chained-eclipse"


@dataclass(frozen=True, slots=True)
class AnimationFrame:
    """All physical state needed to render one instant."""

    tt_jd: float
    time_utc: str
    sun_from_earth_km: np.ndarray
    real_moon_from_earth_km: np.ndarray
    second_moon_from_earth_km: np.ndarray
    real_axis_icrf: np.ndarray
    second_axis_icrf: np.ndarray
    real_sun_moon_distance_km: float
    second_sun_moon_distance_km: float
    real_core_intersection: tuple[float, float, float, float] | None
    second_core_intersection: tuple[float, float, float, float] | None


def load_animation_inputs(
    events_path: str | Path,
    optimized_config_path: str | Path,
) -> tuple[AnimationEvent, OrbitalElements]:
    """Load the first fixed-system event and its exact saved orbit."""

    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    candidates = events.get("fixed_system_events", [])
    if not candidates:
        raise ValueError("events file contains no fixed-system chained eclipse")
    candidate = candidates[0]
    event = AnimationEvent(
        real_maximum_utc=candidate["real_eclipse"]["maximum_utc"],
        second_maximum_utc=candidate["second_eclipse"]["maximum_utc"],
        latitude_deg=float(candidate["best_latitude_deg"]),
        longitude_deg=float(candidate["best_longitude_deg"]),
        event_id=str(candidate["event_id"]),
    )
    config = yaml.safe_load(Path(optimized_config_path).read_text(encoding="utf-8"))
    return event, OrbitalElements(**config["orbital_elements"])


class EclipseAnimationScene:
    """Physical frame sampler shared by movie and still renderers."""

    def __init__(
        self,
        context: EphemerisContext,
        trajectory: Trajectory,
        event: AnimationEvent,
    ) -> None:
        self.context = context
        self.trajectory = trajectory
        self.event = event

    def sample(self, tt_jd: float) -> AnimationFrame:
        """Evaluate DE440s, the numerical moon, and both shadow axes."""

        time = self.context.tt_jd(float(tt_jd))
        earth = np.asarray(self.context.earth.at(time).position.km, dtype=float)
        sun = np.asarray(self.context.sun.at(time).position.km, dtype=float) - earth
        real_moon = np.asarray(self.context.moon.at(time).position.km, dtype=float) - earth
        second_moon = np.asarray(self.trajectory.position(float(tt_jd)), dtype=float)
        real_shadow = real_shadow_state(self.context, time)
        second_shadow = hypothetical_shadow_state(self.context, time, second_moon)
        return AnimationFrame(
            tt_jd=float(tt_jd),
            time_utc=time_iso_utc(time, places=1),
            sun_from_earth_km=sun,
            real_moon_from_earth_km=real_moon,
            second_moon_from_earth_km=second_moon,
            real_axis_icrf=real_shadow.axis_icrf,
            second_axis_icrf=second_shadow.axis_icrf,
            real_sun_moon_distance_km=real_shadow.sun_moon_distance_km,
            second_sun_moon_distance_km=second_shadow.sun_moon_distance_km,
            real_core_intersection=central_point_real(self.context, time),
            second_core_intersection=central_point_hypothetical(
                self.context, time, second_moon
            ),
        )


def build_default_scene(
    *,
    events_path: str | Path = "outputs/events.json",
    optimized_config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    end_margin_minutes: float = 35.0,
) -> tuple[EclipseAnimationScene, AnimationEvent]:
    """Load the result bundle and integrate the second moon through the movie."""

    event, elements = load_animation_inputs(events_path, optimized_config_path)
    context = load_ephemeris(ephemeris_cache)
    end_tt = float(context.time_utc(event.second_maximum_utc).tt) + (
        end_margin_minutes * 60.0 / SECONDS_PER_DAY
    )
    trajectory = integrate_restricted(
        context,
        elements,
        end_tt,
        include_j2=True,
        rtol=3e-11,
        max_step_seconds=7_200.0,
    )
    return EclipseAnimationScene(context, trajectory, event), event


def animation_times(
    scene: EclipseAnimationScene,
    *,
    lead_minutes: float = 12.0,
    trail_minutes: float = 12.0,
    frames: int = 721,
) -> np.ndarray:
    """Return a uniformly spaced TT time base containing both maxima."""

    if frames < 2:
        raise ValueError("an animation needs at least two frames")
    start = float(scene.context.time_utc(scene.event.real_maximum_utc).tt)
    end = float(scene.context.time_utc(scene.event.second_maximum_utc).tt)
    return np.linspace(
        start - lead_minutes * 60.0 / SECONDS_PER_DAY,
        end + trail_minutes * 60.0 / SECONDS_PER_DAY,
        frames,
    )


def _earth_mesh_itrs(rows: int = 49, columns: int = 97) -> tuple[np.ndarray, ...]:
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, rows)
    longitude = np.linspace(-np.pi, np.pi, columns)
    lon, lat = np.meshgrid(longitude, latitude)
    x = WGS84_A_KM * np.cos(lat) * np.cos(lon)
    y = WGS84_A_KM * np.cos(lat) * np.sin(lon)
    z = WGS84_B_KM * np.sin(lat)
    normals = np.stack(
        (x / WGS84_A_KM**2, y / WGS84_A_KM**2, z / WGS84_B_KM**2), axis=-1
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return x, y, z, normals


def _coastline_itrs() -> list[np.ndarray]:
    """Read cached Natural Earth coastlines into Earth-fixed Cartesian lines."""

    path = shapereader.natural_earth(
        resolution="50m", category="physical", name="coastline"
    )
    segments: list[np.ndarray] = []
    for geometry in shapereader.Reader(path).geometries():
        parts: Iterable = geometry.geoms if hasattr(geometry, "geoms") else (geometry,)
        for part in parts:
            coordinates = np.asarray(part.coords, dtype=float)
            lon = np.radians(coordinates[:, 0])
            lat = np.radians(coordinates[:, 1])
            # Geodetic WGS84 surface point, not a spherical approximation.
            prime_vertical = WGS84_A_KM / np.sqrt(
                1.0 - (1.0 - WGS84_B_KM**2 / WGS84_A_KM**2) * np.sin(lat) ** 2
            )
            x = prime_vertical * np.cos(lat) * np.cos(lon)
            y = prime_vertical * np.cos(lat) * np.sin(lon)
            z = (WGS84_B_KM**2 / WGS84_A_KM**2) * prime_vertical * np.sin(lat)
            segments.append(np.column_stack((x, y, z)))
    return segments


def _land_mask(rows: int = 49, columns: int = 97) -> np.ndarray:
    """Sample Natural Earth land polygons on the globe render mesh."""

    path = shapereader.natural_earth(
        resolution="50m", category="physical", name="land"
    )
    land = shapely.union_all(list(shapereader.Reader(path).geometries()))
    longitude = np.linspace(-180.0, 180.0, columns)
    latitude = np.linspace(-90.0, 90.0, rows)
    lon, lat = np.meshgrid(longitude, latitude)
    return np.asarray(shapely.contains_xy(land, lon, lat), dtype=bool)


def _transform_itrs(points: np.ndarray, rotation_icrf_to_itrs: np.ndarray) -> np.ndarray:
    flat = np.asarray(points, dtype=float).reshape(-1, 3)
    return (rotation_icrf_to_itrs.T @ flat.T).T.reshape(points.shape)


def _surface_point_icrf(
    latitude_deg: float,
    longitude_deg: float,
    rotation_icrf_to_itrs: np.ndarray,
) -> np.ndarray:
    lon = np.radians(longitude_deg)
    lat = np.radians(latitude_deg)
    n = WGS84_A_KM / np.sqrt(
        1.0 - (1.0 - WGS84_B_KM**2 / WGS84_A_KM**2) * np.sin(lat) ** 2
    )
    itrs_point = np.asarray(
        (
            n * np.cos(lat) * np.cos(lon),
            n * np.cos(lat) * np.sin(lon),
            (WGS84_B_KM**2 / WGS84_A_KM**2) * n * np.sin(lat),
        )
    )
    return rotation_icrf_to_itrs.T @ itrs_point


def _cone_surface(
    moon_position: np.ndarray,
    axis: np.ndarray,
    moon_radius_km: float,
    sun_distance_km: float,
    *,
    near_earth_only: bool,
    radial_samples: int = 20,
    azimuth_samples: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the finite core cone and its signed radius along the axis."""

    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    center_distance = float(np.dot(-np.asarray(moon_position, dtype=float), axis))
    if near_earth_only:
        along = np.linspace(max(0.0, center_distance - 10_000.0), center_distance + 10_000.0, radial_samples)
    else:
        along = np.linspace(0.0, center_distance + 10_000.0, radial_samples)
    angle = np.arcsin((SUN_RADIUS_KM - moon_radius_km) / sun_distance_km)
    signed_radius = moon_radius_km / np.cos(angle) - along * np.tan(angle)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    theta = np.linspace(0.0, 2.0 * np.pi, azimuth_samples)
    ring = np.cos(theta)[None, :, None] * basis_u + np.sin(theta)[None, :, None] * basis_v
    centers = moon_position[None, None, :] + along[:, None, None] * axis
    surface = centers + np.abs(signed_radius)[:, None, None] * ring
    return surface[..., 0], surface[..., 1], surface[..., 2], signed_radius


def _set_equal_limits(ax: Axes3D, radius: float) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


class MatplotlibEclipseAnimator:
    """Two-scale 3-D renderer for an :class:`EclipseAnimationScene`."""

    def __init__(self, scene: EclipseAnimationScene, times_tt_jd: np.ndarray) -> None:
        self.scene = scene
        self.times = np.asarray(times_tt_jd, dtype=float)
        self.earth_x, self.earth_y, self.earth_z, self.earth_normals = _earth_mesh_itrs()
        self.land_mask = _land_mask()
        self.coastlines = _coastline_itrs()
        self.real_max_tt = float(scene.context.time_utc(scene.event.real_maximum_utc).tt)
        self.second_max_tt = float(scene.context.time_utc(scene.event.second_maximum_utc).tt)
        focus_tt = 0.5 * (self.real_max_tt + self.second_max_tt)
        focus_rotation = np.asarray(
            itrs.rotation_at(scene.context.tt_jd(focus_tt)), dtype=float
        )
        focus = _surface_point_icrf(
            scene.event.latitude_deg,
            scene.event.longitude_deg,
            focus_rotation,
        )
        focus /= np.linalg.norm(focus)
        self.close_elevation_deg = float(np.degrees(np.arcsin(focus[2])))
        self.close_azimuth_deg = float(np.degrees(np.arctan2(focus[1], focus[0])))

    def _draw_earth(self, ax: Axes3D, frame: AnimationFrame) -> None:
        time = self.scene.context.tt_jd(frame.tt_jd)
        rotation = np.asarray(itrs.rotation_at(time), dtype=float)
        points = np.stack((self.earth_x, self.earth_y, self.earth_z), axis=-1)
        points_icrf = _transform_itrs(points, rotation)
        normals_icrf = _transform_itrs(self.earth_normals, rotation)
        sun_hat = frame.sun_from_earth_km / np.linalg.norm(frame.sun_from_earth_km)
        illumination = np.clip(np.sum(normals_icrf * sun_hat, axis=-1), 0.0, 1.0)
        night = np.where(self.land_mask[..., None], LAND_NIGHT, EARTH_NIGHT)
        day = np.where(self.land_mask[..., None], LAND_DAY, EARTH_DAY)
        colors = night + illumination[..., None] * (day - night)
        ax.plot_surface(
            points_icrf[..., 0],
            points_icrf[..., 1],
            points_icrf[..., 2],
            facecolors=colors,
            linewidth=0.0,
            shade=False,
            antialiased=False,
            alpha=1.0,
        )
        for segment in self.coastlines:
            # Lift the cartographic line a few kilometres so mplot3d does not
            # z-fight it into the ellipsoid surface.
            line = _transform_itrs(segment, rotation) * 1.0015
            ax.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color="#DCEBFA",
                linewidth=0.55,
                alpha=0.9,
            )

    def _draw_shadow(
        self,
        ax: Axes3D,
        frame: AnimationFrame,
        *,
        moon_position: np.ndarray,
        axis: np.ndarray,
        moon_radius: float,
        color: str,
        near_earth_only: bool,
        intersection: tuple[float, float, float, float] | None,
        sun_moon_distance_km: float,
    ) -> None:
        x, y, z, signed_radius = _cone_surface(
            moon_position,
            axis,
            moon_radius,
            sun_moon_distance_km,
            near_earth_only=near_earth_only,
        )
        ax.plot_surface(x, y, z, color=color, alpha=0.13, linewidth=0.0, shade=False)
        # A solid axis prevents the translucent cone from becoming visually ambiguous.
        centers = np.column_stack((x.mean(axis=1), y.mean(axis=1), z.mean(axis=1)))
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, linewidth=1.1)
        apex_indices = np.flatnonzero(np.signbit(signed_radius[:-1]) != np.signbit(signed_radius[1:]))
        if len(apex_indices):
            apex = centers[int(apex_indices[0])]
            ax.scatter(*apex, color=color, s=7)
        if near_earth_only and intersection is not None:
            latitude, longitude, _, _ = intersection
            rotation = np.asarray(itrs.rotation_at(self.scene.context.tt_jd(frame.tt_jd)), dtype=float)
            point = _surface_point_icrf(latitude, longitude, rotation) * 1.003
            ax.scatter(*point, color=color, s=24, edgecolors="white", linewidths=0.45, depthshade=False)

    def draw_frame(self, figure: Figure, frame_index: int) -> tuple[Axes3D, Axes3D]:
        """Draw one complete two-scale frame into ``figure``."""

        figure.clear()
        frame = self.scene.sample(float(self.times[frame_index]))
        layout = figure.add_gridspec(1, 2, width_ratios=(1.35, 1.0), wspace=0.02)
        close = figure.add_subplot(layout[0, 0], projection="3d")
        orbit = figure.add_subplot(layout[0, 1], projection="3d")
        for axis in (close, orbit):
            axis.set_facecolor("#050914")
            axis.set_axis_off()
        # Keep the close camera on the Greenland/Iceland corridor while the
        # WGS84 Earth rotates beneath the inertial shadow axes.
        close.view_init(elev=self.close_elevation_deg, azim=self.close_azimuth_deg)
        orbit.view_init(elev=24.0, azim=-56.0)

        self._draw_earth(close, frame)
        self._draw_shadow(
            close,
            frame,
            moon_position=frame.real_moon_from_earth_km,
            axis=frame.real_axis_icrf,
            moon_radius=REAL_MOON_RADIUS_KM,
            color=REAL_COLOR,
            near_earth_only=True,
            intersection=frame.real_core_intersection,
            sun_moon_distance_km=frame.real_sun_moon_distance_km,
        )
        self._draw_shadow(
            close,
            frame,
            moon_position=frame.second_moon_from_earth_km,
            axis=frame.second_axis_icrf,
            moon_radius=SECOND_MOON_RADIUS_KM,
            color=SECOND_COLOR,
            near_earth_only=True,
            intersection=frame.second_core_intersection,
            sun_moon_distance_km=frame.second_sun_moon_distance_km,
        )
        time = self.scene.context.tt_jd(frame.tt_jd)
        rotation = np.asarray(itrs.rotation_at(time), dtype=float)
        common_site = _surface_point_icrf(
            self.scene.event.latitude_deg,
            self.scene.event.longitude_deg,
            rotation,
        ) * 1.006
        close.scatter(
            *common_site,
            color="white",
            marker="*",
            s=28,
            linewidths=0.25,
            depthshade=False,
        )
        close.text(
            *(common_site * 1.015),
            "common site",
            color="white",
            fontsize=7,
            ha="center",
        )
        _set_equal_limits(close, 7_500.0)
        close.set_title("Shadow cones at Earth · physical scale", color="white", pad=4)

        orbit.scatter(0.0, 0.0, 0.0, color="#2E75B6", s=24, label="Earth")
        orbit.scatter(*frame.real_moon_from_earth_km, color=REAL_COLOR, s=32, label="Real Moon")
        orbit.scatter(*frame.second_moon_from_earth_km, color=SECOND_COLOR, s=32, label="Second moon")
        self._draw_shadow(
            orbit,
            frame,
            moon_position=frame.real_moon_from_earth_km,
            axis=frame.real_axis_icrf,
            moon_radius=REAL_MOON_RADIUS_KM,
            color=REAL_COLOR,
            near_earth_only=False,
            intersection=None,
            sun_moon_distance_km=frame.real_sun_moon_distance_km,
        )
        self._draw_shadow(
            orbit,
            frame,
            moon_position=frame.second_moon_from_earth_km,
            axis=frame.second_axis_icrf,
            moon_radius=SECOND_MOON_RADIUS_KM,
            color=SECOND_COLOR,
            near_earth_only=False,
            intersection=None,
            sun_moon_distance_km=frame.second_sun_moon_distance_km,
        )
        sun_hat = frame.sun_from_earth_km / np.linalg.norm(frame.sun_from_earth_km)
        orbit.quiver(0.0, 0.0, 0.0, *(sun_hat * 90_000.0), color="#FFD56A", linewidth=1.2, arrow_length_ratio=0.08)
        orbit.text(*(sun_hat * 105_000.0), "Sun", color="#FFD56A", ha="center")
        _set_equal_limits(orbit, 430_000.0)
        orbit.set_title("True centre positions · moon markers enlarged", color="white", pad=4)
        legend = orbit.legend(loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color("white")

        real_delta = (frame.tt_jd - self.real_max_tt) * SECONDS_PER_DAY
        second_delta = (frame.tt_jd - self.second_max_tt) * SECONDS_PER_DAY
        if abs(real_delta) <= abs(second_delta):
            phase = f"Real Moon maximum {real_delta:+.0f} s"
        else:
            phase = f"Second-moon maximum {second_delta:+.0f} s"
        figure.suptitle(
            f"{frame.time_utc}   ·   {phase}",
            color="white",
            fontsize=13,
            y=0.965,
        )
        figure.text(
            0.5,
            0.018,
            "DE440s Sun/Earth/real Moon · numerically propagated second moon · inertial ICRF camera",
            color="#B8C4D6",
            ha="center",
            fontsize=8,
        )
        return close, orbit

    def render(
        self,
        output_path: str | Path,
        *,
        fps: int = 30,
        dpi: int = 140,
    ) -> Path:
        """Encode an MP4 (H.264) or GIF based on the output suffix."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(13.5, 7.2), facecolor="#050914")

        def update(index: int):
            self.draw_frame(figure, index)
            return ()

        animation = mpl_animation.FuncAnimation(
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
                bitrate=4_500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        else:
            raise ValueError("animation output must end in .mp4 or .gif")
        animation.save(path, writer=writer, dpi=dpi)
        plt.close(figure)
        return path

    def render_still(self, output_path: str | Path, *, frame_index: int, dpi: int = 150) -> Path:
        """Render one movie frame for inexpensive visual QA."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(13.5, 7.2), facecolor="#050914")
        self.draw_frame(figure, frame_index)
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
        plt.close(figure)
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="outputs/events.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output", default="outputs/animations/chained_eclipse_20260812_3d.mp4"
    )
    parser.add_argument("--frames", type=int, default=721)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lead-minutes", type=float, default=12.0)
    parser.add_argument("--trail-minutes", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    scene, _ = build_default_scene(
        events_path=args.events,
        optimized_config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        end_margin_minutes=args.trail_minutes,
    )
    times = animation_times(
        scene,
        lead_minutes=args.lead_minutes,
        trail_minutes=args.trail_minutes,
        frames=args.frames,
    )
    output_path = MatplotlibEclipseAnimator(scene, times).render(
        args.output, fps=args.fps, dpi=args.dpi
    )
    manifest = {
        "schema_version": "1.0",
        "event_id": scene.event.event_id,
        "output": str(output_path.resolve()),
        "start_utc": time_iso_utc(scene.context.tt_jd(float(times[0])), places=1),
        "end_utc": time_iso_utc(scene.context.tt_jd(float(times[-1])), places=1),
        "frames": int(len(times)),
        "fps": int(args.fps),
        "screen_duration_s": len(times) / args.fps,
        "real_maximum_utc": scene.event.real_maximum_utc,
        "second_maximum_utc": scene.event.second_maximum_utc,
        "common_site": {
            "latitude_deg": scene.event.latitude_deg,
            "longitude_deg": scene.event.longitude_deg,
        },
        "physics": {
            "real_bodies": "JPL DE440s",
            "second_moon": "DOP853 restricted propagation with Earth J2 and prescribed Sun/Moon perturbations",
            "earth": "rotating WGS84 ellipsoid",
            "shadow_axes": "existing apparent-axis eclipse geometry",
        },
        "display_scaling": {
            "earth_and_shadow_cones": "physical scale in close view",
            "orbital_centres": "physical scale in wide view",
            "moon_markers": "enlarged display glyphs in wide view",
            "sun": "off-screen direction arrow",
        },
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"3-D chained-eclipse animation: {output_path.resolve()}")
    print(f"Animation manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
