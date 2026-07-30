"""Animate both coupled lunar shadows across a flat world map.

This is the quick-look companion to the more elaborate globe renderer.  Each
frame evaluates topocentric Sun/moon disk overlap on a rotating WGS84 grid, so
the colored footprints are instantaneous eclipse visibility rather than
decorative track widths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.animation as mpl_animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .animation_2d import SurfaceGrid
from .constants import SECONDS_PER_DAY, SPEED_OF_LIGHT_KM_S
from .coupled_eclipse import (
    CoupledEphemeris,
    _earth_rotation_matrix,
    body_radius_km,
    coupled_central_point,
)
from .ephemeris import load_ephemeris, time_iso_utc
from .moon_architecture import architecture_from_config, elements_from_config


REAL_BLUE = (0.10, 0.45, 0.95)
GIANT_ORANGE = (1.00, 0.38, 0.08)
WORLD_EXTENT = (-180.0, 180.0, -70.0, 85.0)


def _matching_event(
    result: dict[str, Any],
    pair: dict[str, Any],
    *,
    events_key: str,
    maximum_key: str,
) -> dict[str, Any]:
    """Return the event row belonging to one member of a saved pair."""

    maximum = pair[maximum_key]
    for event in result[events_key]:
        if event["axis_maximum_utc"] == maximum:
            return event
    raise ValueError(f"no {events_key} row matches {maximum_key}={maximum!r}")


def animation_window_utc(
    result: dict[str, Any],
    pair: dict[str, Any],
    *,
    lead_minutes: float = 10.0,
    trail_minutes: float = 10.0,
) -> tuple[str, str]:
    """Return a window spanning both saved global eclipse footprints."""

    real = _matching_event(
        result,
        pair,
        events_key="real_moon_events",
        maximum_key="real_maximum_utc",
    )
    giant = _matching_event(
        result,
        pair,
        events_key="second_moon_events",
        maximum_key="second_maximum_utc",
    )
    context_start = min(real["global_start_utc"], giant["global_start_utc"])
    context_end = max(real["global_end_utc"], giant["global_end_utc"])
    # Keep this helper independent of Skyfield so its event-selection behavior
    # remains cheap to test.  The minute padding is applied after parsing in
    # ``main``.
    if lead_minutes < 0.0 or trail_minutes < 0.0:
        raise ValueError("lead_minutes and trail_minutes must be non-negative")
    return context_start, context_end


def _retarded_geocentric_vector(
    ephemeris: CoupledEphemeris,
    body: str,
    tt_jd: float,
) -> np.ndarray:
    """Return apparent body direction from Earth's center at reception time."""

    earth_reception = np.asarray(ephemeris.position("earth", tt_jd), dtype=float)
    body_now = np.asarray(ephemeris.position(body, tt_jd), dtype=float)
    initial = body_now - earth_reception
    retarded_tt = tt_jd - np.linalg.norm(initial) / SPEED_OF_LIGHT_KM_S / SECONDS_PER_DAY
    return np.asarray(ephemeris.position(body, retarded_tt), dtype=float) - earth_reception


def _break_dateline(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert gaps where a Plate Carree trail wraps across the date line."""

    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    if longitude.shape != latitude.shape:
        raise ValueError("longitude and latitude arrays must have the same shape")
    if longitude.ndim != 1:
        raise ValueError("trail coordinates must be one-dimensional")
    output_longitude: list[float] = []
    output_latitude: list[float] = []
    previous: float | None = None
    for lon, lat in zip(longitude, latitude, strict=True):
        if not np.isfinite(lon) or not np.isfinite(lat):
            output_longitude.append(float("nan"))
            output_latitude.append(float("nan"))
            previous = None
            continue
        if previous is not None and abs(lon - previous) > 180.0:
            output_longitude.append(float("nan"))
            output_latitude.append(float("nan"))
        output_longitude.append(float(lon))
        output_latitude.append(float(lat))
        previous = float(lon)
    return np.asarray(output_longitude), np.asarray(output_latitude)


class CoupledEclipseMapAnimator:
    """World-map renderer for a coupled real-Moon/giant-Moon eclipse pair."""

    def __init__(
        self,
        ephemeris: CoupledEphemeris,
        times_tt_jd: np.ndarray,
        pair: dict[str, Any],
        *,
        grid_step_deg: float = 1.5,
        extent: tuple[float, float, float, float] = WORLD_EXTENT,
    ) -> None:
        times = np.asarray(times_tt_jd, dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("times_tt_jd must contain at least two increasing values")
        self.ephemeris = ephemeris
        self.times = times
        self.pair = pair
        self.grid = SurfaceGrid(step_deg=grid_step_deg, extent=extent)
        self.extent = extent
        self.real_track, self.giant_track = self._central_tracks()

    def _central_tracks(self) -> tuple[np.ndarray, np.ndarray]:
        tracks: list[np.ndarray] = []
        for body in ("real_moon", "second_moon"):
            track = np.full((len(self.times), 2), np.nan, dtype=float)
            for index, tt_jd in enumerate(self.times):
                point = coupled_central_point(self.ephemeris, body, float(tt_jd))
                if point is not None:
                    track[index] = (float(point[1]), float(point[0]))
            tracks.append(track)
        return tracks[0], tracks[1]

    def _frame_fields(
        self,
        tt_jd: float,
    ) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rotation = _earth_rotation_matrix(self.ephemeris, tt_jd)
        sun_ecef = rotation @ _retarded_geocentric_vector(self.ephemeris, "sun", tt_jd)
        real_ecef = rotation @ _retarded_geocentric_vector(self.ephemeris, "real_moon", tt_jd)
        giant_ecef = rotation @ _retarded_geocentric_vector(self.ephemeris, "second_moon", tt_jd)
        real_obscuration, real_central = self.grid.obscuration(
            sun_ecef,
            real_ecef,
            body_radius_km(self.ephemeris, "real_moon"),
        )
        giant_obscuration, giant_central = self.grid.obscuration(
            sun_ecef,
            giant_ecef,
            body_radius_km(self.ephemeris, "second_moon"),
        )
        return (
            self.ephemeris.time(tt_jd),
            self.grid.daylight(sun_ecef),
            real_obscuration,
            real_central,
            giant_obscuration,
            giant_central,
        )

    @staticmethod
    def _shadow_rgba(
        obscuration: np.ndarray,
        central: np.ndarray,
        rgb: tuple[float, float, float],
        *,
        maximum_alpha: float = 0.70,
    ) -> np.ndarray:
        rgba = np.zeros((*obscuration.shape, 4), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(
            obscuration > 0.0,
            0.04 + maximum_alpha * np.power(obscuration, 0.62),
            0.0,
        )
        rgba[central, :3] = np.asarray(rgb) * 0.24
        rgba[central, 3] = 0.96
        return rgba

    def _envelopes(self, maximum_samples: int = 48) -> tuple[np.ndarray, np.ndarray]:
        indices = np.unique(
            np.linspace(
                0,
                len(self.times) - 1,
                min(maximum_samples, len(self.times)),
                dtype=int,
            )
        )
        real = np.zeros(self.grid.shape, dtype=float)
        giant = np.zeros(self.grid.shape, dtype=float)
        for index in indices:
            _, _, real_now, _, giant_now, _ = self._frame_fields(float(self.times[index]))
            np.maximum(real, real_now, out=real)
            np.maximum(giant, giant_now, out=giant)
        return real, giant

    def render(
        self,
        output_path: str | Path,
        *,
        fps: int = 8,
        dpi: int = 100,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        projection = ccrs.PlateCarree()
        figure = plt.figure(figsize=(13.6, 7.2), facecolor="#16243A")
        axis = figure.add_axes((0.025, 0.105, 0.95, 0.79), projection=projection)
        axis.set_extent(self.extent, crs=projection)
        axis.set_facecolor("#A9C7EF")
        # ``stock_img`` ships inside Cartopy, so this quick-look renderer does
        # not pause to download Natural Earth shapefiles on a fresh machine.
        axis.stock_img()
        axis.gridlines(
            crs=projection,
            linewidth=0.35,
            color="#DDE6F0",
            alpha=0.28,
            linestyle=":",
            zorder=10,
        )

        empty = np.zeros((*self.grid.shape, 4), dtype=float)
        image_extent = (
            self.grid.longitude_deg[0],
            self.grid.longitude_deg[-1],
            self.grid.latitude_deg[0],
            self.grid.latitude_deg[-1],
        )
        real_envelope, giant_envelope = self._envelopes()
        axis.imshow(
            self._shadow_rgba(
                real_envelope,
                np.zeros_like(real_envelope, dtype=bool),
                REAL_BLUE,
                maximum_alpha=0.11,
            ),
            extent=image_extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=3,
        )
        axis.imshow(
            self._shadow_rgba(
                giant_envelope,
                np.zeros_like(giant_envelope, dtype=bool),
                GIANT_ORANGE,
                maximum_alpha=0.11,
            ),
            extent=image_extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=4,
        )
        night_image = axis.imshow(
            empty,
            extent=image_extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=5,
        )
        real_image = axis.imshow(
            empty,
            extent=image_extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=6,
        )
        giant_image = axis.imshow(
            empty,
            extent=image_extent,
            origin="lower",
            transform=projection,
            interpolation="bilinear",
            zorder=7,
        )
        (real_trail_line,) = axis.plot(
            [],
            [],
            color=REAL_BLUE,
            linewidth=1.5,
            zorder=12,
        )
        (giant_trail_line,) = axis.plot(
            [],
            [],
            color=GIANT_ORANGE,
            linewidth=1.5,
            zorder=12,
        )
        real_marker = axis.scatter(
            [],
            [],
            s=36,
            facecolor=REAL_BLUE,
            edgecolor="white",
            linewidth=0.8,
            transform=projection,
            zorder=13,
        )
        giant_marker = axis.scatter(
            [],
            [],
            s=44,
            marker="D",
            facecolor=GIANT_ORANGE,
            edgecolor="white",
            linewidth=0.8,
            transform=projection,
            zorder=13,
        )
        axis.scatter(
            [float(self.pair["best_common_longitude_deg"])],
            [float(self.pair["best_common_latitude_deg"])],
            marker="*",
            s=76,
            facecolor="white",
            edgecolor="#152238",
            linewidth=0.8,
            transform=projection,
            zorder=14,
        )
        legend = axis.legend(
            handles=[
                Patch(facecolor=REAL_BLUE, alpha=0.70, label="Real Moon · annular"),
                Patch(facecolor=GIANT_ORANGE, alpha=0.70, label="Giant moon · total"),
                Line2D(
                    (0,),
                    (0,),
                    marker="*",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor="#152238",
                    markersize=9,
                    label="Shared eclipse site",
                ),
            ],
            loc="lower left",
            ncol=3,
            frameon=True,
            facecolor="#16243A",
            edgecolor="#8595AA",
            framealpha=0.92,
            fontsize=8,
        )
        for label in legend.get_texts():
            label.set_color("white")
        title = figure.text(
            0.5,
            0.95,
            "The 17-hour two-moon eclipse · June 22–23, 2027",
            color="white",
            ha="center",
            fontsize=14,
        )
        timestamp = figure.text(
            0.5,
            0.052,
            "",
            color="#F2F6FA",
            ha="center",
            fontsize=10,
        )
        note = figure.text(
            0.5,
            0.018,
            "Faint color = full event reach · moving color = instantaneous solar obscuration",
            color="#B9C8D9",
            ha="center",
            fontsize=8.5,
        )

        def update(index: int):
            (
                time,
                daylight,
                real_obscuration,
                real_central,
                giant_obscuration,
                giant_central,
            ) = self._frame_fields(float(self.times[index]))
            night_rgba = np.zeros((*daylight.shape, 4), dtype=float)
            night_rgba[..., :3] = (0.015, 0.035, 0.075)
            night_rgba[..., 3] = 0.58 * np.clip(-daylight, 0.0, 1.0) ** 0.35
            night_image.set_data(night_rgba)
            real_image.set_data(self._shadow_rgba(real_obscuration, real_central, REAL_BLUE))
            giant_image.set_data(self._shadow_rgba(giant_obscuration, giant_central, GIANT_ORANGE))

            for track, line, marker in (
                (self.real_track, real_trail_line, real_marker),
                (self.giant_track, giant_trail_line, giant_marker),
            ):
                longitudes, latitudes = _break_dateline(
                    track[: index + 1, 0],
                    track[: index + 1, 1],
                )
                line.set_data(longitudes, latitudes)
                current = track[index]
                if np.all(np.isfinite(current)):
                    marker.set_offsets(current[None, :])
                else:
                    marker.set_offsets(np.empty((0, 2)))

            timestamp.set_text(time_iso_utc(time, places=0))
            return (
                night_image,
                real_image,
                giant_image,
                real_trail_line,
                giant_trail_line,
                real_marker,
                giant_marker,
                title,
                timestamp,
                note,
            )

        movie = mpl_animation.FuncAnimation(
            figure,
            update,
            frames=len(self.times),
            interval=1_000.0 / fps,
            blit=False,
        )
        if path.suffix.lower() == ".gif":
            writer = mpl_animation.PillowWriter(fps=fps)
        elif path.suffix.lower() == ".mp4":
            writer = mpl_animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=3_200,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        else:
            raise ValueError("animation output must end in .mp4 or .gif")
        movie.save(path, writer=writer, dpi=dpi)
        plt.close(figure)
        return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="outputs/bound_binary_giant/coupled/coupled_eclipses.json",
    )
    parser.add_argument("--config", default="config/bound_binary_giant.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output",
        default="outputs/bound_binary_giant/animations/20270622_world_map.mp4",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=-1,
        help="zero-based pair index; negative indices count from the end",
    )
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--lead-minutes", type=float, default=10.0)
    parser.add_argument("--trail-minutes", type=float, default=10.0)
    parser.add_argument("--grid-step-deg", type=float, default=1.5)
    parser.add_argument("--trajectory-step-seconds", type=float, default=600.0)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args(argv)

    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.grid_step_deg <= 0.0:
        parser.error("--grid-step-deg must be positive")
    if args.lead_minutes < 0.0 or args.trail_minutes < 0.0:
        parser.error("--lead-minutes and --trail-minutes must be non-negative")

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    pairs = result["within_12h_pairs"]
    if not pairs:
        raise ValueError("the coupled result contains no eclipse pairs within 12 hours")
    try:
        pair = pairs[args.pair_index]
    except IndexError as exc:
        raise ValueError(
            f"pair_index {args.pair_index} is outside the {len(pairs)} available pairs"
        ) from exc
    start_utc, end_utc = animation_window_utc(
        result,
        pair,
        lead_minutes=args.lead_minutes,
        trail_minutes=args.trail_minutes,
    )
    context = load_ephemeris(args.ephemeris_cache)
    start_tt = float(context.time_utc(start_utc).tt) - args.lead_minutes * 60.0 / SECONDS_PER_DAY
    end_tt = float(context.time_utc(end_utc).tt) + args.trail_minutes * 60.0 / SECONDS_PER_DAY
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    elements = elements_from_config(config)
    architecture = architecture_from_config(config)
    ephemeris = CoupledEphemeris(
        context,
        elements,
        time_iso_utc(context.tt_jd(end_tt), places=1),
        sample_step_seconds=args.trajectory_step_seconds,
        binary_architecture=architecture,
    )
    times = np.linspace(start_tt, end_tt, args.frames)
    output = CoupledEclipseMapAnimator(
        ephemeris,
        times,
        pair,
        grid_step_deg=args.grid_step_deg,
    ).render(args.output, fps=args.fps, dpi=args.dpi)
    manifest = {
        "schema_version": "1.0",
        "output": str(output),
        "projection": "Plate Carree (equirectangular)",
        "extent_deg": list(WORLD_EXTENT),
        "pair_index": args.pair_index,
        "real_maximum_utc": pair["real_maximum_utc"],
        "giant_maximum_utc": pair["second_maximum_utc"],
        "start_utc": time_iso_utc(context.tt_jd(start_tt), places=1),
        "end_utc": time_iso_utc(context.tt_jd(end_tt), places=1),
        "frames": args.frames,
        "fps": args.fps,
        "screen_duration_s": args.frames / args.fps,
        "grid_step_deg": args.grid_step_deg,
        "trajectory_step_seconds": args.trajectory_step_seconds,
        "footprints": "instantaneous topocentric disk overlap on a rotating WGS84 grid",
        "faint_envelopes": "maximum obscuration across 48 evenly spaced animation samples",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Coupled two-moon world-map animation: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
