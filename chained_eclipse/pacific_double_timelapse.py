"""Render the August 2026 Pacific double-eclipse morning.

The animation combines an observer's horizon-aware solar path with an exact
solar-filter close-up.  Central phases receive extra frames so the giant
Moon's sunrise totality and the real Moon's noon annularity remain visible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import matplotlib.animation as mpl_animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .atlanta_timelapse import combined_obscuration_fraction
from .coupled_eclipse import (
    CoupledEphemeris,
    coupled_apparent_geometry,
    coupled_sky_plane_disks,
    coupled_solar_altaz,
)
from .ephemeris import load_ephemeris, time_iso_utc
from .moon_architecture import architecture_from_config, elements_from_config


PACIFIC_TIMEZONE = timezone(timedelta(hours=12), name="UTC+12")
REAL_BLUE = "#2F9BFF"
GIANT_ORANGE = "#FF7138"
SUN_GOLD = "#FFD65A"
BACKGROUND = "#07111D"
PANEL = "#101D2B"
FOREGROUND = "#F4F7FA"
MUTED = "#A9B9C9"
GRID = "#40536A"


@dataclass(frozen=True, slots=True)
class PacificFrame:
    """One observer snapshot during the double-eclipse morning."""

    tt_jd: float
    local_label: str
    disks: dict[str, dict[str, float]]
    solar_altitude_deg: float
    solar_azimuth_deg: float
    real_obscuration: float
    giant_obscuration: float
    combined_obscuration: float


def nonuniform_contact_times(
    boundaries_tt: list[float],
    segment_frame_counts: list[int],
) -> np.ndarray:
    """Sample consecutive contact intervals with independently chosen density."""

    if len(boundaries_tt) != len(segment_frame_counts) + 1:
        raise ValueError("one more boundary than segment count is required")
    if any(count < 1 for count in segment_frame_counts):
        raise ValueError("every segment needs at least one frame")
    if not all(
        boundaries_tt[index] < boundaries_tt[index + 1]
        for index in range(len(boundaries_tt) - 1)
    ):
        raise ValueError("contact boundaries must be strictly increasing")
    pieces = [
        np.linspace(start, end, count, endpoint=False)
        for start, end, count in zip(
            boundaries_tt[:-1],
            boundaries_tt[1:],
            segment_frame_counts,
            strict=True,
        )
    ]
    return np.concatenate((*pieces, np.asarray([boundaries_tt[-1]])))


def _local_datetime(utc_iso: str) -> datetime:
    return datetime.fromisoformat(utc_iso.replace("Z", "+00:00")).astimezone(
        PACIFIC_TIMEZONE
    )


def _local_label(context, tt_jd: float) -> str:
    instant = _local_datetime(time_iso_utc(context.tt_jd(tt_jd), places=3))
    return instant.strftime("%-I:%M %p")


def _frame_data(
    ephemeris: CoupledEphemeris,
    times_tt_jd: np.ndarray,
    *,
    latitude_deg: float,
    longitude_deg: float,
) -> list[PacificFrame]:
    frames: list[PacificFrame] = []
    for tt_jd in np.asarray(times_tt_jd, dtype=float):
        disks = coupled_sky_plane_disks(
            ephemeris,
            float(tt_jd),
            latitude_deg,
            longitude_deg,
        )
        real = coupled_apparent_geometry(
            ephemeris,
            "real_moon",
            float(tt_jd),
            latitude_deg,
            longitude_deg,
        )
        giant = coupled_apparent_geometry(
            ephemeris,
            "second_moon",
            float(tt_jd),
            latitude_deg,
            longitude_deg,
        )
        altitude, azimuth = coupled_solar_altaz(
            ephemeris,
            float(tt_jd),
            latitude_deg,
            longitude_deg,
        )
        frames.append(
            PacificFrame(
                tt_jd=float(tt_jd),
                local_label=_local_label(ephemeris.context, float(tt_jd)),
                disks=disks,
                solar_altitude_deg=altitude,
                solar_azimuth_deg=azimuth,
                real_obscuration=real.obscuration,
                giant_obscuration=giant.obscuration,
                combined_obscuration=combined_obscuration_fraction(disks),
            )
        )
    return frames


class PacificDoubleTimelapse:
    """Horizon path and exact disk renderer for the double eclipse."""

    def __init__(
        self,
        frames: list[PacificFrame],
        *,
        context,
        giant_partial_tt: tuple[float, float],
        giant_total_tt: tuple[float, float],
        real_partial_tt: tuple[float, float],
        real_annular_tt: tuple[float, float],
    ) -> None:
        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        if not all(
            frames[index].tt_jd < frames[index + 1].tt_jd
            for index in range(len(frames) - 1)
        ):
            raise ValueError("frames must be strictly increasing")
        self.frames = frames
        self.context = context
        self.giant_partial_tt = giant_partial_tt
        self.giant_total_tt = giant_total_tt
        self.real_partial_tt = real_partial_tt
        self.real_annular_tt = real_annular_tt
        self.start_tt = frames[0].tt_jd
        self.end_tt = frames[-1].tt_jd
        self.azimuth_path = np.asarray([frame.solar_azimuth_deg for frame in frames])
        self.altitude_path = np.asarray([frame.solar_altitude_deg for frame in frames])

    def _elapsed_hours(self, tt_jd: float) -> float:
        return (tt_jd - self.start_tt) * 24.0

    def render(
        self,
        output_path: str | Path,
        *,
        fps: int = 12,
        dpi: int = 100,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(13.6, 7.2), facecolor=BACKGROUND)
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(1.12, 1.0),
            height_ratios=(12.0, 2.15),
            left=0.065,
            right=0.965,
            bottom=0.105,
            top=0.805,
            hspace=0.18,
            wspace=0.16,
        )
        sky = figure.add_subplot(grid[0, 0])
        close = figure.add_subplot(grid[0, 1])
        timeline = figure.add_subplot(grid[1, :])
        for axis in (sky, close):
            axis.set_facecolor(PANEL)
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.grid(color=GRID, linewidth=0.5, alpha=0.45, linestyle=":")
            for spine in axis.spines.values():
                spine.set_edgecolor(GRID)

        azimuth_padding = 8.0
        sky.set_xlim(
            float(np.min(self.azimuth_path)) - azimuth_padding,
            float(np.max(self.azimuth_path)) + azimuth_padding,
        )
        sky.set_ylim(
            min(-8.0, float(np.min(self.altitude_path)) - 3.0),
            min(90.0, float(np.max(self.altitude_path)) + 6.0),
        )
        sky.axhspan(sky.get_ylim()[0], 0.0, color=BACKGROUND, alpha=0.82)
        sky.axhline(0.0, color=FOREGROUND, linewidth=1.2)
        sky.plot(
            self.azimuth_path,
            self.altitude_path,
            color=SUN_GOLD,
            linewidth=1.2,
            alpha=0.35,
            linestyle=":",
        )
        (elapsed_path,) = sky.plot([], [], color=SUN_GOLD, linewidth=2.4)
        (sun_marker,) = sky.plot(
            [],
            [],
            marker="o",
            markersize=13,
            markerfacecolor=SUN_GOLD,
            markeredgecolor=FOREGROUND,
            markeredgewidth=1.4,
            linestyle="none",
        )
        sky.set_xlabel("Solar azimuth (deg from north)", color=MUTED)
        sky.set_ylabel("Solar altitude (deg)", color=MUTED)
        sky.set_title("Sun’s path from the eastern horizon to noon", color=FOREGROUND, pad=9)
        horizon_label = sky.text(
            0.02,
            0.03,
            "below horizon",
            transform=sky.transAxes,
            color=MUTED,
            fontsize=8,
            va="bottom",
        )
        altitude_label = sky.text(
            0.0,
            0.0,
            "",
            color=FOREGROUND,
            fontsize=8.5,
            ha="left",
            va="bottom",
        )

        close.set_aspect("equal", adjustable="box")
        close_limit = 0.72
        close.set_xlim(-close_limit, close_limit)
        close.set_ylim(-close_limit, close_limit)
        close.set_xlabel("Celestial east (deg)", color=MUTED)
        close.set_ylabel("Celestial north (deg)", color=MUTED)
        close.set_title("Solar-filter view · exact angular scale", color=FOREGROUND, pad=9)
        close.axhline(0.0, color=GRID, linewidth=0.6)
        close.axvline(0.0, color=GRID, linewidth=0.6)
        sun_close = Circle((0.0, 0.0), 0.262, facecolor=SUN_GOLD, edgecolor=FOREGROUND)
        close.add_patch(sun_close)
        real_fill = Circle((0.0, 0.0), 0.19, facecolor=BACKGROUND, edgecolor="none")
        giant_fill = Circle((0.0, 0.0), 0.30, facecolor=BACKGROUND, edgecolor="none")
        real_outline = Circle(
            (0.0, 0.0),
            0.19,
            facecolor="none",
            edgecolor=REAL_BLUE,
            linewidth=2.4,
        )
        giant_outline = Circle(
            (0.0, 0.0),
            0.30,
            facecolor="none",
            edgecolor=GIANT_ORANGE,
            linewidth=2.4,
            linestyle="--",
        )
        real_fill.set_clip_path(sun_close)
        giant_fill.set_clip_path(sun_close)
        for patch in (giant_fill, real_fill, giant_outline, real_outline):
            close.add_patch(patch)

        legend = figure.legend(
            handles=[
                Line2D((0,), (0,), color=REAL_BLUE, linewidth=2.5, label="Real Moon"),
                Line2D(
                    (0,),
                    (0,),
                    color=GIANT_ORANGE,
                    linewidth=2.5,
                    linestyle="--",
                    label="Giant moon",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=2,
            frameon=False,
        )
        for text in legend.get_texts():
            text.set_color(FOREGROUND)

        duration_hours = self._elapsed_hours(self.end_tt)
        timeline.set_facecolor(BACKGROUND)
        timeline.set_xlim(0.0, duration_hours)
        timeline.set_ylim(0.0, 1.0)
        timeline.spines[["top", "right", "left"]].set_visible(False)
        timeline.spines["bottom"].set_color(GRID)
        timeline.tick_params(axis="x", colors=MUTED, labelsize=8)
        timeline.tick_params(axis="y", left=False, labelleft=False)

        def interval_hours(interval: tuple[float, float]) -> tuple[float, float]:
            start = self._elapsed_hours(interval[0])
            return start, self._elapsed_hours(interval[1]) - start

        giant_start, giant_width = interval_hours(self.giant_partial_tt)
        giant_total_start, giant_total_width = interval_hours(self.giant_total_tt)
        real_start, real_width = interval_hours(self.real_partial_tt)
        real_annular_start, real_annular_width = interval_hours(self.real_annular_tt)
        timeline.broken_barh(
            [(giant_start, giant_width)],
            (0.60, 0.22),
            facecolors=GIANT_ORANGE,
            alpha=0.36,
        )
        timeline.broken_barh(
            [(giant_total_start, giant_total_width)],
            (0.60, 0.22),
            facecolors=GIANT_ORANGE,
            alpha=1.0,
        )
        timeline.broken_barh(
            [(real_start, real_width)],
            (0.20, 0.22),
            facecolors=REAL_BLUE,
            alpha=0.36,
        )
        timeline.broken_barh(
            [(real_annular_start, real_annular_width)],
            (0.20, 0.22),
            facecolors=REAL_BLUE,
            alpha=1.0,
        )
        timeline.text(
            0.0,
            0.71,
            "Giant moon · total at sunrise",
            color=FOREGROUND,
            va="center",
            ha="left",
            fontsize=8,
        )
        timeline.text(
            0.0,
            0.31,
            "Real Moon · annular near noon",
            color=FOREGROUND,
            va="center",
            ha="left",
            fontsize=8,
        )
        tick_hours = np.linspace(0.0, duration_hours, 5)
        tick_labels = []
        for hours in tick_hours:
            tt_jd = self.start_tt + hours / 24.0
            local = _local_datetime(time_iso_utc(self.context.tt_jd(tt_jd)))
            tick_labels.append(local.strftime("%-I:%M %p"))
        timeline.set_xticks(tick_hours, tick_labels)
        current_line = timeline.axvline(0.0, color=FOREGROUND, linewidth=1.4)

        title = figure.text(
            0.5,
            0.967,
            "",
            color=FOREGROUND,
            ha="center",
            va="top",
            fontsize=15,
        )
        status = figure.text(
            0.5,
            0.932,
            "",
            color=MUTED,
            ha="center",
            va="top",
            fontsize=9.5,
        )
        figure.text(
            0.5,
            0.025,
            "18.79° N, 162.95° E · UTC+12 · central phases slowed · disks to scale",
            color=MUTED,
            ha="center",
            fontsize=8.5,
        )

        def move_patch(patch: Circle, disk: dict[str, float]) -> None:
            patch.center = (disk["east_deg"], disk["north_deg"])
            patch.set_radius(disk["angular_radius_deg"])

        def phase_label(frame: PacificFrame) -> str:
            if frame.solar_altitude_deg < -0.833:
                return "Sun below horizon"
            if frame.combined_obscuration >= 0.9995:
                return "TOTALITY AT SUNRISE"
            real_disk = frame.disks["real_moon"]
            sun_disk = frame.disks["sun"]
            center_distance = np.hypot(
                real_disk["east_deg"] - sun_disk["east_deg"],
                real_disk["north_deg"] - sun_disk["north_deg"],
            )
            if (
                center_distance + real_disk["angular_radius_deg"]
                <= sun_disk["angular_radius_deg"]
            ):
                return "ANNULAR PHASE"
            if frame.combined_obscuration > 0.0:
                return "Partial eclipse"
            return "Clear Sun"

        def update(index: int):
            frame = self.frames[index]
            real_disk = frame.disks["real_moon"]
            giant_disk = frame.disks["second_moon"]
            sun_disk = frame.disks["sun"]
            elapsed_path.set_data(
                self.azimuth_path[: index + 1],
                self.altitude_path[: index + 1],
            )
            sun_marker.set_data([frame.solar_azimuth_deg], [frame.solar_altitude_deg])
            altitude_label.set_position(
                (frame.solar_azimuth_deg + 2.0, frame.solar_altitude_deg + 1.5)
            )
            altitude_label.set_text(f"{frame.solar_altitude_deg:.1f}° altitude")
            sun_close.set_radius(sun_disk["angular_radius_deg"])
            for patch in (real_fill, real_outline):
                move_patch(patch, real_disk)
            for patch in (giant_fill, giant_outline):
                move_patch(patch, giant_disk)
            current_line.set_xdata([self._elapsed_hours(frame.tt_jd)] * 2)
            title.set_text(f"Pacific double eclipse · {frame.local_label} UTC+12")
            status.set_text(
                f"{phase_label(frame)}   ·   "
                f"combined obscuration {100.0 * frame.combined_obscuration:.1f}%   ·   "
                f"real {100.0 * frame.real_obscuration:.1f}%   ·   "
                f"giant {100.0 * frame.giant_obscuration:.1f}%"
            )
            visible = frame.solar_altitude_deg >= -0.833
            sun_close.set_alpha(1.0 if visible else 0.20)
            for patch in (real_fill, giant_fill, real_outline, giant_outline):
                patch.set_alpha(1.0 if visible else 0.20)
            return (
                elapsed_path,
                sun_marker,
                altitude_label,
                horizon_label,
                sun_close,
                real_fill,
                giant_fill,
                real_outline,
                giant_outline,
                current_line,
                title,
                status,
            )

        movie = mpl_animation.FuncAnimation(
            figure,
            update,
            frames=len(self.frames),
            interval=1_000.0 / fps,
            blit=False,
        )
        if path.suffix.lower() == ".gif":
            writer = mpl_animation.PillowWriter(fps=fps)
        elif path.suffix.lower() == ".mp4":
            writer = mpl_animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=4_000,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        else:
            raise ValueError("animation output must end in .mp4 or .gif")
        movie.save(path, writer=writer, dpi=dpi)
        plt.close(figure)
        return path.resolve()


def build_pacific_double_timelapse(
    *,
    results_path: str | Path,
    config_path: str | Path,
    ephemeris_cache: str | Path,
    output_path: str | Path,
    pair_index: int = 0,
    fps: int = 12,
    dpi: int = 100,
    trajectory_step_seconds: float = 600.0,
) -> dict[str, object]:
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    pair = result["within_12h_pairs"][pair_index]
    latitude = float(pair["best_common_latitude_deg"])
    longitude = float(pair["best_common_longitude_deg"])
    real = pair["real_local"]
    giant = pair["second_local"]
    contact_values = (
        giant["c1_utc"],
        giant["c2_utc"],
        giant["c3_utc"],
        giant["c4_utc"],
        real["c1_utc"],
        real["c2_utc"],
        real["c3_utc"],
        real["c4_utc"],
    )
    if any(value is None for value in contact_values):
        raise ValueError("the selected pair must have all eight local contacts")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    context = load_ephemeris(ephemeris_cache)
    elements = elements_from_config(config)
    ephemeris = CoupledEphemeris(
        context,
        elements,
        result["end_utc"],
        sample_step_seconds=trajectory_step_seconds,
        binary_architecture=architecture_from_config(config),
    )
    contacts = [float(context.time_utc(value).tt) for value in contact_values]
    times = nonuniform_contact_times(
        contacts,
        [18, 30, 28, 12, 22, 30, 21],
    )
    observer_frames = _frame_data(
        ephemeris,
        times,
        latitude_deg=latitude,
        longitude_deg=longitude,
    )
    animator = PacificDoubleTimelapse(
        observer_frames,
        context=context,
        giant_partial_tt=(contacts[0], contacts[3]),
        giant_total_tt=(contacts[1], contacts[2]),
        real_partial_tt=(contacts[4], contacts[7]),
        real_annular_tt=(contacts[5], contacts[6]),
    )
    output = animator.render(output_path, fps=fps, dpi=dpi)
    manifest = {
        "schema_version": "1.0",
        "output": str(output),
        "location": {
            "name": "western Pacific common site",
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "display_timezone": "UTC+12",
        },
        "start_utc": contact_values[0],
        "end_utc": contact_values[-1],
        "giant_totality": {
            "c2_utc": contact_values[1],
            "maximum_utc": giant["maximum_utc"],
            "c3_utc": contact_values[2],
            "duration_s": giant["central_duration_s"],
        },
        "real_annularity": {
            "c2_utc": contact_values[5],
            "maximum_utc": real["maximum_utc"],
            "c3_utc": contact_values[6],
            "duration_s": real["central_duration_s"],
        },
        "frames": len(times),
        "fps": fps,
        "screen_duration_s": len(times) / fps,
        "time_sampling": "nonuniform; central phases slowed",
        "disk_geometry": "topocentric light-time-corrected apparent angular disks",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
        default="outputs/bound_binary_giant/animations/20260821_pacific_double.mp4",
    )
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--trajectory-step-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    manifest = build_pacific_double_timelapse(
        results_path=args.results,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_path=args.output,
        pair_index=args.pair_index,
        fps=args.fps,
        dpi=args.dpi,
        trajectory_step_seconds=args.trajectory_step_seconds,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
