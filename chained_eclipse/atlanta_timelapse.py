"""Render Atlanta's all-day two-moon solar-eclipse geometry.

The Sun remains fixed at the origin while both apparent lunar disks move in
celestial east/north coordinates.  A wide field reveals the looping tracks; a
solar-disk close-up shows the actual overlap at the same angular scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.animation as mpl_animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import yaml

from .coupled_eclipse import (
    CoupledEphemeris,
    coupled_apparent_geometry,
    coupled_sky_plane_disks,
    solve_coupled_local,
)
from .ephemeris import load_ephemeris, time_iso_utc
from .moon_architecture import architecture_from_config, elements_from_config


ATLANTA_LATITUDE_DEG = 33.7488
ATLANTA_LONGITUDE_DEG = -84.3880
ATLANTA_TIMEZONE = ZoneInfo("America/New_York")
REAL_BLUE = "#2F9BFF"
GIANT_ORANGE = "#FF7138"
SUN_GOLD = "#FFD65A"
BACKGROUND = "#07111D"
PANEL = "#101D2B"
FOREGROUND = "#F4F7FA"
MUTED = "#A9B9C9"
GRID = "#40536A"


@dataclass(frozen=True, slots=True)
class ObserverFrame:
    """One solar-centered observer snapshot."""

    tt_jd: float
    local_label: str
    disks: dict[str, dict[str, float]]
    real_obscuration: float
    giant_obscuration: float
    combined_obscuration: float


def combined_obscuration_fraction(
    disks: dict[str, dict[str, float]],
    *,
    quad_segs: int = 96,
) -> float:
    """Return the union of both lunar silhouettes inside the solar disk."""

    if quad_segs < 8:
        raise ValueError("quad_segs must be at least 8")
    sun_data = disks["sun"]
    sun = Point(sun_data["east_deg"], sun_data["north_deg"]).buffer(
        sun_data["angular_radius_deg"],
        quad_segs=quad_segs,
    )
    silhouettes = [
        Point(disks[name]["east_deg"], disks[name]["north_deg"]).buffer(
            disks[name]["angular_radius_deg"],
            quad_segs=quad_segs,
        )
        for name in ("real_moon", "second_moon")
    ]
    covered = unary_union(silhouettes).intersection(sun)
    return float(np.clip(covered.area / sun.area, 0.0, 1.0))


def _local_datetime(utc_iso: str) -> datetime:
    return datetime.fromisoformat(utc_iso.replace("Z", "+00:00")).astimezone(ATLANTA_TIMEZONE)


def _local_label(context, tt_jd: float) -> str:
    instant = _local_datetime(time_iso_utc(context.tt_jd(tt_jd), places=3))
    return instant.strftime("%-I:%M %p EDT")


def _frame_data(
    ephemeris: CoupledEphemeris,
    times_tt_jd: np.ndarray,
) -> list[ObserverFrame]:
    frames: list[ObserverFrame] = []
    for tt_jd in np.asarray(times_tt_jd, dtype=float):
        disks = coupled_sky_plane_disks(
            ephemeris,
            float(tt_jd),
            ATLANTA_LATITUDE_DEG,
            ATLANTA_LONGITUDE_DEG,
        )
        real = coupled_apparent_geometry(
            ephemeris,
            "real_moon",
            float(tt_jd),
            ATLANTA_LATITUDE_DEG,
            ATLANTA_LONGITUDE_DEG,
        )
        giant = coupled_apparent_geometry(
            ephemeris,
            "second_moon",
            float(tt_jd),
            ATLANTA_LATITUDE_DEG,
            ATLANTA_LONGITUDE_DEG,
        )
        frames.append(
            ObserverFrame(
                tt_jd=float(tt_jd),
                local_label=_local_label(ephemeris.context, float(tt_jd)),
                disks=disks,
                real_obscuration=real.obscuration,
                giant_obscuration=giant.obscuration,
                combined_obscuration=combined_obscuration_fraction(disks),
            )
        )
    return frames


class AtlantaEclipseTimelapse:
    """Solar-centered two-scale renderer for Atlanta's eclipse day."""

    def __init__(
        self,
        frames: list[ObserverFrame],
        *,
        context,
        real_interval_tt: tuple[float, float],
        giant_interval_tt: tuple[float, float],
    ) -> None:
        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        if not all(
            frames[index].tt_jd < frames[index + 1].tt_jd for index in range(len(frames) - 1)
        ):
            raise ValueError("frames must be strictly increasing")
        self.frames = frames
        self.context = context
        self.real_interval_tt = real_interval_tt
        self.giant_interval_tt = giant_interval_tt
        self.start_tt = frames[0].tt_jd
        self.end_tt = frames[-1].tt_jd
        self.real_path = np.asarray(
            [
                (
                    frame.disks["real_moon"]["east_deg"],
                    frame.disks["real_moon"]["north_deg"],
                )
                for frame in frames
            ]
        )
        self.giant_path = np.asarray(
            [
                (
                    frame.disks["second_moon"]["east_deg"],
                    frame.disks["second_moon"]["north_deg"],
                )
                for frame in frames
            ]
        )

    def _elapsed_hours(self, tt_jd: float) -> float:
        return (tt_jd - self.start_tt) * 24.0

    def render(
        self,
        output_path: str | Path,
        *,
        fps: int = 10,
        dpi: int = 100,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(13.6, 7.2), facecolor=BACKGROUND)
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(1.18, 1.0),
            height_ratios=(12.0, 2.0),
            left=0.065,
            right=0.965,
            bottom=0.095,
            top=0.88,
            hspace=0.18,
            wspace=0.16,
        )
        wide = figure.add_subplot(grid[0, 0])
        close = figure.add_subplot(grid[0, 1])
        timeline = figure.add_subplot(grid[1, :])
        for axis in (wide, close):
            axis.set_facecolor(PANEL)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.grid(color=GRID, linewidth=0.5, alpha=0.45, linestyle=":")
            for spine in axis.spines.values():
                spine.set_edgecolor(GRID)

        wide_limit = max(
            1.8,
            min(
                3.6,
                1.08
                * max(
                    np.max(np.abs(self.real_path)),
                    np.max(np.abs(self.giant_path)),
                ),
            ),
        )
        wide.set_xlim(-wide_limit, wide_limit)
        wide.set_ylim(-wide_limit, wide_limit)
        wide.set_xlabel("Celestial east (deg)", color=MUTED)
        wide.set_ylabel("Celestial north (deg)", color=MUTED)
        wide.set_title("Full solar-centered field", color=FOREGROUND, pad=9)
        wide.axhline(0.0, color=GRID, linewidth=0.6)
        wide.axvline(0.0, color=GRID, linewidth=0.6)
        wide.plot(
            self.real_path[:, 0],
            self.real_path[:, 1],
            color=REAL_BLUE,
            linewidth=1.0,
            alpha=0.28,
            linestyle=":",
        )
        wide.plot(
            self.giant_path[:, 0],
            self.giant_path[:, 1],
            color=GIANT_ORANGE,
            linewidth=1.0,
            alpha=0.28,
            linestyle=":",
        )
        (real_elapsed,) = wide.plot([], [], color=REAL_BLUE, linewidth=2.1)
        (giant_elapsed,) = wide.plot([], [], color=GIANT_ORANGE, linewidth=2.1)
        sun_wide = Circle((0.0, 0.0), 0.262, facecolor=SUN_GOLD, edgecolor=FOREGROUND)
        real_wide = Circle(
            (0.0, 0.0),
            0.223,
            facecolor="none",
            edgecolor=REAL_BLUE,
            linewidth=2.0,
        )
        giant_wide = Circle(
            (0.0, 0.0),
            0.296,
            facecolor="none",
            edgecolor=GIANT_ORANGE,
            linewidth=2.0,
            linestyle="--",
        )
        wide.add_patch(sun_wide)
        wide.add_patch(real_wide)
        wide.add_patch(giant_wide)

        close_limit = 0.72
        close.set_xlim(-close_limit, close_limit)
        close.set_ylim(-close_limit, close_limit)
        close.set_xlabel("Celestial east (deg)", color=MUTED)
        close.set_ylabel("Celestial north (deg)", color=MUTED)
        close.set_title("Solar-filter close-up · exact angular scale", color=FOREGROUND, pad=9)
        close.axhline(0.0, color=GRID, linewidth=0.6)
        close.axvline(0.0, color=GRID, linewidth=0.6)
        sun_close = Circle((0.0, 0.0), 0.262, facecolor=SUN_GOLD, edgecolor=FOREGROUND)
        close.add_patch(sun_close)
        real_fill = Circle((0.0, 0.0), 0.223, facecolor=BACKGROUND, edgecolor="none")
        giant_fill = Circle((0.0, 0.0), 0.296, facecolor=BACKGROUND, edgecolor="none")
        real_outline = Circle(
            (0.0, 0.0),
            0.223,
            facecolor="none",
            edgecolor=REAL_BLUE,
            linewidth=2.4,
        )
        giant_outline = Circle(
            (0.0, 0.0),
            0.296,
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
            bbox_to_anchor=(0.5, 0.925),
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

        real_start, real_width = interval_hours(self.real_interval_tt)
        giant_start, giant_width = interval_hours(self.giant_interval_tt)
        timeline.broken_barh(
            [(real_start, real_width)],
            (0.62, 0.20),
            facecolors=REAL_BLUE,
            alpha=0.78,
        )
        timeline.broken_barh(
            [(giant_start, giant_width)],
            (0.22, 0.20),
            facecolors=GIANT_ORANGE,
            alpha=0.82,
        )
        timeline.text(
            0.0,
            0.72,
            "Real Moon partial phase",
            color=FOREGROUND,
            va="center",
            ha="left",
            fontsize=8,
        )
        timeline.text(
            0.0,
            0.32,
            "Giant-moon partial phase",
            color=FOREGROUND,
            va="center",
            ha="left",
            fontsize=8,
        )
        tick_hours = np.linspace(0.0, duration_hours, 4)
        tick_labels = []
        for hours in tick_hours:
            tt_jd = self.start_tt + hours / 24.0
            label = _local_datetime(time_iso_utc(self.context.tt_jd(tt_jd)))
            tick_labels.append(label.strftime("%-I:%M %p"))
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
            "Sun fixed at center · celestial north up · lunar disks and separations to scale",
            color=MUTED,
            ha="center",
            fontsize=8.5,
        )

        def move_patch(patch: Circle, disk: dict[str, float]) -> None:
            patch.center = (disk["east_deg"], disk["north_deg"])
            patch.set_radius(disk["angular_radius_deg"])

        def update(index: int):
            frame = self.frames[index]
            real_disk = frame.disks["real_moon"]
            giant_disk = frame.disks["second_moon"]
            sun_disk = frame.disks["sun"]
            real_elapsed.set_data(
                self.real_path[: index + 1, 0],
                self.real_path[: index + 1, 1],
            )
            giant_elapsed.set_data(
                self.giant_path[: index + 1, 0],
                self.giant_path[: index + 1, 1],
            )
            sun_wide.set_radius(sun_disk["angular_radius_deg"])
            sun_close.set_radius(sun_disk["angular_radius_deg"])
            for patch in (real_wide, real_fill, real_outline):
                move_patch(patch, real_disk)
            for patch in (giant_wide, giant_fill, giant_outline):
                move_patch(patch, giant_disk)
            current_line.set_xdata([self._elapsed_hours(frame.tt_jd)] * 2)
            title.set_text(f"Atlanta eclipse geometry · {frame.local_label}")
            status.set_text(
                f"Combined solar obscuration {100.0 * frame.combined_obscuration:.1f}%   ·   "
                f"real Moon {100.0 * frame.real_obscuration:.1f}%   ·   "
                f"giant moon {100.0 * frame.giant_obscuration:.1f}%"
            )
            return (
                real_elapsed,
                giant_elapsed,
                sun_wide,
                sun_close,
                real_wide,
                giant_wide,
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


def build_atlanta_timelapse(
    *,
    results_path: str | Path,
    config_path: str | Path,
    ephemeris_cache: str | Path,
    output_path: str | Path,
    pair_index: int = 5,
    frames: int = 121,
    fps: int = 10,
    dpi: int = 100,
    trajectory_step_seconds: float = 600.0,
) -> dict[str, object]:
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    pair = result["within_12h_pairs"][pair_index]
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
    real_reference = float(context.time_utc(pair["real_maximum_utc"]).tt)
    giant_reference = float(context.time_utc(pair["second_maximum_utc"]).tt)
    real = solve_coupled_local(
        ephemeris,
        "real_moon",
        real_reference,
        ATLANTA_LATITUDE_DEG,
        ATLANTA_LONGITUDE_DEG,
        half_window_hours=10.0,
        step_seconds=30.0,
    )
    giant = solve_coupled_local(
        ephemeris,
        "second_moon",
        giant_reference,
        ATLANTA_LATITUDE_DEG,
        ATLANTA_LONGITUDE_DEG,
        half_window_hours=6.0,
        step_seconds=30.0,
    )
    if real.c1_utc is None or real.c4_utc is None:
        raise ValueError("Atlanta has no bounded real-Moon partial phase")
    if giant.c1_utc is None or giant.c4_utc is None:
        raise ValueError("Atlanta has no bounded giant-moon partial phase")
    start_tt = float(context.time_utc(real.c1_utc).tt)
    end_tt = float(context.time_utc(real.c4_utc).tt)
    times = np.linspace(start_tt, end_tt, frames)
    observer_frames = _frame_data(ephemeris, times)
    real_interval = (start_tt, end_tt)
    giant_interval = (
        float(context.time_utc(giant.c1_utc).tt),
        float(context.time_utc(giant.c4_utc).tt),
    )
    animator = AtlantaEclipseTimelapse(
        observer_frames,
        context=context,
        real_interval_tt=real_interval,
        giant_interval_tt=giant_interval,
    )
    output = animator.render(output_path, fps=fps, dpi=dpi)
    manifest = {
        "schema_version": "1.0",
        "output": str(output),
        "location": {
            "name": "Atlanta, Georgia",
            "latitude_deg": ATLANTA_LATITUDE_DEG,
            "longitude_deg": ATLANTA_LONGITUDE_DEG,
        },
        "start_utc": real.c1_utc,
        "end_utc": real.c4_utc,
        "giant_start_utc": giant.c1_utc,
        "giant_end_utc": giant.c4_utc,
        "frames": frames,
        "fps": fps,
        "screen_duration_s": frames / fps,
        "coordinate_system": "solar-centered celestial east/north gnomonic projection",
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
        default="outputs/bound_binary_giant/animations/20270622_atlanta_geometry.mp4",
    )
    parser.add_argument("--pair-index", type=int, default=5)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--trajectory-step-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    manifest = build_atlanta_timelapse(
        results_path=args.results,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_path=args.output,
        pair_index=args.pair_index,
        frames=args.frames,
        fps=args.fps,
        dpi=args.dpi,
        trajectory_step_seconds=args.trajectory_step_seconds,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
