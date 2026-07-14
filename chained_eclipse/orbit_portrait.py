"""Render the two lunar trajectories from the coupled four-body simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import yaml

from .constants import OBLIQUITY_J2000_DEG, WGS84_A_KM
from .coupled_eclipse import CoupledEphemeris
from .ephemeris import load_ephemeris
from .models import OrbitalElements


INK = "#102039"
MUTED = "#68778F"
GRID = "#DCE4EC"
REAL_BLUE = "#24649A"
SECOND_ORANGE = "#D47424"
EARTH_BLUE = "#2474A8"
EARTH_EDGE = "#163C5B"


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def build_orbit_portrait(
    *,
    config_path: str | Path = "config/optimized_system.yaml",
    output_path: str | Path = "outputs/coupled/orbits/two_moon_orbits.png",
    days: float = 31.0,
    step_minutes: float = 20.0,
) -> Path:
    context = load_ephemeris("data/ephemeris")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    end_tt = float(context.time_utc(elements.epoch_utc).tt) + days
    end_utc = context.tt_jd(end_tt).utc_strftime("%Y-%m-%dT%H:%M:%SZ")
    ephemeris = CoupledEphemeris(
        context,
        elements,
        end_utc,
        sample_step_seconds=step_minutes * 60.0,
    )
    times = np.arange(
        ephemeris.epoch_tt_jd,
        end_tt + step_minutes / 1_440.0,
        step_minutes / 1_440.0,
    )
    rotation = _rotation_x(-np.radians(OBLIQUITY_J2000_DEG))
    real = (rotation @ ephemeris.relative("real_moon", times).T).T
    second = (rotation @ ephemeris.relative("second_moon", times).T).T

    limit = 440_000.0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    ):
        figure = plt.figure(figsize=(14.4, 8.2), facecolor="white")
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(1.08, 0.92),
            left=0.055,
            right=0.975,
            bottom=0.105,
            top=0.80,
            wspace=0.04,
        )
        top = figure.add_subplot(grid[0, 0])
        oblique = figure.add_subplot(grid[0, 1], projection="3d")

        for radius in (100_000.0, 200_000.0, 300_000.0, 400_000.0):
            top.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    edgecolor=GRID,
                    linewidth=0.75,
                    linestyle=(0, (2.5, 4.0)),
                    zorder=0,
                )
            )
        top.plot(real[:, 0], real[:, 1], color=REAL_BLUE, linewidth=2.2, label="Real Moon")
        top.plot(
            second[:, 0],
            second[:, 1],
            color=SECOND_ORANGE,
            linewidth=1.65,
            alpha=0.94,
            label="Second moon",
        )
        top.scatter(
            real[0, 0], real[0, 1], s=105, color=REAL_BLUE, edgecolor="white", linewidth=1.3, zorder=6
        )
        top.scatter(
            second[0, 0],
            second[0, 1],
            s=78,
            color=SECOND_ORANGE,
            edgecolor="white",
            linewidth=1.3,
            zorder=6,
        )
        top.add_patch(
            Circle(
                (0.0, 0.0),
                WGS84_A_KM,
                facecolor=EARTH_BLUE,
                edgecolor=EARTH_EDGE,
                linewidth=1.2,
                zorder=7,
            )
        )
        top.text(0.0, -20_000.0, "Earth", ha="center", va="top", fontsize=9.5, color=INK)
        top.set_xlim(-limit, limit)
        top.set_ylim(-limit, limit)
        top.set_aspect("equal")
        top.set_title("Top-down on the ecliptic", loc="left", fontsize=15, fontweight="bold", pad=14)
        top.set_xlabel("Ecliptic x (km)")
        top.set_ylabel("Ecliptic y (km)")
        top.spines[["top", "right", "bottom", "left"]].set_visible(False)
        top.tick_params(length=0)
        top.grid(False)
        legend = top.legend(
            loc="lower left",
            frameon=False,
            fontsize=10,
            handlelength=2.8,
            borderaxespad=0.2,
        )
        for line in legend.get_lines():
            line.set_linewidth(3.0)

        oblique.plot(real[:, 0], real[:, 1], real[:, 2], color=REAL_BLUE, linewidth=2.1)
        oblique.plot(
            second[:, 0], second[:, 1], second[:, 2], color=SECOND_ORANGE, linewidth=1.6
        )
        oblique.scatter(
            [real[0, 0]], [real[0, 1]], [real[0, 2]], s=80, color=REAL_BLUE, edgecolor="white"
        )
        oblique.scatter(
            [second[0, 0]],
            [second[0, 1]],
            [second[0, 2]],
            s=62,
            color=SECOND_ORANGE,
            edgecolor="white",
        )
        oblique.scatter([0.0], [0.0], [0.0], s=150, color=EARTH_BLUE, edgecolor=EARTH_EDGE)
        plane = np.linspace(-limit, limit, 2)
        plane_x, plane_y = np.meshgrid(plane, plane)
        oblique.plot_surface(
            plane_x,
            plane_y,
            np.zeros_like(plane_x),
            color=GRID,
            alpha=0.13,
            linewidth=0,
            shade=False,
        )
        oblique.set_xlim(-limit, limit)
        oblique.set_ylim(-limit, limit)
        oblique.set_zlim(-150_000.0, 150_000.0)
        oblique.set_box_aspect((1.0, 1.0, 0.42))
        oblique.view_init(elev=24.0, azim=-58.0)
        oblique.set_title("Oblique view reveals the tilt", loc="left", fontsize=15, fontweight="bold", pad=14)
        oblique.set_xlabel("x (km)", labelpad=8)
        oblique.set_ylabel("y (km)", labelpad=8)
        oblique.set_zlabel("z (km)", labelpad=5)
        oblique.tick_params(labelsize=8, pad=0)
        for axis in (oblique.xaxis, oblique.yaxis, oblique.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["color"] = GRID
            axis._axinfo["grid"]["linewidth"] = 0.55

        figure.text(
            0.055,
            0.94,
            "Two moons, one Earth",
            fontsize=27,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )
        figure.text(
            0.055,
            0.887,
            "Actual coupled four-body trajectories · 10 July–10 August 2026 · distances to scale",
            fontsize=12.5,
            color=MUTED,
            ha="left",
            va="top",
        )
        callout = figure.text(
            0.975,
            0.905,
            "Real Moon  ≈ 384,000 km  ·  27.3 d\nSecond moon  ≈ 180,000 km  ·  8.79 d",
            fontsize=10.5,
            color=INK,
            ha="right",
            va="top",
            linespacing=1.45,
        )
        callout.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
        figure.text(
            0.055,
            0.035,
            "Moon markers are enlarged for legibility; Earth and orbital distances share one physical scale. "
            "Curves are Earth-relative positions in the mean ecliptic J2000 frame.",
            fontsize=9.2,
            color=MUTED,
            ha="left",
            va="bottom",
        )
        figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
        figure.savefig(output.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
        plt.close(figure)
    return output.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--output", default="outputs/coupled/orbits/two_moon_orbits.png")
    parser.add_argument("--days", type=float, default=31.0)
    parser.add_argument("--step-minutes", type=float, default=20.0)
    args = parser.parse_args(argv)
    path = build_orbit_portrait(
        config_path=args.config,
        output_path=args.output,
        days=args.days,
        step_minutes=args.step_minutes,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
