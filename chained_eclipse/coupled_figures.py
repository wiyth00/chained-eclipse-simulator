"""Figures for the fully coupled 2026 chained-eclipse result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import yaml

from .constants import SECONDS_PER_DAY
from .coupled_eclipse import (
    CoupledEphemeris,
    coupled_sky_plane_disks,
    generate_coupled_track,
)
from .ephemeris import load_ephemeris
from .mapping import plot_contact_timeline, plot_world_tracks
from .moon_architecture import architecture_from_config, elements_from_config


def _plot_compound_sky_sequence(
    ephemeris: CoupledEphemeris,
    pair: dict[str, object],
    output_path: Path,
) -> Path:
    """Render the two apparent lunar disks around the common-site maximum."""

    latitude = float(pair["best_common_latitude_deg"])
    longitude = float(pair["best_common_longitude_deg"])
    center_tt = float(ephemeris.context.time_utc(pair["second_local"]["maximum_utc"]).tt)
    offsets_minutes = (-40.0, 0.0, 40.0)
    samples = [
        coupled_sky_plane_disks(
            ephemeris,
            center_tt + offset * 60.0 / SECONDS_PER_DAY,
            latitude,
            longitude,
        )
        for offset in offsets_minutes
    ]
    extent = 0.0
    for sample in samples:
        for disk in sample.values():
            extent = max(
                extent,
                abs(disk["east_deg"]) + disk["angular_radius_deg"],
                abs(disk["north_deg"]) + disk["angular_radius_deg"],
            )
    limit = max(0.42, 1.12 * extent)
    colors = {"real_moon": "#24649A", "second_moon": "#D47424"}
    labels = {"real_moon": "Real Moon", "second_moon": "Giant moon"}
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 6.2),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.80, wspace=0.04)
    figure.suptitle(
        "What the compound eclipse looks like from the common site",
        y=0.965,
        fontsize=19,
        fontweight="bold",
    )
    for axis, offset, sample in zip(axes, offsets_minutes, samples, strict=True):
        sun = sample["sun"]
        axis.add_patch(
            Circle(
                (sun["east_deg"], sun["north_deg"]),
                sun["angular_radius_deg"],
                facecolor="#FFD166",
                edgecolor="#D49A17",
                linewidth=1.6,
                zorder=1,
            )
        )
        # Paint the more distant moon first, then retain colored outlines as a
        # scientific overlay even where one black silhouette hides the other.
        moon_names = sorted(
            ("real_moon", "second_moon"),
            key=lambda name: sample[name]["distance_km"],
            reverse=True,
        )
        for zorder, name in enumerate(moon_names, start=2):
            disk = sample[name]
            axis.add_patch(
                Circle(
                    (disk["east_deg"], disk["north_deg"]),
                    disk["angular_radius_deg"],
                    facecolor="#111827",
                    edgecolor=colors[name],
                    linewidth=2.4,
                    alpha=0.92,
                    zorder=zorder,
                )
            )
        for name in ("real_moon", "second_moon"):
            disk = sample[name]
            axis.add_patch(
                Circle(
                    (disk["east_deg"], disk["north_deg"]),
                    disk["angular_radius_deg"],
                    fill=False,
                    edgecolor=colors[name],
                    linewidth=2.0,
                    zorder=6,
                )
            )
        time = ephemeris.time(center_tt + offset * 60.0 / SECONDS_PER_DAY)
        axis.set_title(time.utc_strftime("%H:%M UTC"), fontsize=13, fontweight="bold")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.grid(True, color="#DCE4EC", linewidth=0.7)
        axis.axhline(0.0, color="#B9C6D5", linewidth=0.7)
        axis.axvline(0.0, color="#B9C6D5", linewidth=0.7)
        axis.text(
            0.03,
            0.04,
            f"{offset:+.0f} min",
            transform=axis.transAxes,
            fontsize=10,
            color="#68778F",
        )
    axes[0].set_ylabel("Celestial north (deg)")
    handles = [
        Line2D((0,), (0,), color=colors[name], linewidth=3.0, label=labels[name])
        for name in ("real_moon", "second_moon")
    ]
    figure.supxlabel("Celestial east (deg)", y=0.155)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.025,
        "Colored outlines identify the moons; their visible silhouettes are black. "
        "The middle panel is the giant moon's local maximum.",
        ha="center",
        fontsize=9.5,
        color="#68778F",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path.resolve()


def build_coupled_figures(
    *,
    results_path: str | Path = "outputs/coupled/coupled_eclipses.json",
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    output_dir: str | Path = "outputs/coupled/figures/20260812_coupled_chain",
    pair_index: int = 0,
) -> dict[str, str]:
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    pairs = result["within_12h_pairs"]
    if not pairs:
        raise ValueError("the coupled result contains no eclipse pairs within 12 hours")
    try:
        pair = pairs[pair_index]
    except IndexError as exc:
        raise ValueError(
            f"pair_index {pair_index} is outside the {len(pairs)} available pairs"
        ) from exc
    context = load_ephemeris(ephemeris_cache)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = elements_from_config(config)
    binary_architecture = architecture_from_config(config)
    ephemeris = CoupledEphemeris(
        context,
        elements,
        result["end_utc"],
        sample_step_seconds=300.0,
        binary_architecture=binary_architecture,
    )
    real_tt = float(context.time_utc(pair["real_maximum_utc"]).tt)
    second_tt = float(context.time_utc(pair["second_maximum_utc"]).tt)
    real_track = generate_coupled_track(ephemeris, "real_moon", real_tt, step_seconds=20.0)
    second_track = generate_coupled_track(ephemeris, "second_moon", second_tt, step_seconds=20.0)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "world_tracks.png"
    figure, _ = plot_world_tracks(
        {
            "label": "Real Moon",
            "longitude_deg": real_track["longitude_deg"],
            "latitude_deg": real_track["latitude_deg"],
        },
        {
            "label": "Second moon",
            "longitude_deg": second_track["longitude_deg"],
            "latitude_deg": second_track["latitude_deg"],
        },
        (
            pair["best_common_latitude_deg"],
            pair["best_common_longitude_deg"],
            "Coupled-model common site",
        ),
        title="Paired eclipses in the coupled four-body model",
        subtitle=(
            f"Real Moon: {pair['real_type']}; second moon: {pair['second_type']}; "
            f"closest tracks {pair['track_distance_km']:.1f} km apart."
        ),
        output_path=map_path,
    )
    plt.close(figure)
    timeline_path = output / "contact_timeline.png"
    real_local = {"body": "Real Moon", **pair["real_local"]}
    second_local = {"body": "Second moon", **pair["second_local"]}
    figure, _ = plot_contact_timeline(
        real_local,
        second_local,
        title="Coupled-model paired eclipses at the common site",
        output_path=timeline_path,
    )
    plt.close(figure)
    sky_path = _plot_compound_sky_sequence(
        ephemeris,
        pair,
        output / "apparent_sky_sequence.png",
    )
    return {
        "map": str(map_path.resolve()),
        "timeline": str(timeline_path.resolve()),
        "apparent_sky": str(sky_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/coupled/coupled_eclipses.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument("--output-dir", default="outputs/coupled/figures/20260812_coupled_chain")
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="zero-based within-12h pair index; negative indices count from the end",
    )
    args = parser.parse_args(argv)
    paths = build_coupled_figures(
        results_path=args.results,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_dir=args.output_dir,
        pair_index=args.pair_index,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
