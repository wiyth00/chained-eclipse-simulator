"""Figures for the fully coupled 2026 chained-eclipse result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from .coupled_eclipse import CoupledEphemeris, generate_coupled_track
from .ephemeris import load_ephemeris
from .mapping import plot_contact_timeline, plot_world_tracks
from .models import OrbitalElements


def build_coupled_figures(
    *,
    results_path: str | Path = "outputs/coupled/coupled_eclipses.json",
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    output_dir: str | Path = "outputs/coupled/figures/20260812_coupled_chain",
) -> dict[str, str]:
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    pair = result["within_12h_pairs"][0]
    context = load_ephemeris(ephemeris_cache)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    ephemeris = CoupledEphemeris(
        context,
        elements,
        result["end_utc"],
        sample_step_seconds=300.0,
    )
    real_tt = float(context.time_utc(pair["real_maximum_utc"]).tt)
    second_tt = float(context.time_utc(pair["second_maximum_utc"]).tt)
    real_track = generate_coupled_track(
        ephemeris, "real_moon", real_tt, step_seconds=20.0
    )
    second_track = generate_coupled_track(
        ephemeris, "second_moon", second_tt, step_seconds=20.0
    )
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
        title="The chained eclipse survives in the coupled four-body model",
        subtitle=(
            "Second-moon totality occurs first; real-Moon totality follows "
            f"{pair['local_maximum_separation_hours']:.2f} h later."
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
        title="Coupled-model chained eclipse at the common Arctic site",
        output_path=timeline_path,
    )
    plt.close(figure)
    return {"map": str(map_path.resolve()), "timeline": str(timeline_path.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/coupled/coupled_eclipses.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output-dir", default="outputs/coupled/figures/20260812_coupled_chain"
    )
    args = parser.parse_args(argv)
    paths = build_coupled_figures(
        results_path=args.results,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_dir=args.output_dir,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
