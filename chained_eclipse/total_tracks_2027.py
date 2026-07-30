"""Plot every total second-moon solar-eclipse track found in 2027."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .eclipse_geometry import generate_central_track
from .ephemeris import load_ephemeris, time_iso_utc
from .models import OrbitalElements
from .orbital_dynamics import integrate_restricted


COLORS = ("#2166AC", "#1B9E77", "#E6AB02", "#D95F02", "#7570B3")


def plot_total_tracks(
    *,
    events_path: str | Path = "outputs/next_second_moon_eclipses.json",
    config_path: str | Path = "config/optimized_system.yaml",
    ephemeris_cache: str | Path = "data/ephemeris",
    output_dir: str | Path = "outputs/figures/2027_second_moon_total_tracks",
) -> dict[str, str]:
    payload = json.loads(Path(events_path).read_text(encoding="utf-8"))
    events = [
        event
        for event in payload["events"]
        if event["eclipse_type"] == "total" and event["maximum_utc"].startswith("2027-")
    ]
    if not events:
        raise ValueError("no 2027 total eclipses found in standalone results")

    context = load_ephemeris(ephemeris_cache)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = OrbitalElements(**config["orbital_elements"])
    end_tt = float(context.time_utc(events[-1]["global_end_utc"]).tt) + 0.02
    trajectory = integrate_restricted(
        context,
        elements,
        end_tt,
        rtol=3e-11,
        max_step_seconds=7_200.0,
    )

    projection = ccrs.Robinson(central_longitude=180.0)
    pacific_data = ccrs.PlateCarree(central_longitude=180.0)
    figure = plt.figure(figsize=(14.0, 7.6), facecolor="white")
    axis = figure.add_axes((0.04, 0.12, 0.92, 0.78), projection=projection)
    axis.set_global()
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCECF3", zorder=0)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECE8DC", zorder=1)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#DCECF3", zorder=2)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#77746B",
        linewidth=0.35,
        zorder=3,
    )
    axis.coastlines(resolution="50m", color="#28343D", linewidth=0.55, zorder=4)
    axis.gridlines(linewidth=0.4, color="#7C8994", alpha=0.45, linestyle=":")

    rows: list[dict[str, object]] = []
    maximum_times: list[float] = []
    for index, (event, color) in enumerate(zip(events, COLORS, strict=True), start=1):
        maximum_tt = float(context.time_utc(event["maximum_utc"]).tt)
        maximum_times.append(maximum_tt)
        track = generate_central_track(
            context,
            maximum_tt,
            "second_moon",
            position_provider=trajectory.position,
            body_radius_km=elements.radius_km,
            half_window_hours=3.0,
            step_seconds=20.0,
        )
        shifted_longitude = (
            np.asarray(track["longitude_deg"], dtype=float) - 180.0 + 180.0
        ) % 360.0 - 180.0
        label = context.time_utc(event["maximum_utc"]).utc_strftime("%b %d")
        axis.plot(
            shifted_longitude,
            track["latitude_deg"],
            color=color,
            linewidth=2.2,
            transform=pacific_data,
            zorder=6,
            label=f"{index}. {label}",
        )
        maximum_shifted = (float(event["longitude_deg"]) - 180.0 + 180.0) % 360.0 - 180.0
        axis.scatter(
            [maximum_shifted],
            [event["latitude_deg"]],
            s=44,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            transform=pacific_data,
            zorder=7,
        )
        axis.text(
            maximum_shifted + 2.0,
            event["latitude_deg"] + (2.0 if index < 5 else -4.0),
            str(index),
            color=color,
            fontsize=9,
            fontweight="bold",
            transform=pacific_data,
            zorder=8,
        )
        for tt_jd, latitude, longitude, core in zip(
            track["tt_jd"],
            track["latitude_deg"],
            track["longitude_deg"],
            track["signed_core_radius_km"],
            strict=True,
        ):
            rows.append(
                {
                    "event_number": index,
                    "event_date": label,
                    "time_utc": time_iso_utc(context.tt_jd(float(tt_jd)), places=3),
                    "latitude_deg": float(latitude),
                    "longitude_deg": float(longitude),
                    "umbra_radius_km": float(core),
                }
            )

    gaps_days = np.diff(maximum_times)
    mean_gap_days = float(np.mean(gaps_days))
    legend = axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(events),
        frameon=True,
        facecolor="white",
        edgecolor="#8C969E",
        framealpha=0.90,
        fontsize=9,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)
    figure.suptitle(
        "Five total solar eclipses from the fixed second-moon orbit · 2027",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.045,
        f"May 28–July 2 · mean spacing {mean_gap_days:.2f} days · "
        "lines are 20-second samples of the total-eclipse centerline",
        ha="center",
        fontsize=9.5,
        color="#4A5159",
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "all_total_tracks.png"
    figure.savefig(map_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    csv_path = output / "all_total_tracks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary_path = output / "events.json"
    summary_path.write_text(
        json.dumps(
            {
                "event_count": len(events),
                "mean_spacing_days": mean_gap_days,
                "spacing_days": gaps_days.tolist(),
                "events": events,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "map": str(map_path.resolve()),
        "track_csv": str(csv_path.resolve()),
        "event_json": str(summary_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="outputs/next_second_moon_eclipses.json")
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument("--ephemeris-cache", default="data/ephemeris")
    parser.add_argument(
        "--output-dir", default="outputs/figures/2027_second_moon_total_tracks"
    )
    args = parser.parse_args(argv)
    result = plot_total_tracks(
        events_path=args.events,
        config_path=args.config,
        ephemeris_cache=args.ephemeris_cache,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
