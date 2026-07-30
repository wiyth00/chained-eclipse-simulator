"""Build an equirectangular map of the standout coupled-model eclipse tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .coupled_eclipse import CoupledEphemeris, generate_coupled_track
from .ephemeris import load_ephemeris, time_iso_utc
from .moon_architecture import architecture_from_config, elements_from_config


REAL_BLUE = "#24649A"
SECOND_ORANGE = "#D47424"


def _split_at_antimeridian(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> list[dict[str, list[float]]]:
    """Split a geographic polyline wherever it crosses the map seam."""

    if len(longitudes) == 0:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(longitudes)) > 180.0) + 1
    indices = np.r_[0, breaks, len(longitudes)]
    segments: list[dict[str, list[float]]] = []
    for start, end in zip(indices[:-1], indices[1:], strict=True):
        if end - start < 2:
            continue
        segments.append(
            {
                "longitude_deg": np.round(longitudes[start:end], 4).tolist(),
                "latitude_deg": np.round(latitudes[start:end], 4).tolist(),
            }
        )
    return segments


def build_standout_tracks(
    *,
    climate_path: str | Path = "outputs/coupled/eclipse_climate_30y/climate.json",
    config_path: str | Path = "config/optimized_system.yaml",
    output_dir: str | Path | None = None,
    per_body: int = 4,
    step_seconds: float = 60.0,
) -> dict[str, Any]:
    """Generate track coordinates and a world map for the longest total events."""

    climate_source = Path(climate_path)
    climate = json.loads(climate_source.read_text(encoding="utf-8"))
    dynamics_model = str(climate.get("dynamics_model", "baseline")).strip().lower()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    elements = elements_from_config(config)
    binary_architecture = architecture_from_config(config)
    context = load_ephemeris("data/ephemeris")
    ephemeris_class: type[CoupledEphemeris]
    if dynamics_model == "baseline":
        ephemeris_class = CoupledEphemeris
    elif dynamics_model == "enhanced":
        # Keep the enhanced force stack optional for baseline-only use.
        from .enhanced_ephemeris import EnhancedEphemeris

        ephemeris_class = EnhancedEphemeris
    else:
        raise ValueError(
            f"unsupported climate dynamics_model {dynamics_model!r}; "
            "expected 'baseline' or 'enhanced'"
        )
    ephemeris_kwargs: dict[str, Any] = {
        "sample_step_seconds": float(climate["trajectory"]["sample_step_seconds"])
    }
    ephemeris_kwargs["binary_architecture"] = binary_architecture
    ephemeris = ephemeris_class(
        context,
        elements,
        str(climate["end_utc"]),
        **ephemeris_kwargs,
    )

    selected: list[dict[str, Any]] = []
    for body in ("real_moon", "second_moon"):
        events = [
            event
            for event in climate["notable_solar_events"]
            if event["body"] == body
            and event["eclipse_type"] in {"total", "hybrid"}
        ]
        events.sort(
            key=lambda event: float(event.get("central_duration_s") or 0.0),
            reverse=True,
        )
        selected.extend(events[:per_body])
    selected.sort(key=lambda event: str(event["maximum_utc"]))

    track_events: list[dict[str, Any]] = []
    for index, event in enumerate(selected, start=1):
        maximum_tt = float(context.time_utc(str(event["maximum_utc"])).tt)
        track = generate_coupled_track(
            ephemeris,
            str(event["body"]),
            maximum_tt,
            half_window_hours=4.0,
            step_seconds=step_seconds,
        )
        longitudes = np.asarray(track["longitude_deg"], dtype=float)
        latitudes = np.asarray(track["latitude_deg"], dtype=float)
        cores = np.asarray(track["signed_core_radius_km"], dtype=float)
        track_events.append(
            {
                "id": index,
                "body": event["body"],
                "body_label": event["body_label"],
                "eclipse_type": event["eclipse_type"],
                "maximum_utc": event["maximum_utc"],
                "latitude_deg": event["latitude_deg"],
                "longitude_deg": event["longitude_deg"],
                "magnitude": event["magnitude"],
                "solar_altitude_deg": event["solar_altitude_deg"],
                "central_duration_s": event["central_duration_s"],
                "maximum_core_radius_km": float(np.max(cores)),
                "segments": _split_at_antimeridian(longitudes, latitudes),
                "sample_count": int(len(longitudes)),
                "track_start_utc": (
                    time_iso_utc(context.tt_jd(float(track["tt_jd"][0])))
                    if len(track["tt_jd"])
                    else None
                ),
                "track_end_utc": (
                    time_iso_utc(context.tt_jd(float(track["tt_jd"][-1])))
                    if len(track["tt_jd"])
                    else None
                ),
            }
        )

    # By default keep derived tracks beside their source catalog.  This retains
    # the historical baseline directory while naturally routing an enhanced
    # catalog to eclipse_climate_30y_enhanced (or any custom catalog directory).
    output = climate_source.parent if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_title = (
        f"{per_body} longest total-solar-eclipse tracks from each moon · "
        f"{dynamics_model.title()} dynamics"
    )
    payload = {
        "schema_version": "1.0",
        "dynamics_model": dynamics_model,
        "title": figure_title,
        "selection": (
            f"Top {per_body} central-duration total or hybrid solar eclipses "
            "for each moon"
        ),
        "step_seconds": step_seconds,
        "events": track_events,
    }
    json_path = output / "standout_tracks.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot_standout_tracks(
        track_events,
        output / "standout_tracks.png",
        title=figure_title,
    )
    return payload


def _plot_standout_tracks(
    events: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    projection = ccrs.PlateCarree()
    figure = plt.figure(figsize=(15.2, 8.2), facecolor="white")
    axis = figure.add_axes((0.04, 0.12, 0.92, 0.74), projection=projection)
    axis.set_global()
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCECF3", zorder=0)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECE8DC", zorder=1)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#DCECF3", zorder=2)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#77746B",
        linewidth=0.3,
        zorder=3,
    )
    axis.coastlines(resolution="50m", color="#28343D", linewidth=0.5, zorder=4)
    grid = axis.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="#7C8994",
        alpha=0.45,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    grid.top_labels = False
    grid.right_labels = False

    handles: list[Any] = []
    labels: list[str] = []
    for event in events:
        color = REAL_BLUE if event["body"] == "real_moon" else SECOND_ORANGE
        linewidth = 1.5 + min(float(event["central_duration_s"]) / 300.0, 1.8)
        handle = None
        for segment in event["segments"]:
            (handle,) = axis.plot(
                segment["longitude_deg"],
                segment["latitude_deg"],
                color=color,
                linewidth=linewidth,
                alpha=0.86,
                transform=projection,
                zorder=6,
            )
        axis.scatter(
            [event["longitude_deg"]],
            [event["latitude_deg"]],
            s=44,
            marker="*",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            transform=projection,
            zorder=7,
        )
        duration = float(event["central_duration_s"])
        if handle is not None:
            handles.append(handle)
            labels.append(
                f"{str(event['maximum_utc'])[:10]} · {int(duration // 60)}m {duration % 60:04.1f}s"
            )

    figure.text(
        0.04,
        0.965,
        title,
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
        color="#102039",
    )
    figure.text(
        0.04,
        0.91,
        "Blue: real Moon · orange: second moon · stars: global maximum · line width follows central duration",
        fontsize=11.5,
        ha="left",
        va="top",
        color="#68778F",
    )
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=8.4,
    )
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--climate", default="outputs/coupled/eclipse_climate_30y/climate.json"
    )
    parser.add_argument("--config", default="config/optimized_system.yaml")
    parser.add_argument(
        "--output-dir",
        help="Defaults to the directory containing the selected climate catalog",
    )
    parser.add_argument("--per-body", type=int, default=4)
    parser.add_argument("--step-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    result = build_standout_tracks(
        climate_path=args.climate,
        config_path=args.config,
        output_dir=args.output_dir,
        per_body=args.per_body,
        step_seconds=args.step_seconds,
    )
    print(json.dumps({"event_count": len(result["events"])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
