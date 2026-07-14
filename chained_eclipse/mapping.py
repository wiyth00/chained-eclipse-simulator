"""Deterministic static figures for chained-eclipse results.

The plotting layer deliberately accepts small plotting dataclasses *or* plain
mappings/objects.  Search and stability code can therefore hand figures either
native result objects or JSON-ready dictionaries without coupling those layers
to Matplotlib.

Partial-eclipse boundaries are never invented here.  They must be supplied as
explicit boundary polylines or as a sampled geographic field and contour level;
the legend labels the latter as ``grid-derived``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.util import add_cyclic_point
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
from numpy.typing import ArrayLike


# Restrained, colorblind-safe-enough roots plus neutral scaffolding.  Series
# also differ by line style and marker, so color is not the sole distinction.
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
LAND = "#ECEAE4"
OCEAN = "#F7FAFC"
REAL_BLUE = "#245B8F"
SECOND_ORANGE = "#C86F24"
SUN_GOLD = "#B58A1F"
PALETTE = (REAL_BLUE, SECOND_ORANGE, "#7A6A9D", "#6F7F45", "#B45C73")

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
}


@dataclass(frozen=True, slots=True)
class TrackPlotData:
    """One eclipse path and optional partial-boundary evidence."""

    label: str
    longitude_deg: ArrayLike
    latitude_deg: ArrayLike
    partial_boundaries: Sequence[tuple[ArrayLike, ArrayLike]] = ()
    grid_longitude_deg: ArrayLike | None = None
    grid_latitude_deg: ArrayLike | None = None
    grid_values: ArrayLike | None = None
    grid_boundary_level: float = 0.0
    boundary_note: str | None = None


@dataclass(frozen=True, slots=True)
class AngularSeriesPlotData:
    """Topocentric angular geometry sampled around one eclipse maximum."""

    label: str
    time_offset_seconds: ArrayLike
    center_separation_deg: ArrayLike
    solar_angular_diameter_deg: ArrayLike
    occulter_angular_diameter_deg: ArrayLike
    maximum_utc: str | None = None
    color: str | None = None


@dataclass(frozen=True, slots=True)
class StabilitySeriesPlotData:
    """Osculating elements for one body sampled over a stability integration."""

    label: str
    time_years: ArrayLike
    semimajor_axis_km: ArrayLike
    eccentricity: ArrayLike
    inclination_deg: ArrayLike
    color: str | None = None


def _value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _as_array(value: ArrayLike, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).squeeze()
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {result.shape}")
    return result


def _broadcast_array(value: ArrayLike, size: int, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).squeeze()
    if result.ndim == 0:
        return np.full(size, float(result))
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{name} must be scalar or length {size}; got shape {result.shape}")
    return result


def _save(fig: Figure, output_path: str | PathLike[str] | None, dpi: int) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def _clean_axes(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def _normalize_longitudes(longitudes: np.ndarray) -> np.ndarray:
    return (longitudes + 180.0) % 360.0 - 180.0


def _break_dateline(longitude: ArrayLike, latitude: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    lon = _as_array(longitude, name="longitude_deg")
    lat = _as_array(latitude, name="latitude_deg")
    if lon.shape != lat.shape:
        raise ValueError("longitude and latitude arrays must have identical shapes")
    lon = _normalize_longitudes(lon)
    invalid = ~np.isfinite(lon) | ~np.isfinite(lat)
    jumps = np.zeros(len(lon), dtype=bool)
    if len(lon) > 1:
        jumps[1:] = np.abs(np.diff(lon)) > 180.0
    split = invalid | jumps
    lon = lon.astype(float, copy=True)
    lat = lat.astype(float, copy=True)
    lon[split] = np.nan
    lat[split] = np.nan
    return lon, lat


def _coerce_boundaries(source: Any) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    raw = _value(source, "partial_boundaries", "boundaries", default=())
    boundaries: list[tuple[np.ndarray, np.ndarray]] = []
    for boundary in raw or ():
        if isinstance(boundary, Mapping) or is_dataclass(boundary):
            lon = _value(boundary, "longitude_deg", "longitudes_deg", "lon")
            lat = _value(boundary, "latitude_deg", "latitudes_deg", "lat")
        else:
            try:
                lon, lat = boundary
            except (TypeError, ValueError) as exc:
                raise ValueError("each partial boundary must provide longitude and latitude") from exc
        boundaries.append((_as_array(lon, name="boundary longitude"), _as_array(lat, name="boundary latitude")))

    if boundaries:
        return tuple(boundaries)

    longitudes = _value(
        source,
        "partial_boundary_longitude_deg",
        "partial_boundary_longitudes_deg",
        "boundary_longitude_deg",
        "boundary_longitudes_deg",
    )
    latitudes = _value(
        source,
        "partial_boundary_latitude_deg",
        "partial_boundary_latitudes_deg",
        "boundary_latitude_deg",
        "boundary_latitudes_deg",
    )
    if longitudes is None and latitudes is None:
        return ()
    if longitudes is None or latitudes is None:
        raise ValueError("partial boundary longitude and latitude must both be provided")
    lon_array = np.asarray(longitudes, dtype=float)
    lat_array = np.asarray(latitudes, dtype=float)
    if lon_array.shape != lat_array.shape:
        raise ValueError("partial boundary longitude and latitude shapes differ")
    if lon_array.ndim == 1:
        return ((lon_array, lat_array),)
    if lon_array.ndim == 2:
        return tuple((lon_array[index], lat_array[index]) for index in range(lon_array.shape[0]))
    raise ValueError("partial boundary arrays must be one- or two-dimensional")


def _coerce_track(source: TrackPlotData | Mapping[str, Any] | Any, fallback_label: str) -> TrackPlotData:
    if isinstance(source, TrackPlotData):
        return source
    if isinstance(source, tuple) and len(source) == 2:
        longitude, latitude = source
        return TrackPlotData(fallback_label, longitude, latitude)
    longitude = _value(
        source,
        "longitude_deg",
        "longitudes_deg",
        "central_longitude_deg",
        "central_longitudes_deg",
        "lon",
    )
    latitude = _value(
        source,
        "latitude_deg",
        "latitudes_deg",
        "central_latitude_deg",
        "central_latitudes_deg",
        "lat",
    )
    if longitude is None or latitude is None:
        raise ValueError(f"{fallback_label} track needs central longitude and latitude arrays")
    return TrackPlotData(
        label=str(_value(source, "label", "name", default=fallback_label)),
        longitude_deg=longitude,
        latitude_deg=latitude,
        partial_boundaries=_coerce_boundaries(source),
        grid_longitude_deg=_value(source, "grid_longitude_deg", "grid_longitudes_deg", "grid_lon"),
        grid_latitude_deg=_value(source, "grid_latitude_deg", "grid_latitudes_deg", "grid_lat"),
        grid_values=_value(source, "grid_values", "partial_grid_values", "magnitude_grid"),
        grid_boundary_level=float(
            _value(source, "grid_boundary_level", "partial_level", default=0.0)
        ),
        boundary_note=_value(source, "boundary_note", "partial_boundary_note"),
    )


def _plot_track(
    ax: GeoAxes,
    track: TrackPlotData,
    *,
    color: str,
    linestyle: str,
) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    lon, lat = _break_dateline(track.longitude_deg, track.latitude_deg)
    central_handle = ax.plot(
        lon,
        lat,
        color=color,
        linewidth=2.4,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
        zorder=7,
        label=f"{track.label} central line",
    )[0]
    handles.append(central_handle)
    labels.append(f"{track.label} central line")

    boundary_plotted = False
    for boundary_lon, boundary_lat in track.partial_boundaries:
        b_lon, b_lat = _break_dateline(boundary_lon, boundary_lat)
        ax.plot(
            b_lon,
            b_lat,
            color=color,
            linewidth=1.15,
            linestyle=(0, (2.0, 2.5)),
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        boundary_plotted = True

    grid_parts = (track.grid_longitude_deg, track.grid_latitude_deg, track.grid_values)
    grid_complete = all(value is not None for value in grid_parts)
    if any(value is not None for value in grid_parts) and not grid_complete:
        raise ValueError(
            f"{track.label} grid boundary needs longitude, latitude, and values"
        )
    if grid_complete:
        grid_lon = np.asarray(track.grid_longitude_deg, dtype=float)
        grid_lat = np.asarray(track.grid_latitude_deg, dtype=float)
        grid_values = np.asarray(track.grid_values, dtype=float)
        if grid_lon.ndim == grid_lat.ndim == 1:
            expected = (len(grid_lat), len(grid_lon))
            if grid_values.shape != expected:
                raise ValueError(
                    f"{track.label} grid values need shape {expected}; got {grid_values.shape}"
                )
            if len(grid_lon) > 2 and np.ptp(grid_lon) < 359.999:
                grid_values, grid_lon = add_cyclic_point(grid_values, coord=grid_lon)
            mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
        elif grid_lon.ndim == grid_lat.ndim == 2:
            if grid_lon.shape != grid_lat.shape or grid_lon.shape != grid_values.shape:
                raise ValueError(f"{track.label} two-dimensional grid arrays must share a shape")
            mesh_lon, mesh_lat = grid_lon, grid_lat
        else:
            raise ValueError(f"{track.label} grid coordinates must both be 1-D or both be 2-D")
        if not np.nanmin(grid_values) <= track.grid_boundary_level <= np.nanmax(grid_values):
            raise ValueError(
                f"{track.label} grid boundary level {track.grid_boundary_level:g} "
                "falls outside the sampled field"
            )
        ax.contour(
            mesh_lon,
            mesh_lat,
            grid_values,
            levels=[track.grid_boundary_level],
            colors=[color],
            linewidths=1.15,
            linestyles=[(0, (2.0, 2.5))],
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        boundary_plotted = True

    if boundary_plotted:
        source = track.boundary_note or ("grid-derived" if grid_complete else "numerical")
        handles.append(Line2D([], [], color=color, linewidth=1.15, linestyle=(0, (2.0, 2.5))))
        labels.append(f"{track.label} partial boundary ({source})")
    return handles, labels


def _coerce_location(location: Any) -> tuple[float, float, str]:
    if isinstance(location, tuple) or isinstance(location, list):
        if len(location) < 2:
            raise ValueError("best_location tuple must contain latitude and longitude")
        latitude, longitude = location[:2]
        label = str(location[2]) if len(location) > 2 else "Best common location"
        return float(latitude), float(longitude), label
    latitude = _value(location, "latitude_deg", "best_latitude_deg", "lat")
    longitude = _value(location, "longitude_deg", "best_longitude_deg", "lon")
    if latitude is None or longitude is None:
        raise ValueError("best_location needs latitude and longitude")
    return float(latitude), float(longitude), str(
        _value(location, "label", "name", default="Best common location")
    )


def plot_world_tracks(
    real_track: TrackPlotData | Mapping[str, Any] | Any,
    second_track: TrackPlotData | Mapping[str, Any] | Any,
    best_location: Any,
    *,
    title: str = "Chained solar-eclipse ground tracks",
    subtitle: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    output_path: str | PathLike[str] | None = None,
    dpi: int = 180,
) -> tuple[Figure, GeoAxes]:
    """Plot both central tracks, supported partial boundaries, and best site.

    ``extent`` is ``(west, east, south, north)`` in degrees.  Omit it for the
    required world map.  Grid-derived boundaries are generated only when a
    sampled field and explicit contour level are supplied.
    """

    real = _coerce_track(real_track, "Real Moon")
    second = _coerce_track(second_track, "Second moon")
    latitude, longitude, location_label = _coerce_location(best_location)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.2, 7.2), constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_facecolor(OCEAN)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=1)
        ax.coastlines(resolution="50m", color=MUTED, linewidth=0.55, zorder=3)
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor="#9AA4B2",
            linewidth=0.35,
            alpha=0.75,
            zorder=2,
        )
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        gridlines = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            xlocs=np.arange(-180, 181, 60),
            ylocs=np.arange(-60, 61, 30),
            color=GRID,
            linewidth=0.55,
            alpha=0.75,
            zorder=2,
        )
        gridlines.n_steps = 80

        handles, labels = _plot_track(ax, real, color=REAL_BLUE, linestyle="-")
        second_handles, second_labels = _plot_track(
            ax, second, color=SECOND_ORANGE, linestyle=(0, (7.0, 3.0))
        )
        handles.extend(second_handles)
        labels.extend(second_labels)

        site = ax.scatter(
            [longitude],
            [latitude],
            marker="*",
            s=155,
            facecolor=INK,
            edgecolor="white",
            linewidth=1.0,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        handles.append(site)
        labels.append(location_label)
        ax.annotate(
            location_label,
            xy=(longitude, latitude),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": GRID, "alpha": 0.94},
            zorder=11,
        )

        fig.suptitle(title, x=0.07, y=0.965, ha="left", fontsize=17, fontweight="bold", color=INK)
        if subtitle:
            fig.text(0.07, 0.925, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=min(3, len(handles)),
            fontsize=8.7,
            handlelength=3.0,
            columnspacing=1.5,
        )
        fig.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.13)
        _save(fig, output_path, dpi)
    return fig, ax


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, np.datetime64):
        microseconds = value.astype("datetime64[us]").astype(np.int64)
        parsed = datetime.fromtimestamp(float(microseconds) / 1_000_000.0, tz=timezone.utc)
    elif hasattr(value, "utc_datetime"):
        parsed = value.utc_datetime()
    else:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _contact_payload(source: Any, fallback_label: str) -> dict[str, Any]:
    maximum = _parse_datetime(_value(source, "maximum_utc", "maximum", "midpoint_utc"))
    if maximum is None:
        raise ValueError(f"{fallback_label} circumstances need maximum_utc")
    return {
        "label": str(_value(source, "body", "label", "name", default=fallback_label)),
        "kind": str(_value(source, "eclipse_type", "kind", default="eclipse")),
        "magnitude": _value(source, "magnitude"),
        "c1": _parse_datetime(_value(source, "c1_utc", "first_contact_utc", "c1")),
        "c2": _parse_datetime(_value(source, "c2_utc", "second_contact_utc", "c2")),
        "maximum": maximum,
        "c3": _parse_datetime(_value(source, "c3_utc", "third_contact_utc", "c3")),
        "c4": _parse_datetime(_value(source, "c4_utc", "fourth_contact_utc", "c4")),
    }


def _format_contact_time(value: datetime, include_date: bool) -> str:
    return value.strftime("%b %d\n%H:%M:%S" if include_date else "%H:%M:%S")


def plot_contact_timeline(
    real_circumstances: Any,
    second_circumstances: Any,
    *,
    title: str = "Eclipse contact timeline",
    subtitle: str = "Contact times at the best common observing location · UTC",
    output_path: str | PathLike[str] | None = None,
    dpi: int = 180,
) -> tuple[Figure, Axes]:
    """Plot C1–C4 and maximum for both local eclipses on one UTC axis."""

    real = _contact_payload(real_circumstances, "Real Moon")
    second = _contact_payload(second_circumstances, "Second moon")
    rows = ((real, 1.0, REAL_BLUE), (second, 0.0, SECOND_ORANGE))
    known_times = [row[key] for row, _, _ in rows for key in ("c1", "c2", "maximum", "c3", "c4") if row[key] is not None]
    span_seconds = (max(known_times) - min(known_times)).total_seconds()
    include_date = min(known_times).date() != max(known_times).date()

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        for row, y, color in rows:
            external_start = row["c1"] or row["maximum"]
            external_end = row["c4"] or row["maximum"]
            compact_event = (
                (external_end - external_start).total_seconds() < 3_600
            )
            ax.plot(
                [external_start, external_end],
                [y, y],
                color=color,
                linewidth=12,
                alpha=0.18,
                zorder=1,
            )
            if row["c2"] is not None and row["c3"] is not None:
                ax.plot(
                    [row["c2"], row["c3"]],
                    [y, y],
                    color=color,
                    linewidth=12,
                    alpha=0.78,
                    zorder=2,
                )

            contacts = (
                ("C1", row["c1"], "o", "white"),
                ("C2", row["c2"], "o", color),
                ("Max", row["maximum"], "D", color),
                ("C3", row["c3"], "o", color),
                ("C4", row["c4"], "o", "white"),
            )
            for index, (name, instant, marker, fill) in enumerate(contacts):
                if instant is None:
                    continue
                ax.scatter(
                    [instant],
                    [y],
                    marker=marker,
                    s=54 if name == "Max" else 40,
                    facecolor=fill,
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=4,
                )
                direction = 1 if y > 0.5 or (compact_event and name == "Max") else -1
                horizontal_offsets = (0, -40, 0, 40, 0)
                vertical_offsets = (
                    (24, 48, 24, 48, 24)
                    if compact_event
                    else (23, 42, 24, 42, 23)
                )
                ax.annotate(
                    f"{name}\n{_format_contact_time(instant, include_date)}",
                    xy=(instant, y),
                    xytext=(horizontal_offsets[index], direction * vertical_offsets[index]),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if direction > 0 else "top",
                    fontsize=8,
                    color=INK,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6, "alpha": 0.65},
                )

        first_max = real["maximum"]
        second_max = second["maximum"]
        earlier, later = sorted((first_max, second_max))
        gap_seconds = abs((second_max - first_max).total_seconds())
        gap_hours, remainder = divmod(int(round(gap_seconds)), 3_600)
        gap_minutes, gap_seconds_round = divmod(remainder, 60)
        gap_text = f"Midpoint gap  {gap_hours:d}h {gap_minutes:02d}m {gap_seconds_round:02d}s"
        ax.annotate(
            "",
            xy=(later, 0.5),
            xytext=(earlier, 0.5),
            arrowprops={"arrowstyle": "<->", "color": MUTED, "linewidth": 1.0},
            zorder=3,
        )
        ax.text(
            earlier + (later - earlier) / 2,
            0.5,
            gap_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color=MUTED,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"},
            zorder=5,
        )

        ylabels = []
        for row in (real, second):
            magnitude = "" if row["magnitude"] is None else f" · mag {float(row['magnitude']):.3f}"
            ylabels.append(f"{row['label']}\n{row['kind']}{magnitude}")
        ax.set_yticks((1.0, 0.0), labels=ylabels)
        ax.set_ylim(-0.92, 1.72)
        ax.set_xlabel("UTC")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        if span_seconds <= 18 * 3_600:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d", tz=timezone.utc))
        else:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
        ax.margins(x=0.055)
        _clean_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.55)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=30)
        ax.text(0.0, 1.08, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9.5, va="bottom")
        fig.subplots_adjust(left=0.17, right=0.98, top=0.77, bottom=0.24)
        _save(fig, output_path, dpi)
    return fig, ax


def _datetime_offsets_seconds(times: Sequence[Any], maximum: Any | None) -> np.ndarray:
    parsed = [_parse_datetime(item) for item in times]
    if any(item is None for item in parsed):
        raise ValueError("angular-series timestamps cannot contain nulls")
    typed = [item for item in parsed if item is not None]
    reference = _parse_datetime(maximum) if maximum is not None else typed[len(typed) // 2]
    assert reference is not None
    return np.asarray([(item - reference).total_seconds() for item in typed], dtype=float)


def _coerce_angular_series(source: Any, fallback_label: str, fallback_color: str) -> AngularSeriesPlotData:
    if isinstance(source, AngularSeriesPlotData):
        return source
    offsets = _value(source, "time_offset_seconds", "offset_seconds", "time_seconds")
    maximum = _value(source, "maximum_utc", "maximum")
    if offsets is None:
        times = _value(source, "times_utc", "time_utc", "times")
        if times is None:
            raise ValueError(f"{fallback_label} angular series needs offsets or UTC timestamps")
        offsets = _datetime_offsets_seconds(times, maximum)
    separation = _value(
        source,
        "center_separation_deg",
        "separation_deg",
        "angular_separation_deg",
    )
    solar_diameter = _value(
        source,
        "solar_angular_diameter_deg",
        "sun_angular_diameter_deg",
        "solar_diameter_deg",
    )
    occulter_diameter = _value(
        source,
        "occulter_angular_diameter_deg",
        "moon_angular_diameter_deg",
        "angular_diameter_deg",
    )
    if any(value is None for value in (separation, solar_diameter, occulter_diameter)):
        raise ValueError(f"{fallback_label} angular series is missing separation or disk diameter data")
    color = _value(source, "color", default=fallback_color)
    return AngularSeriesPlotData(
        label=str(_value(source, "label", "body", "name", default=fallback_label)),
        time_offset_seconds=offsets,
        center_separation_deg=separation,
        solar_angular_diameter_deg=solar_diameter,
        occulter_angular_diameter_deg=occulter_diameter,
        maximum_utc=None if maximum is None else str(maximum),
        color=fallback_color if color is None else str(color),
    )


def plot_angular_geometry(
    series: AngularSeriesPlotData | Mapping[str, Any] | Sequence[Any],
    *,
    title: str = "Angular eclipse geometry",
    subtitle: str = "Topocentric apparent angles around each local maximum",
    output_path: str | PathLike[str] | None = None,
    dpi: int = 180,
) -> tuple[Figure, np.ndarray]:
    """Plot center separation/contact thresholds and apparent disk diameters."""

    if isinstance(series, Sequence) and not isinstance(series, (str, bytes, Mapping, AngularSeriesPlotData)):
        raw_series = list(series)
    else:
        raw_series = [series]
    if not raw_series:
        raise ValueError("at least one angular series is required")
    fallback = (("Real Moon", REAL_BLUE), ("Second moon", SECOND_ORANGE))
    normalized = [
        _coerce_angular_series(
            item,
            fallback[min(index, 1)][0],
            fallback[min(index, 1)][1],
        )
        for index, item in enumerate(raw_series)
    ]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(normalized),
            2,
            figsize=(13.0, 3.7 * len(normalized) + 1.2),
            squeeze=False,
            sharex="row",
        )
        for row_index, item in enumerate(normalized):
            offsets = _as_array(item.time_offset_seconds, name="time_offset_seconds")
            order = np.argsort(offsets)
            offsets_minutes = offsets[order] / 60.0
            size = len(offsets)
            separation = _broadcast_array(item.center_separation_deg, size, name="center_separation_deg")[order] * 60.0
            solar = _broadcast_array(item.solar_angular_diameter_deg, size, name="solar_angular_diameter_deg")[order] * 60.0
            occulter = _broadcast_array(item.occulter_angular_diameter_deg, size, name="occulter_angular_diameter_deg")[order] * 60.0
            external_threshold = (solar + occulter) / 2.0
            internal_threshold = np.abs(occulter - solar) / 2.0
            color = item.color or PALETTE[row_index % len(PALETTE)]

            alignment_ax, diameter_ax = axes[row_index]
            alignment_ax.plot(offsets_minutes, separation, color=INK, linewidth=2.0, label="Center separation")
            alignment_ax.plot(
                offsets_minutes,
                external_threshold,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=r"External threshold  $r_\odot+r_m$",
            )
            alignment_ax.plot(
                offsets_minutes,
                internal_threshold,
                color=color,
                linewidth=1.3,
                linestyle=":",
                label=r"Internal threshold  $|r_m-r_\odot|$",
            )
            partial = separation <= external_threshold
            central = separation <= internal_threshold
            alignment_ax.fill_between(
                offsets_minutes,
                separation,
                external_threshold,
                where=partial,
                color=color,
                alpha=0.12,
                interpolate=True,
            )
            alignment_ax.fill_between(
                offsets_minutes,
                separation,
                internal_threshold,
                where=central,
                color=color,
                alpha=0.25,
                interpolate=True,
            )
            alignment_ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            alignment_ax.set_ylabel("Angular radius / separation (arcmin)")
            alignment_ax.set_title(f"{item.label}: alignment", loc="left")
            alignment_ax.legend(loc="upper right", fontsize=8)
            _clean_axes(alignment_ax)

            diameter_ax.plot(
                offsets_minutes,
                solar,
                color=SUN_GOLD,
                linewidth=2.0,
                label="Sun",
            )
            diameter_ax.plot(
                offsets_minutes,
                occulter,
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=item.label,
            )
            diameter_ax.fill_between(
                offsets_minutes,
                solar,
                occulter,
                color=color,
                alpha=0.10,
            )
            diameter_ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            diameter_ax.set_ylabel("Apparent diameter (arcmin)")
            diameter_ax.set_title(f"{item.label}: apparent disk sizes", loc="left")
            diameter_ax.legend(loc="upper right", fontsize=8)
            _clean_axes(diameter_ax)

            for panel in (alignment_ax, diameter_ax):
                panel.set_xlabel("Minutes from local maximum")
                panel.xaxis.set_major_locator(MaxNLocator(nbins=8))

        fig.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.065, 0.952, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.89, bottom=0.10, hspace=0.48, wspace=0.22)
        _save(fig, output_path, dpi)
    return fig, axes


def _dataclass_mapping(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _coerce_stability_series(source: Any, fallback_label: str, fallback_color: str) -> StabilitySeriesPlotData:
    if isinstance(source, StabilitySeriesPlotData):
        return source
    if is_dataclass(source):
        source = _dataclass_mapping(source)
    time_years = _value(source, "time_years", "years", "integration_time_years", "time_yr")
    semimajor = _value(source, "semimajor_axis_km", "a_km", "semimajor_axis")
    eccentricity = _value(source, "eccentricity", "e")
    inclination = _value(source, "inclination_deg", "inclination", "i_deg")
    if any(value is None for value in (time_years, semimajor, eccentricity, inclination)):
        raise ValueError(f"{fallback_label} stability series is missing an orbital-element array")
    color = _value(source, "color", default=fallback_color)
    return StabilitySeriesPlotData(
        label=str(_value(source, "label", "body", "name", default=fallback_label)),
        time_years=time_years,
        semimajor_axis_km=semimajor,
        eccentricity=eccentricity,
        inclination_deg=inclination,
        color=fallback_color if color is None else str(color),
    )


def _stability_payload(data: Any) -> tuple[list[StabilitySeriesPlotData], Any, Any]:
    energy = _value(data, "relative_energy_error", "energy_error", "relative_energy_drift")
    energy_time = _value(data, "energy_time_years", "time_years", "years", "integration_time_years")
    nested = _value(data, "series", "bodies")
    if isinstance(nested, Mapping):
        common_time = _value(data, "time_years", "years", "integration_time_years")
        raw_series = []
        for label, payload in nested.items():
            if is_dataclass(payload):
                payload = _dataclass_mapping(payload)
            payload = dict(payload) if isinstance(payload, Mapping) else payload
            if isinstance(payload, dict):
                payload.setdefault("label", str(label))
                if common_time is not None:
                    payload.setdefault("time_years", common_time)
            raw_series.append(payload)
    elif nested is not None:
        raw_series = list(nested)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
        raw_series = list(data)
    else:
        raw_series = [data]
    series = [
        _coerce_stability_series(
            item,
            "Second moon" if index == 0 else f"Body {index + 1}",
            PALETTE[index % len(PALETTE)],
        )
        for index, item in enumerate(raw_series)
    ]
    return series, energy, energy_time


def _draw_threshold(ax: Axes, value: Any, label: str) -> None:
    if value is None:
        return
    values = np.asarray(value, dtype=float).reshape(-1)
    for index, threshold in enumerate(values):
        ax.axhline(
            threshold,
            color=MUTED,
            linewidth=0.85,
            linestyle=(0, (3, 3)),
            alpha=0.75,
            label=label if index == 0 else None,
        )


def plot_stability_elements(
    data: StabilitySeriesPlotData | Mapping[str, Any] | Sequence[Any] | Any,
    *,
    relative_energy_error: ArrayLike | None = None,
    energy_time_years: ArrayLike | None = None,
    thresholds: Mapping[str, Any] | None = None,
    title: str = "Numerical stability over the integration",
    subtitle: str = "Osculating Earth-centred elements; energy panel shows absolute relative error",
    output_path: str | PathLike[str] | None = None,
    dpi: int = 180,
) -> tuple[Figure, np.ndarray]:
    """Plot semimajor axis, eccentricity, inclination, and optional energy error."""

    series, embedded_energy, embedded_energy_time = _stability_payload(data)
    energy = embedded_energy if relative_energy_error is None else relative_energy_error
    energy_time = embedded_energy_time if energy_time_years is None else energy_time_years
    panel_count = 4 if energy is not None else 3
    thresholds = thresholds or {}

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            panel_count,
            1,
            figsize=(12.8, 2.45 * panel_count + 1.4),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes[:, 0]
        element_specs = (
            ("semimajor_axis_km", "Semimajor axis (km)"),
            ("eccentricity", "Eccentricity"),
            ("inclination_deg", "Inclination (deg)"),
        )
        for item_index, item in enumerate(series):
            time = _as_array(item.time_years, name=f"{item.label} time_years")
            color = item.color or PALETTE[item_index % len(PALETTE)]
            linestyle = "-" if item_index == 0 else (0, (6, 2.5))
            for ax, (field_name, _) in zip(flat_axes[:3], element_specs, strict=True):
                values = _broadcast_array(getattr(item, field_name), len(time), name=field_name)
                ax.plot(time, values, color=color, linewidth=1.35, linestyle=linestyle, label=item.label)
                if len(values):
                    ax.axhline(values[0], color=color, linewidth=0.65, alpha=0.22)

        for ax, (field_name, ylabel) in zip(flat_axes[:3], element_specs, strict=True):
            ax.set_ylabel(ylabel)
            _draw_threshold(ax, thresholds.get(field_name), "Configured limit")
            _clean_axes(ax)
        flat_axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        flat_axes[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        flat_axes[2].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        if energy is not None:
            energy_values = _as_array(energy, name="relative_energy_error")
            if energy_time is None:
                energy_times = _as_array(series[0].time_years, name="time_years")
            else:
                energy_times = _as_array(energy_time, name="energy_time_years")
            if len(energy_values) != len(energy_times):
                raise ValueError("relative energy error and its time array must have equal length")
            absolute_error = np.abs(energy_values)
            positive = absolute_error > 0.0
            energy_ax = flat_axes[3]
            if np.any(positive):
                floor = np.min(absolute_error[positive]) * 0.5
                energy_ax.plot(
                    energy_times,
                    np.maximum(absolute_error, floor),
                    color=INK,
                    linewidth=1.2,
                    label="Relative energy error",
                )
                energy_ax.set_yscale("log")
            else:
                energy_ax.plot(energy_times, absolute_error, color=INK, linewidth=1.2)
            energy_ax.set_ylabel("|ΔE / E₀|")
            _draw_threshold(energy_ax, thresholds.get("relative_energy_error"), "Configured limit")
            _clean_axes(energy_ax)

        flat_axes[-1].set_xlabel("Integration time (years)")
        handles, labels = flat_axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.97, 0.955),
                ncol=min(3, len(handles)),
                fontsize=8.7,
            )
        fig.suptitle(title, x=0.075, y=0.985, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.075, 0.953, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)
        fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.085, hspace=0.18)
        _save(fig, output_path, dpi)
    return fig, flat_axes


# Report-facing aliases with concise names.
plot_event_map = plot_world_tracks
plot_timeline = plot_contact_timeline
plot_angular_diagnostics = plot_angular_geometry
plot_stability = plot_stability_elements


__all__ = [
    "AngularSeriesPlotData",
    "StabilitySeriesPlotData",
    "TrackPlotData",
    "plot_angular_diagnostics",
    "plot_angular_geometry",
    "plot_contact_timeline",
    "plot_event_map",
    "plot_stability",
    "plot_stability_elements",
    "plot_timeline",
    "plot_world_tracks",
]
