"""Reference validation against published NASA/GSFC eclipse circumstances.

The legacy NASA pages call the uniform ephemeris time scale TDT.  TDT was
renamed Terrestrial Time (TT), so model/reference timing is compared in TT.
The pages also publish a future ``UT`` prediction derived from an assumed
Delta-T.  That value is retained in the output, but is not silently treated as
UTC: future UT1, leap seconds, and therefore UTC labels are not known exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from pyproj import Geod

from .constants import SECONDS_PER_DAY
from .eclipse_geometry import solve_local_circumstances
from .ephemeris import EphemerisContext, enumerate_real_solar_eclipses
from .models import RealEclipse

TIMING_TARGET_SECONDS = 60.0
POSITION_TARGET_KM = 25.0

_WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True, slots=True)
class PublishedEclipseReference:
    """Published circumstances at NASA's point of greatest eclipse."""

    event_id: str
    eclipse_type: str
    published_greatest_ut: str
    published_greatest_tdt: str
    published_delta_t_s: float
    latitude_deg: float
    longitude_deg: float
    central_duration_s: float
    source_url: str
    source_ephemeris: str = "VSOP87/ELP2000-85"
    coordinate_reference: str = "WGS84 geodetic, NASA central-path convention"
    timing_reference: str = "TDT (equivalent to TT)"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


NASA_GSFC_REFERENCES: tuple[PublishedEclipseReference, ...] = (
    PublishedEclipseReference(
        event_id="20260812_real",
        eclipse_type="total",
        published_greatest_ut="2026-08-12T17:45:53.800",
        published_greatest_tdt="2026-08-12T17:47:05.200",
        published_delta_t_s=71.4,
        latitude_deg=65.225,
        longitude_deg=-25.228333333333,
        central_duration_s=138.2,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2026Aug12Tbeselm.html"
        ),
    ),
    PublishedEclipseReference(
        event_id="20270206_real",
        eclipse_type="annular",
        published_greatest_ut="2027-02-06T15:59:35.700",
        published_greatest_tdt="2027-02-06T16:00:47.300",
        published_delta_t_s=71.6,
        latitude_deg=-31.303333333333,
        longitude_deg=-48.468333333333,
        central_duration_s=470.9,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2027Feb06Abeselm.html"
        ),
    ),
    PublishedEclipseReference(
        event_id="20270802_real",
        eclipse_type="total",
        published_greatest_ut="2027-08-02T10:06:37.700",
        published_greatest_tdt="2027-08-02T10:07:49.400",
        published_delta_t_s=71.7,
        latitude_deg=25.505,
        longitude_deg=33.183333333333,
        central_duration_s=382.6,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2027Aug02Tbeselm.html"
        ),
    ),
    PublishedEclipseReference(
        event_id="20280126_real",
        eclipse_type="annular",
        published_greatest_ut="2028-01-26T15:07:46.500",
        published_greatest_tdt="2028-01-26T15:08:58.400",
        published_delta_t_s=71.9,
        latitude_deg=2.958333333333,
        longitude_deg=-51.561666666667,
        central_duration_s=627.1,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2028Jan26Abeselm.html"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class EclipseValidationResult:
    """One DE440s result compared with one published NASA circumstance."""

    reference: PublishedEclipseReference
    model_eclipse_type: str
    model_maximum_utc: str
    model_maximum_tt: str
    model_latitude_deg: float
    model_longitude_deg: float
    signed_timing_error_tt_s: float
    timing_error_tt_s: float
    coordinate_error_wgs84_km: float
    model_central_duration_s: float
    central_duration_error_s: float
    nominal_utc_minus_published_ut_s: float
    type_matches: bool
    timing_within_target: bool
    coordinate_within_target: bool
    passed: bool
    model_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_published_time(value: str) -> datetime:
    """Parse a timezone-free NASA wall-clock label without changing its scale."""

    return datetime.fromisoformat(value)


def _tt_time(context: EphemerisContext, value: str):
    calendar = _parse_published_time(value)
    seconds = calendar.second + calendar.microsecond / 1_000_000.0
    return context.timescale.tt(
        calendar.year,
        calendar.month,
        calendar.day,
        calendar.hour,
        calendar.minute,
        seconds,
    )


def _format_tt(time) -> str:
    year, month, day, hour, minute, seconds = time.tt_calendar()
    start = datetime(int(year), int(month), int(day), int(hour), int(minute))
    value = start + timedelta(seconds=float(seconds))
    return value.isoformat(timespec="milliseconds") + " TT"


def wgs84_geodesic_distance_km(
    latitude_1_deg: float,
    longitude_1_deg: float,
    latitude_2_deg: float,
    longitude_2_deg: float,
) -> float:
    """Geodesic inverse distance on the WGS84 reference ellipsoid.

    ``pyproj.Geod`` (Karney's algorithm) is the project's canonical geodesic
    implementation, shared with ``search.py`` and ``sensitivity.py``.  The
    hand-rolled Vincenty inverse that previously lived here is preserved in
    ``tests/test_numeric_crosschecks.py`` as an independent reference, where
    the two implementations are asserted to agree to sub-millimetre level
    over validation-scale point pairs.
    """

    if latitude_1_deg == latitude_2_deg and longitude_1_deg == longitude_2_deg:
        return 0.0
    _, _, distance_m = _WGS84_GEOD.inv(
        longitude_1_deg,
        latitude_1_deg,
        longitude_2_deg,
        latitude_2_deg,
    )
    return float(distance_m) / 1_000.0


def _validate_one(
    context: EphemerisContext,
    reference: PublishedEclipseReference,
    event: RealEclipse,
) -> EclipseValidationResult:
    if event.latitude_deg is None or event.longitude_deg is None:
        raise ValueError(f"{event.event_id} was not classified as a central eclipse")

    model_time = context.time_utc(event.maximum_utc)
    published_tt = _tt_time(context, reference.published_greatest_tdt)
    signed_timing_error = float(
        (float(model_time.tt) - float(published_tt.tt)) * SECONDS_PER_DAY
    )
    coordinate_error = wgs84_geodesic_distance_km(
        reference.latitude_deg,
        reference.longitude_deg,
        event.latitude_deg,
        event.longitude_deg,
    )
    local = solve_local_circumstances(
        context,
        float(model_time.tt),
        reference.latitude_deg,
        reference.longitude_deg,
        "real_moon",
        bracket_step_seconds=30.0,
    )
    duration_error = local.central_duration_s - reference.central_duration_s
    model_utc = datetime.fromisoformat(event.maximum_utc.replace("Z", "+00:00"))
    nominal_published_ut = _parse_published_time(reference.published_greatest_ut).replace(
        tzinfo=timezone.utc
    )
    nominal_utc_minus_ut = (model_utc - nominal_published_ut).total_seconds()
    type_matches = event.eclipse_type == reference.eclipse_type
    timing_within_target = abs(signed_timing_error) < TIMING_TARGET_SECONDS
    coordinate_within_target = coordinate_error < POSITION_TARGET_KM
    passed = type_matches and timing_within_target and coordinate_within_target
    return EclipseValidationResult(
        reference=reference,
        model_eclipse_type=event.eclipse_type,
        model_maximum_utc=event.maximum_utc,
        model_maximum_tt=_format_tt(model_time),
        model_latitude_deg=event.latitude_deg,
        model_longitude_deg=event.longitude_deg,
        signed_timing_error_tt_s=signed_timing_error,
        timing_error_tt_s=abs(signed_timing_error),
        coordinate_error_wgs84_km=coordinate_error,
        model_central_duration_s=local.central_duration_s,
        central_duration_error_s=duration_error,
        nominal_utc_minus_published_ut_s=nominal_utc_minus_ut,
        type_matches=type_matches,
        timing_within_target=timing_within_target,
        coordinate_within_target=coordinate_within_target,
        passed=passed,
        model_source=event.source,
    )


def validate_known_eclipses(
    context: EphemerisContext,
    references: Sequence[PublishedEclipseReference] = NASA_GSFC_REFERENCES,
) -> list[EclipseValidationResult]:
    """Numerically locate the referenced eclipses and return structured comparisons."""

    if not references:
        return []
    published_ut = [_parse_published_time(item.published_greatest_ut) for item in references]
    search_start = min(published_ut).replace(tzinfo=timezone.utc) - timedelta(days=2)
    search_end = max(published_ut).replace(tzinfo=timezone.utc) + timedelta(days=2)
    events = enumerate_real_solar_eclipses(
        context,
        context.time_utc(search_start.isoformat()),
        context.time_utc(search_end.isoformat()),
    )
    events_by_id = {event.event_id: event for event in events}
    missing = [item.event_id for item in references if item.event_id not in events_by_id]
    if missing:
        raise LookupError(f"DE440s search did not recover known eclipses: {', '.join(missing)}")
    return [_validate_one(context, item, events_by_id[item.event_id]) for item in references]


def build_validation_report(
    context: EphemerisContext,
    references: Sequence[PublishedEclipseReference] = NASA_GSFC_REFERENCES,
) -> dict[str, object]:
    """Return a JSON-ready validation report with time-scale provenance."""

    results = validate_known_eclipses(context, references)
    return {
        "schema_version": "1.0",
        "reference_authority": "NASA/GSFC Five Millennium Canon eclipse pages",
        "comparison_basis": (
            "Model TT versus published TDT (the historical name for TT); "
            "WGS84 geodesic distance at greatest eclipse"
        ),
        "published_ut_note": (
            "NASA's published UT is a future UT1 prediction based on its listed Delta-T. "
            "The model UTC label is reported but UTC-versus-UT is not the timing pass/fail basis."
        ),
        "targets": {
            "maximum_eclipse_timing_error_tt_s": TIMING_TARGET_SECONDS,
            "central_path_coordinate_error_wgs84_km": POSITION_TARGET_KM,
        },
        "all_passed": all(result.passed for result in results),
        "results": [result.to_dict() for result in results],
    }
