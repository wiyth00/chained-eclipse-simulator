"""Reference validation against published NASA/GSFC eclipse circumstances.

The legacy NASA pages call the uniform ephemeris time scale TDT.  TDT was
renamed Terrestrial Time (TT), so model/reference timing is compared in TT.
The pages also publish a future ``UT`` prediction derived from an assumed
Delta-T.  That value is retained in the output, but is not silently treated as
UTC: future UT1, leap seconds, and therefore UTC labels are not known exactly.

Non-central (partial) eclipses have no central line, so NASA publishes no
greatest-eclipse coordinates or central duration for them.  Such references
store ``None`` for those fields and are validated on eclipse type and TT
timing only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Sequence

from .constants import SECONDS_PER_DAY, WGS84_A_KM, WGS84_F
from .eclipse_geometry import solve_local_circumstances
from .ephemeris import EphemerisContext, enumerate_real_solar_eclipses
from .models import RealEclipse

TIMING_TARGET_SECONDS = 60.0
POSITION_TARGET_KM = 25.0


@dataclass(frozen=True, slots=True)
class PublishedEclipseReference:
    """Published circumstances at NASA's point of greatest eclipse.

    ``latitude_deg``, ``longitude_deg``, and ``central_duration_s`` are
    ``None`` for non-central (partial) eclipses: NASA publishes no
    greatest-eclipse coordinates or central duration for those, so they are
    validated on eclipse type and TT timing only.
    """

    event_id: str
    eclipse_type: str
    published_greatest_ut: str
    published_greatest_tdt: str
    published_delta_t_s: float
    latitude_deg: float | None
    longitude_deg: float | None
    central_duration_s: float | None
    source_url: str
    source_ephemeris: str = "VSOP87/ELP2000-85"
    coordinate_reference: str = "WGS84 geodetic, NASA central-path convention"
    timing_reference: str = "TDT (equivalent to TT)"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


NASA_GSFC_REFERENCES: tuple[PublishedEclipseReference, ...] = (
    PublishedEclipseReference(
        event_id="20240408_real",
        eclipse_type="total",
        published_greatest_ut="2024-04-08T18:17:18.300",
        published_greatest_tdt="2024-04-08T18:18:29.000",
        # NASA's page states Delta-T = 70.6 s while its own TDT/UT pair
        # differs by 70.7 s; both values are quoted verbatim from the source.
        published_delta_t_s=70.6,
        latitude_deg=25.286666666667,
        longitude_deg=-104.138333333333,
        central_duration_s=268.1,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2024Apr08Tbeselm.html"
        ),
    ),
    PublishedEclipseReference(
        event_id="20250329_real",
        eclipse_type="partial",
        published_greatest_ut="2025-03-29T10:47:24.700",
        published_greatest_tdt="2025-03-29T10:48:35.700",
        published_delta_t_s=71.0,
        latitude_deg=None,
        longitude_deg=None,
        central_duration_s=None,
        source_url=(
            "https://eclipse.gsfc.nasa.gov/SEbeselm/SEbeselm2001/"
            "SE2025Mar29Pbeselm.html"
        ),
    ),
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
    """One DE440s result compared with one published NASA circumstance.

    Coordinate and duration fields are ``None`` for timing-only (non-central)
    references, and ``coordinate_within_target`` is ``None`` there; ``passed``
    then rests on eclipse type and TT timing alone.
    """

    reference: PublishedEclipseReference
    model_eclipse_type: str
    model_maximum_utc: str
    model_maximum_tt: str
    model_latitude_deg: float | None
    model_longitude_deg: float | None
    signed_timing_error_tt_s: float
    timing_error_tt_s: float
    coordinate_error_wgs84_km: float | None
    model_central_duration_s: float | None
    central_duration_error_s: float | None
    nominal_utc_minus_published_ut_s: float
    type_matches: bool
    timing_within_target: bool
    coordinate_within_target: bool | None
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
    """Vincenty inverse distance on the WGS84 reference ellipsoid.

    Validation points are close together, well away from Vincenty's
    near-antipodal non-convergence case.
    """

    if latitude_1_deg == latitude_2_deg and longitude_1_deg == longitude_2_deg:
        return 0.0

    semi_major = WGS84_A_KM
    flattening = WGS84_F
    semi_minor = semi_major * (1.0 - flattening)
    phi_1 = math.radians(latitude_1_deg)
    phi_2 = math.radians(latitude_2_deg)
    longitude_delta = math.radians(
        (longitude_2_deg - longitude_1_deg + 180.0) % 360.0 - 180.0
    )
    reduced_1 = math.atan((1.0 - flattening) * math.tan(phi_1))
    reduced_2 = math.atan((1.0 - flattening) * math.tan(phi_2))
    sin_u1, cos_u1 = math.sin(reduced_1), math.cos(reduced_1)
    sin_u2, cos_u2 = math.sin(reduced_2), math.cos(reduced_2)
    longitude = longitude_delta

    for _ in range(100):
        sin_longitude = math.sin(longitude)
        cos_longitude = math.cos(longitude)
        sin_sigma = math.hypot(
            cos_u2 * sin_longitude,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_longitude,
        )
        if sin_sigma == 0.0:
            return 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_longitude
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_longitude / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
        cos_two_sigma_m = (
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            if cos_sq_alpha > 1e-16
            else 0.0
        )
        coefficient = flattening / 16.0 * cos_sq_alpha * (
            4.0 + flattening * (4.0 - 3.0 * cos_sq_alpha)
        )
        previous = longitude
        longitude = longitude_delta + (1.0 - coefficient) * flattening * sin_alpha * (
            sigma
            + coefficient
            * sin_sigma
            * (
                cos_two_sigma_m
                + coefficient
                * cos_sigma
                * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
            )
        )
        if abs(longitude - previous) < 1e-12:
            break
    else:
        raise RuntimeError("WGS84 Vincenty inverse failed to converge")

    reduced_sq = cos_sq_alpha * (
        (semi_major * semi_major - semi_minor * semi_minor) / (semi_minor * semi_minor)
    )
    series_a = 1.0 + reduced_sq / 16_384.0 * (
        4_096.0 + reduced_sq * (-768.0 + reduced_sq * (320.0 - 175.0 * reduced_sq))
    )
    series_b = reduced_sq / 1_024.0 * (
        256.0 + reduced_sq * (-128.0 + reduced_sq * (74.0 - 47.0 * reduced_sq))
    )
    delta_sigma = series_b * sin_sigma * (
        cos_two_sigma_m
        + series_b
        / 4.0
        * (
            cos_sigma * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
            - series_b
            / 6.0
            * cos_two_sigma_m
            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
            * (-3.0 + 4.0 * cos_two_sigma_m * cos_two_sigma_m)
        )
    )
    return semi_minor * series_a * (sigma - delta_sigma)


def _validate_one(
    context: EphemerisContext,
    reference: PublishedEclipseReference,
    event: RealEclipse,
) -> EclipseValidationResult:
    timing_only = reference.latitude_deg is None or reference.longitude_deg is None
    if not timing_only and (event.latitude_deg is None or event.longitude_deg is None):
        raise ValueError(f"{event.event_id} was not classified as a central eclipse")

    model_time = context.time_utc(event.maximum_utc)
    published_tt = _tt_time(context, reference.published_greatest_tdt)
    signed_timing_error = float(
        (float(model_time.tt) - float(published_tt.tt)) * SECONDS_PER_DAY
    )
    if timing_only:
        coordinate_error = None
        model_duration = None
        duration_error = None
    else:
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
        model_duration = local.central_duration_s
        duration_error = (
            model_duration - reference.central_duration_s
            if reference.central_duration_s is not None
            else None
        )
    model_utc = datetime.fromisoformat(event.maximum_utc.replace("Z", "+00:00"))
    nominal_published_ut = _parse_published_time(reference.published_greatest_ut).replace(
        tzinfo=timezone.utc
    )
    nominal_utc_minus_ut = (model_utc - nominal_published_ut).total_seconds()
    type_matches = event.eclipse_type == reference.eclipse_type
    timing_within_target = abs(signed_timing_error) < TIMING_TARGET_SECONDS
    coordinate_within_target = (
        None if coordinate_error is None else coordinate_error < POSITION_TARGET_KM
    )
    passed = (
        type_matches
        and timing_within_target
        and coordinate_within_target is not False
    )
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
        model_central_duration_s=model_duration,
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
        "non_central_reference_note": (
            "NASA publishes no greatest-eclipse coordinates or central duration for "
            "non-central (partial) eclipses; those references are validated on "
            "eclipse type and TT timing only."
        ),
        "targets": {
            "maximum_eclipse_timing_error_tt_s": TIMING_TARGET_SECONDS,
            "central_path_coordinate_error_wgs84_km": POSITION_TARGET_KM,
        },
        "all_passed": all(result.passed for result in results),
        "results": [result.to_dict() for result in results],
    }
