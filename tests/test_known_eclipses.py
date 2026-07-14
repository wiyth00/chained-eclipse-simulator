"""Numerical reference tests for published real-Moon solar eclipses."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from chained_eclipse.ephemeris import load_ephemeris
from chained_eclipse.validation import (
    NASA_GSFC_REFERENCES,
    POSITION_TARGET_KM,
    TIMING_TARGET_SECONDS,
    build_validation_report,
    validate_known_eclipses,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_IDS = tuple(item.event_id for item in NASA_GSFC_REFERENCES)


@pytest.fixture(scope="module")
def validation_results():
    context = load_ephemeris(PROJECT_ROOT / "data" / "ephemeris")
    return validate_known_eclipses(context)


@pytest.fixture(scope="module")
def results_by_id(validation_results):
    return {result.reference.event_id: result for result in validation_results}


def test_reference_fixtures_preserve_nasa_time_scales_and_sources() -> None:
    """The published TDT/UT difference must reproduce NASA's stated Delta-T."""

    assert len(NASA_GSFC_REFERENCES) >= 3
    for reference in NASA_GSFC_REFERENCES:
        published_ut = datetime.fromisoformat(reference.published_greatest_ut)
        published_tdt = datetime.fromisoformat(reference.published_greatest_tdt)
        assert (published_tdt - published_ut).total_seconds() == pytest.approx(
            reference.published_delta_t_s,
            abs=0.051,
        )
        assert reference.timing_reference == "TDT (equivalent to TT)"
        assert reference.coordinate_reference.startswith("WGS84")
        assert reference.source_url.startswith("https://eclipse.gsfc.nasa.gov/")


@pytest.mark.parametrize("event_id", EVENT_IDS)
def test_de440s_greatest_eclipse_timing_in_tt(results_by_id, event_id: str) -> None:
    """Compare physical ephemeris time in TT, not future UT1 mislabeled as UTC."""

    result = results_by_id[event_id]
    assert result.timing_error_tt_s < TIMING_TARGET_SECONDS
    assert result.timing_within_target


@pytest.mark.parametrize("event_id", EVENT_IDS)
def test_de440s_central_point_within_25_km(results_by_id, event_id: str) -> None:
    result = results_by_id[event_id]
    assert result.coordinate_error_wgs84_km < POSITION_TARGET_KM
    assert result.coordinate_within_target


@pytest.mark.parametrize("event_id", EVENT_IDS)
def test_de440s_eclipse_type_matches_nasa(results_by_id, event_id: str) -> None:
    result = results_by_id[event_id]
    assert result.model_eclipse_type == result.reference.eclipse_type
    assert result.type_matches
    assert result.passed


def test_structured_report_exposes_reference_provenance() -> None:
    context = load_ephemeris(PROJECT_ROOT / "data" / "ephemeris")
    report = build_validation_report(context, NASA_GSFC_REFERENCES[:3])
    assert report["all_passed"] is True
    assert "TT versus published TDT" in report["comparison_basis"]
    assert "not the timing pass/fail basis" in report["published_ut_note"]
    for item in report["results"]:
        reference = item["reference"]
        assert reference["published_greatest_ut"]
        assert reference["published_greatest_tdt"]
        assert reference["source_url"].startswith("https://eclipse.gsfc.nasa.gov/")
