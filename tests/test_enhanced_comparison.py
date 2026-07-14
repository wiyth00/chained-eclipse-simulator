"""Focused catalog-matching tests for enhanced climate comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from chained_eclipse.enhanced_comparison import compare_catalogs, write_comparison


def _event(
    body: str,
    maximum_utc: str,
    eclipse_type: str = "total",
    latitude_deg: float | None = 0.0,
    longitude_deg: float | None = 0.0,
) -> dict[str, object]:
    event: dict[str, object] = {
        "body": body,
        "maximum_utc": maximum_utc,
        "eclipse_type": eclipse_type,
    }
    if latitude_deg is not None:
        event["latitude_deg"] = latitude_deg
    if longitude_deg is not None:
        event["longitude_deg"] = longitude_deg
    return event


def _pair(
    real_maximum_utc: str,
    second_maximum_utc: str,
    separation_hours: float,
    *,
    definition_a: bool,
    definition_b: bool,
    both_total: bool,
    status: str,
) -> dict[str, object]:
    return {
        "real_maximum_utc": real_maximum_utc,
        "second_maximum_utc": second_maximum_utc,
        "separation_hours": separation_hours,
        "track_distance_km": 220.0,
        "definition_a_same_location": definition_a,
        "definition_b_500km": definition_b,
        "both_total_at_common_location": both_total,
        "status": status,
    }


def _catalogs() -> tuple[dict[str, object], dict[str, object]]:
    baseline_real = "2027-01-01T00:00:00Z"
    baseline_second = "2027-01-01T06:00:00Z"
    enhanced_real = "2027-01-01T01:00:00Z"
    enhanced_second = "2027-01-01T08:00:00Z"
    baseline: dict[str, object] = {
        "schema_version": "1.0",
        "mode": "baseline point masses",
        "start_utc": "2026-07-10T00:00:00Z",
        "end_utc": "2056-07-10T00:00:00Z",
        "trajectory": {"force_model": "Newtonian point masses"},
        "solar_events": [
            _event("real_moon", baseline_real, "total", 0.0, 179.0),
            _event("second_moon", baseline_second, "total", 20.0, -40.0),
            _event("real_moon", "2027-01-15T00:00:00Z", "partial", 50.0, 10.0),
        ],
        "lunar_events": [
            _event("real_moon", "2027-02-01T00:00:00Z", "partial", None, None)
        ],
        "temporal_solar_pairs_within_12h": [
            _pair(
                baseline_real,
                baseline_second,
                6.0,
                definition_a=True,
                definition_b=True,
                both_total=True,
                status="same-location chain",
            )
        ],
    }
    enhanced: dict[str, object] = {
        "schema_version": "1.0",
        "mode": "enhanced forces",
        "start_utc": "2026-07-10T00:00:00Z",
        "end_utc": "2056-07-10T00:00:00Z",
        "trajectory": {"force_model": {"earth_j2": True, "solar_1pn": True}},
        "solar_events": [
            _event("real_moon", enhanced_real, "annular", 0.0, -179.0),
            _event("second_moon", enhanced_second, "total", 21.0, -40.0),
            _event("real_moon", "2027-03-01T00:00:00Z", "partial", 5.0, 5.0),
        ],
        "lunar_events": [
            _event("real_moon", "2027-01-31T23:30:00Z", "total", None, None)
        ],
        "temporal_solar_pairs_within_12h": [
            _pair(
                enhanced_real,
                enhanced_second,
                7.0,
                definition_a=False,
                definition_b=True,
                both_total=False,
                status="regional pair",
            )
        ],
    }
    return baseline, enhanced


def test_comparison_reports_timing_geography_types_counts_and_chains() -> None:
    baseline, enhanced = _catalogs()
    result = compare_catalogs(baseline, enhanced, max_match_days=2.0)
    summary = result["summary"]

    assert summary["baseline_event_count"] == 4
    assert summary["enhanced_event_count"] == 4
    assert summary["matched_event_count"] == 3
    assert summary["added_event_count"] == 1
    assert summary["removed_event_count"] == 1
    assert summary["type_change_count"] == 2
    assert summary["maximum_absolute_time_shift_seconds"] == 7_200.0

    real_solar = next(
        row
        for row in result["matched_events"]
        if row["domain"] == "solar" and row["body"] == "real_moon"
    )
    assert real_solar["time_shift_seconds"] == 3_600.0
    assert real_solar["type_changed"] is True
    # The WGS84 geodesic crosses the date line instead of going 358 degrees around.
    assert real_solar["global_maximum_point_shift_km"] == pytest.approx(222.64, rel=2.0e-3)

    lunar = next(row for row in result["matched_events"] if row["domain"] == "lunar")
    assert lunar["time_shift_seconds"] == -1_800.0
    assert lunar["global_maximum_point_shift_km"] is None

    pairs = result["pair_and_chain_changes"]
    assert pairs["pair_count_delta"] == 0
    assert pairs["changed_matched_pair_count"] == 1
    assert pairs["matched_pairs"][0]["separation_shift_hours"] == 1.0
    assert pairs["chain_flag_counts"]["definition_a_same_location"]["delta"] == -1
    assert pairs["chain_flag_counts"]["definition_b_500km"]["delta"] == 0


def test_matching_is_one_to_one_and_chooses_nearest_event() -> None:
    baseline = {
        "solar_events": [
            _event("real_moon", "2030-01-01T00:00:00Z"),
            _event("real_moon", "2030-01-10T00:00:00Z"),
        ],
        "lunar_events": [],
    }
    enhanced = {
        "solar_events": [_event("real_moon", "2030-01-02T00:00:00Z")],
        "lunar_events": [],
    }

    result = compare_catalogs(baseline, enhanced, max_match_days=20.0)

    assert result["summary"]["matched_event_count"] == 1
    assert result["summary"]["removed_event_count"] == 1
    assert result["matched_events"][0]["baseline_maximum_utc"].startswith("2030-01-01")
    assert result["removed_events"][0]["maximum_utc"].startswith("2030-01-10")


def test_write_comparison_creates_strict_json_csv_and_plot(tmp_path: Path) -> None:
    baseline, enhanced = _catalogs()
    baseline_path = tmp_path / "baseline.json"
    enhanced_path = tmp_path / "enhanced.json"
    output = tmp_path / "comparison"
    baseline_path.write_text(json.dumps(baseline))
    enhanced_path.write_text(json.dumps(enhanced))

    result = write_comparison(
        baseline_path,
        enhanced_path,
        output,
        max_match_days=2.0,
    )

    assert result["summary"]["matched_event_count"] == 3
    persisted = json.loads((output / "comparison.json").read_text())
    assert persisted["sources"]["baseline_climate_json"] == str(baseline_path.resolve())
    assert "NaN" not in (output / "comparison.json").read_text()
    with (output / "matched_events.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["domain"] for row in rows} == {"solar", "lunar"}
    assert (output / "comparison.png").stat().st_size > 20_000


@pytest.mark.parametrize("maximum_days", [0.0, -1.0, float("nan")])
def test_invalid_matching_window_is_rejected(maximum_days: float) -> None:
    baseline, enhanced = _catalogs()
    with pytest.raises(ValueError, match="max_match_days"):
        compare_catalogs(baseline, enhanced, max_match_days=maximum_days)
