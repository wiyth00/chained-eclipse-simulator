"""Focused output and content checks for the enhanced technical report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chained_eclipse.enhanced_report import generate_enhanced_report


def _climate() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mode": "30-year coupled enhanced eclipse climate",
        "dynamics_model": "enhanced",
        "start_utc": "2026-07-10T00:00:00Z",
        "end_utc": "2056-07-10T00:00:00Z",
        "summary": {
            "solar_event_counts": {
                "real_moon": {"total": 2, "partial": 1},
                "second_moon": {"total": 4},
            },
            "lunar_event_counts": {
                "real_moon": {"total": 1},
                "second_moon": {"partial": 2},
            },
            "inclination_ranges_deg": {
                "real_moon": {"minimum": 0.9, "maximum": 5.4},
                "second_moon": {"minimum": 4.2, "maximum": 19.5},
            },
            "temporal_solar_pair_count_within_12h": 1,
            "definition_a_same_location_pair_count": 1,
            "definition_b_500km_pair_count": 1,
        },
        "trajectory": {
            "integrator": "REBOUND IAS15",
            "sample_step_seconds": 3600.0,
            "planetary_perturbers": {
                "bodies": ["mercury", "venus", "mars", "jupiter"],
            },
            "force_model": {
                "newtonian": "Fully coupled Newtonian active bodies",
                "earth_tides_and_spin": {
                    "enabled": True,
                    "model": "constant-time-lag equilibrium tide",
                    "calibration_target": "38.2 mm/year lunar recession",
                },
                "earth_j2_and_first_post_newtonian": {
                    "earth_j2": {
                        "enabled": True,
                        "j2": 0.00108262668,
                        "equatorial_radius_km": 6378.137,
                    },
                    "first_post_newtonian": {"enabled": True},
                },
            },
            "alternate_earth_rotation": {
                "final_delta_mean_solar_lod_ms": 0.622711,
                "final_delta_ut1_s": -3.414723,
                "final_longitude_shift_deg": 0.014267,
                "maximum_abs_longitude_shift_deg": 0.014267,
                "final_pole_separation_arcsec": 135.324,
                "maximum_pole_separation_arcsec": 135.325,
            },
            "omissions": ["lunar permanent-figure forces"],
        },
        "temporal_solar_pairs_within_12h": [
            {
                "real_maximum_utc": "2026-08-12T02:30:00Z",
                "second_maximum_utc": "2026-08-12T00:30:00Z",
                "separation_hours": 2.0,
                "local_maximum_separation_hours": 1.5,
                "track_distance_km": 9.8,
                "best_common_latitude_deg": 82.686,
                "best_common_longitude_deg": -98.555,
                "definition_a_same_location": True,
                "definition_b_500km": True,
                "both_total_at_common_location": True,
                "real_type": "total",
                "second_type": "total",
                "real_local": {
                    "eclipse_type": "total",
                    "maximum_utc": "2026-08-12T02:20:00Z",
                },
                "second_local": {
                    "eclipse_type": "total",
                    "maximum_utc": "2026-08-12T00:50:00Z",
                },
            }
        ],
        "limitations": ["DE440 supplies epoch states only"],
    }


def _comparison() -> dict[str, object]:
    return {
        "summary": {
            "baseline_event_count": 11,
            "enhanced_event_count": 10,
            "event_count_delta": -1,
            "matched_event_count": 9,
            "added_event_count": 1,
            "removed_event_count": 2,
            "type_change_count": 3,
            "median_absolute_time_shift_seconds": 3600.0,
            "maximum_absolute_time_shift_seconds": 90000.0,
            "median_global_maximum_point_shift_km": 42.5,
            "maximum_global_maximum_point_shift_km": 880.2,
        },
        "matching": {"max_match_days": 10.0},
        "event_counts": {
            "by_domain_and_body": [
                {
                    "domain": "solar",
                    "body": "real_moon",
                    "baseline": 4,
                    "enhanced": 3,
                    "delta": -1,
                }
            ]
        },
        "pair_and_chain_changes": {
            "baseline_pair_count": 1,
            "enhanced_pair_count": 1,
        },
        "limitations": ["Maximum-point displacement is not full-track distance"],
    }


def test_report_writes_result_first_markdown_and_portable_html(tmp_path: Path) -> None:
    climate_path = tmp_path / "climate.json"
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    comparison_path = comparison_dir / "comparison.json"
    validation_path = tmp_path / "validation.json"
    convergence_path = tmp_path / "convergence.json"
    trajectory_convergence_path = tmp_path / "trajectory_convergence.json"
    climate_path.write_text(json.dumps(_climate()))
    comparison_path.write_text(json.dumps(_comparison()))
    validation_path.write_text(
        json.dumps(
            {
                "all_passed": True,
                "reference_authority": "Published eclipse canon",
                "results": [
                    {
                        "reference": {"event_id": "known-eclipse"},
                        "model_eclipse_type": "total",
                        "timing_error_tt_s": 0.6,
                        "coordinate_error_wgs84_km": 0.7,
                        "central_duration_error_s": 2.0,
                        "passed": True,
                    }
                ],
            }
        )
    )
    convergence_path.write_text(
        json.dumps(
            {
                "detector_cadence_comparison": {
                    "baseline_detector_step_seconds": 600,
                    "comparison_detector_step_seconds": 300,
                    "solar_type_mismatches": 0,
                    "lunar_type_mismatches": 0,
                    "maximum_axis_time_delta_seconds": {"real_moon_solar": 0.015},
                },
                "interpretation": "Central eclipses converge at tested cadences.",
            }
        )
    )
    trajectory_convergence_path.write_text(
        json.dumps(
            {
                "summary": {
                    "baseline_event_count": 10,
                    "enhanced_event_count": 10,
                    "matched_event_count": 10,
                    "type_change_count": 0,
                    "maximum_absolute_time_shift_seconds": 0.013,
                    "maximum_global_maximum_point_shift_km": 0.012,
                }
            }
        )
    )
    (tmp_path / "eclipse_climate.png").write_bytes(b"image-placeholder")
    (comparison_dir / "comparison.png").write_bytes(b"image-placeholder")
    (comparison_dir / "matched_events.csv").write_text("domain,body\n")

    result = generate_enhanced_report(
        climate_path,
        comparison_path,
        validation_path=validation_path,
        convergence_path=[convergence_path, trajectory_convergence_path],
    )

    assert result["total_events"] == 10
    assert result["double_totality_pair_count"] == 1
    assert result["headline"].index("Second moon") < result["headline"].index("Real Moon")
    markdown = (tmp_path / "report.md").read_text()
    html = (tmp_path / "report.html").read_text()
    assert markdown.startswith("# Enhanced two-moon eclipse climate")
    assert "Headline result" in markdown.splitlines()[4]
    assert "Earth rotation changes enough" in markdown
    assert "-3.414723 s" in markdown
    assert "Published real-Moon eclipse validation: all checks passed" in markdown
    assert "maximum saved axis-time difference 0.015000 s" in markdown
    assert "10 versus 10 events, 0 type changes" in markdown
    assert "![Thirty-year eclipse climate](eclipse_climate.png)" in markdown
    assert "![Baseline comparison](comparison/comparison.png)" in markdown
    assert html.startswith("<!doctype html>")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert 'src="eclipse_climate.png"' in html
    assert 'src="comparison/comparison.png"' in html
    assert 'href="climate.json"' in html
    assert 'href="comparison/matched_events.csv"' in html
    assert str(tmp_path) not in html


def test_report_rejects_a_baseline_catalog(tmp_path: Path) -> None:
    climate = _climate()
    climate["dynamics_model"] = "baseline"
    climate_path = tmp_path / "climate.json"
    comparison_path = tmp_path / "comparison.json"
    climate_path.write_text(json.dumps(climate))
    comparison_path.write_text(json.dumps(_comparison()))

    with pytest.raises(ValueError, match="dynamics_model='enhanced'"):
        generate_enhanced_report(climate_path, comparison_path)
