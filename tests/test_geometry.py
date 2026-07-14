"""Unit and end-to-end checks for apparent disks, contacts, and tracks."""

from pathlib import Path

import numpy as np
import pytest

from chained_eclipse.eclipse_geometry import (
    angular_separation,
    disk_overlap_fraction,
    generate_central_track,
    solve_local_circumstances,
)
from chained_eclipse.ephemeris import load_ephemeris

ROOT = Path(__file__).resolve().parents[1]


def test_angular_separation_is_stable_at_zero_and_right_angle() -> None:
    assert angular_separation(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.0
    assert angular_separation(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    ) == pytest.approx(np.pi / 2.0)


def test_disk_overlap_handles_containment_and_disjoint_cases() -> None:
    assert disk_overlap_fraction(3.0, 1.0, 1.0) == 0.0
    assert disk_overlap_fraction(0.0, 1.0, 1.1) == 1.0
    assert disk_overlap_fraction(0.0, 1.0, 0.5) == pytest.approx(0.25)
    assert 0.0 < disk_overlap_fraction(1.0, 1.0, 1.0) < 1.0


def test_2026_local_contacts_are_total_and_physically_ordered() -> None:
    context = load_ephemeris(ROOT / "data" / "ephemeris")
    approximate = float(context.time_utc("2026-08-12T17:45:56Z").tt)
    circumstances = solve_local_circumstances(
        context,
        approximate,
        65.225,
        -25.228333333333,
        "real_moon",
        bracket_step_seconds=30.0,
    )
    assert circumstances.eclipse_type == "total"
    assert circumstances.obscuration == 1.0
    assert 130.0 < circumstances.central_duration_s < 150.0
    assert circumstances.c1_utc < circumstances.c2_utc < circumstances.maximum_utc
    assert circumstances.maximum_utc < circumstances.c3_utc < circumstances.c4_utc
    track = generate_central_track(
        context, approximate, "real_moon", half_window_hours=2.0, step_seconds=300.0
    )
    assert len(track["tt_jd"]) > 10
    assert np.all(np.isfinite(track["latitude_deg"]))

