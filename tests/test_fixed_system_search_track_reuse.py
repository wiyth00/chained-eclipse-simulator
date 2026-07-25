"""Regression test for the real-Moon centerline hoist in ``fixed_system_search``.

The real-Moon central track depends only on ``real_tt``, so it must be built
once per real eclipse, not once per second-moon candidate. On ``main`` the
``generate_central_track(..., "real_moon", ...)`` call sat inside the inner
``for second_event`` loop, so this test sees three real-Moon builds instead of
one and fails.

Everything below the search loop is stubbed so the test runs without an
ephemeris kernel: ``minimum_track_distance_km`` reports a distance beyond the
4,000 km cutoff, which makes every candidate ``continue`` before any local
circumstances are solved.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chained_eclipse import search as search_module

SECOND_MOON_CANDIDATES = 3


def _fake_track(tt_jd: float) -> dict[str, np.ndarray]:
    return {
        "tt_jd": np.asarray([tt_jd]),
        "latitude_deg": np.asarray([0.0]),
        "longitude_deg": np.asarray([0.0]),
    }


@pytest.fixture
def recorded_bodies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the ``body`` argument of every ``generate_central_track`` call."""

    bodies: list[str] = []

    def fake_generate_central_track(context, tt_jd, body, **kwargs):
        bodies.append(body)
        return _fake_track(float(tt_jd))

    def fake_find_hypothetical_eclipses_near(context, trajectory, center_tt_jd, **kwargs):
        return [
            search_module.HypotheticalEclipse(
                maximum_tt_jd=center_tt_jd + 0.01 * (index + 1),
                eclipse_type="total",
                latitude_deg=0.0,
                longitude_deg=0.0,
                penumbra_margin_km=100.0,
                core_radius_km=10.0,
            )
            for index in range(SECOND_MOON_CANDIDATES)
        ]

    monkeypatch.setattr(search_module, "generate_central_track", fake_generate_central_track)
    monkeypatch.setattr(
        search_module, "find_hypothetical_eclipses_near", fake_find_hypothetical_eclipses_near
    )
    # Beyond the 4,000 km cutoff, so each candidate is rejected before any
    # ephemeris-backed local-circumstances solve is attempted.
    monkeypatch.setattr(
        search_module, "minimum_track_distance_km", lambda first, second: (9_999.0, 0, 0)
    )
    return bodies


def test_real_moon_track_is_built_once_per_real_eclipse(recorded_bodies: list[str]) -> None:
    context = SimpleNamespace(time_utc=lambda value: SimpleNamespace(tt=2_461_000.5))
    trajectory = SimpleNamespace(position=lambda tt_jd: np.zeros(3))
    real_eclipse = SimpleNamespace(
        event_id="2026-08-12_real", maximum_utc="2026-08-12T15:01:19Z"
    )

    candidates, tracks = search_module.fixed_system_search(
        context, [real_eclipse], trajectory
    )

    assert candidates == []
    assert tracks == {}
    assert recorded_bodies.count("real_moon") == 1, (
        "the real-Moon centerline depends only on real_tt and must be built once "
        "per real eclipse, not once per second-moon candidate"
    )
    assert recorded_bodies.count("second_moon") == SECOND_MOON_CANDIDATES


def test_no_real_moon_track_when_there_are_no_candidates(
    monkeypatch: pytest.MonkeyPatch, recorded_bodies: list[str]
) -> None:
    monkeypatch.setattr(
        search_module, "find_hypothetical_eclipses_near", lambda *args, **kwargs: []
    )
    context = SimpleNamespace(time_utc=lambda value: SimpleNamespace(tt=2_461_000.5))
    trajectory = SimpleNamespace(position=lambda tt_jd: np.zeros(3))
    real_eclipse = SimpleNamespace(
        event_id="2026-08-12_real", maximum_utc="2026-08-12T15:01:19Z"
    )

    candidates, tracks = search_module.fixed_system_search(
        context, [real_eclipse], trajectory
    )

    assert candidates == []
    assert tracks == {}
    assert recorded_bodies == []
