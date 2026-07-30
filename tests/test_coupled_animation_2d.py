"""Fast tests for the coupled world-map animation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chained_eclipse.coupled_animation_2d import (
    _break_dateline,
    animation_window_utc,
)


def _result() -> tuple[dict[str, object], dict[str, object]]:
    pair = {
        "real_maximum_utc": "2027-06-22T17:57:03Z",
        "second_maximum_utc": "2027-06-22T20:44:49Z",
    }
    result = {
        "real_moon_events": [
            {
                "axis_maximum_utc": pair["real_maximum_utc"],
                "global_start_utc": "2027-06-22T10:28:35Z",
                "global_end_utc": "2027-06-23T03:06:51Z",
            }
        ],
        "second_moon_events": [
            {
                "axis_maximum_utc": pair["second_maximum_utc"],
                "global_start_utc": "2027-06-22T17:46:40Z",
                "global_end_utc": "2027-06-22T23:42:32Z",
            }
        ],
    }
    return result, pair


def test_animation_window_spans_both_global_events() -> None:
    result, pair = _result()

    assert animation_window_utc(result, pair) == (
        "2027-06-22T10:28:35Z",
        "2027-06-23T03:06:51Z",
    )


def test_animation_window_rejects_negative_padding() -> None:
    result, pair = _result()

    with pytest.raises(ValueError, match="non-negative"):
        animation_window_utc(result, pair, lead_minutes=-1.0)


def test_break_dateline_inserts_gap_without_moving_points() -> None:
    longitude, latitude = _break_dateline(
        np.asarray((170.0, 179.0, -179.0, -160.0)),
        np.asarray((10.0, 11.0, 12.0, 13.0)),
    )

    np.testing.assert_allclose(
        longitude,
        np.asarray((170.0, 179.0, np.nan, -179.0, -160.0)),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        latitude,
        np.asarray((10.0, 11.0, np.nan, 12.0, 13.0)),
        equal_nan=True,
    )
