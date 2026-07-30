"""Tests for the Pacific double-eclipse animation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chained_eclipse.pacific_double_timelapse import nonuniform_contact_times


def test_nonuniform_contact_times_preserves_boundaries_and_order() -> None:
    times = nonuniform_contact_times([1.0, 2.0, 4.0], [2, 3])

    assert times == pytest.approx([1.0, 1.5, 2.0, 2.0 + 2.0 / 3.0, 2.0 + 4.0 / 3.0, 4.0])
    assert np.all(np.diff(times) > 0.0)


@pytest.mark.parametrize(
    ("boundaries", "counts"),
    [
        ([1.0, 2.0], []),
        ([1.0, 2.0], [0]),
        ([2.0, 1.0], [1]),
    ],
)
def test_nonuniform_contact_times_rejects_invalid_inputs(
    boundaries: list[float],
    counts: list[int],
) -> None:
    with pytest.raises(ValueError):
        nonuniform_contact_times(boundaries, counts)
