from __future__ import annotations

import numpy as np
import pytest

from chained_eclipse.interpolation import CubicSpline


def test_not_a_knot_spline_reproduces_scalar_cubic_and_extrapolation() -> None:
    coordinates = np.asarray((-2.0, -0.5, 0.75, 2.5, 4.0))
    values = coordinates**3 - 2.0 * coordinates**2 + 0.5 * coordinates - 7.0
    spline = CubicSpline(coordinates, values)
    query = np.linspace(-3.0, 5.0, 41)
    expected = query**3 - 2.0 * query**2 + 0.5 * query - 7.0

    np.testing.assert_allclose(spline(query), expected, rtol=2.0e-14, atol=2.0e-13)
    np.testing.assert_allclose(
        spline(query, 1),
        3.0 * query**2 - 4.0 * query + 0.5,
        rtol=3.0e-14,
        atol=3.0e-13,
    )
    assert float(spline(1.25)) == pytest.approx(
        1.25**3 - 2.0 * 1.25**2 + 0.5 * 1.25 - 7.0,
        rel=2.0e-14,
    )


def test_not_a_knot_spline_handles_vector_values() -> None:
    coordinates = np.asarray((0.0, 1.0, 2.0, 4.0))
    values = np.column_stack(
        (
            coordinates**3,
            -2.0 * coordinates**3 + 3.0,
            0.25 * coordinates**3 - coordinates**2,
        )
    )
    spline = CubicSpline(coordinates, values, axis=0)
    query = np.asarray((0.5, 1.5, 3.0))
    expected = np.column_stack(
        (
            query**3,
            -2.0 * query**3 + 3.0,
            0.25 * query**3 - query**2,
        )
    )

    np.testing.assert_allclose(spline(query), expected, rtol=2.0e-14, atol=2.0e-13)


def test_two_point_spline_is_linear() -> None:
    spline = CubicSpline(np.asarray((1.0, 3.0)), np.asarray((2.0, 8.0)))
    np.testing.assert_allclose(
        spline(np.asarray((0.0, 2.0, 4.0))),
        (-1.0, 5.0, 11.0),
    )
