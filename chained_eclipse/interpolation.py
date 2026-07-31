"""Small interpolation primitives used by the trajectory cache.

The project only needs interpolation along the leading sample axis. Keeping
the not-a-knot cubic implementation here avoids importing SciPy's full
interpolation and optimization stack merely to read or build an ephemeris.
"""

from __future__ import annotations

import numpy as np


class CubicSpline:
    """One-dimensional not-a-knot cubic spline along axis zero.

    The boundary condition and extrapolation match SciPy's ``CubicSpline``
    defaults for the value-only calls used by this project.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        *,
        axis: int = 0,
    ) -> None:
        if axis != 0:
            raise ValueError("CubicSpline supports only axis=0")
        self._x = np.asarray(coordinates, dtype=float)
        self._y = np.asarray(values, dtype=float)
        if (
            self._x.ndim != 1
            or len(self._x) < 2
            or self._y.ndim < 1
            or self._y.shape[0] != len(self._x)
        ):
            raise ValueError("spline coordinates and leading value axis must match")
        if not np.all(np.isfinite(self._x)) or not np.all(np.isfinite(self._y)):
            raise ValueError("spline inputs must be finite")
        steps = np.diff(self._x)
        if np.any(steps <= 0.0):
            raise ValueError("spline coordinates must be strictly increasing")
        self._second = _not_a_knot_second_derivatives(
            self._x,
            self._y,
            steps,
        )

    def __call__(
        self,
        coordinates: float | np.ndarray,
        derivative_order: int = 0,
    ) -> np.ndarray:
        query = np.asarray(coordinates, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("spline query must be finite")
        if derivative_order not in (0, 1, 2, 3):
            raise ValueError("spline derivative order must be 0, 1, 2, or 3")
        flat = query.reshape(-1)
        indices = np.searchsorted(self._x, flat, side="right") - 1
        indices = np.clip(indices, 0, len(self._x) - 2)
        left_x = self._x[indices]
        steps = self._x[indices + 1] - left_x
        right_weight = (flat - left_x) / steps
        left_weight = 1.0 - right_weight
        trailing_dimensions = (1,) * (self._y.ndim - 1)
        left_weight = left_weight.reshape((-1, *trailing_dimensions))
        right_weight = right_weight.reshape((-1, *trailing_dimensions))
        steps = steps.reshape((-1, *trailing_dimensions))
        if derivative_order == 0:
            result = (
                left_weight * self._y[indices]
                + right_weight * self._y[indices + 1]
                + (
                    (left_weight**3 - left_weight) * self._second[indices]
                    + (right_weight**3 - right_weight)
                    * self._second[indices + 1]
                )
                * steps**2
                / 6.0
            )
        elif derivative_order == 1:
            result = (
                (self._y[indices + 1] - self._y[indices]) / steps
                + (
                    (1.0 - 3.0 * left_weight**2) * self._second[indices]
                    + (3.0 * right_weight**2 - 1.0)
                    * self._second[indices + 1]
                )
                * steps
                / 6.0
            )
        elif derivative_order == 2:
            result = (
                left_weight * self._second[indices]
                + right_weight * self._second[indices + 1]
            )
        else:
            result = (
                self._second[indices + 1] - self._second[indices]
            ) / steps
        result = result.reshape((*query.shape, *self._y.shape[1:]))
        return np.asarray(result)


def _not_a_knot_second_derivatives(
    coordinates: np.ndarray,
    values: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    count = len(coordinates)
    if count == 2:
        return np.zeros_like(values)
    slopes = np.diff(values, axis=0) / steps.reshape(
        (-1, *((1,) * (values.ndim - 1)))
    )
    if count == 3:
        curvature = 2.0 * (slopes[1] - slopes[0]) / (steps[0] + steps[1])
        return np.broadcast_to(curvature, values.shape).copy()

    interior_count = count - 2
    lower = np.empty(interior_count - 1, dtype=float)
    diagonal = np.empty(interior_count, dtype=float)
    upper = np.empty(interior_count - 1, dtype=float)
    right_hand_side = 6.0 * (slopes[1:] - slopes[:-1])

    first, second = steps[0], steps[1]
    diagonal[0] = (first + second) * (2.0 + first / second)
    upper[0] = second - first**2 / second
    for index in range(1, interior_count - 1):
        left_step = steps[index]
        right_step = steps[index + 1]
        lower[index - 1] = left_step
        diagonal[index] = 2.0 * (left_step + right_step)
        upper[index] = right_step
    penultimate, last = steps[-2], steps[-1]
    lower[-1] = penultimate - last**2 / penultimate
    diagonal[-1] = (penultimate + last) * (2.0 + last / penultimate)

    interior = _solve_tridiagonal(
        lower,
        diagonal,
        upper,
        right_hand_side,
    )
    second_derivatives = np.empty_like(values)
    second_derivatives[1:-1] = interior
    second_derivatives[0] = (
        (first + second) * interior[0] - first * interior[1]
    ) / second
    second_derivatives[-1] = (
        (penultimate + last) * interior[-1] - last * interior[-2]
    ) / penultimate
    return second_derivatives


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a real tridiagonal system with one or more right-hand sides."""

    diagonal = diagonal.copy()
    solution = np.asarray(right_hand_side, dtype=float).copy()
    for index in range(1, len(diagonal)):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        solution[index] -= factor * solution[index - 1]
    solution[-1] /= diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        solution[index] = (
            solution[index] - upper[index] * solution[index + 1]
        ) / diagonal[index]
    return solution


__all__ = ["CubicSpline"]
