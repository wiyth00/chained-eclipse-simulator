"""Configuration and scenario tests for coupled structured-body tides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from chained_eclipse.rotational_dynamics import (
    ALL_ACTIVE_MASSIVE_BODIES,
    NBodyTideConfig,
    PermanentFigureConfig,
    RotationalBodyConfig,
    rotational_tide_config_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload():
    return yaml.safe_load(
        (ROOT / "config" / "bound_binary_giant.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_bound_binary_resolves_three_explicit_structured_bodies() -> None:
    config = rotational_tide_config_from_mapping(_payload())
    assert config is not None

    bodies = config.resolved_bodies()
    assert config.active_scenario == "nominal"
    assert config.interaction_scope == ALL_ACTIVE_MASSIVE_BODIES
    assert [body.name for body in bodies] == [
        "earth",
        "real_moon",
        "second_moon",
    ]
    assert all(body.initial_spin_rate_rad_s > 0.0 for body in bodies)
    assert bodies[2].scenario_label == "nominal_rocky_mutual_synchronous"
    assert bodies[2].permanent_figure.enabled is False
    json.dumps(config.to_dict(), allow_nan=False)


def test_giant_sensitivity_scenarios_are_ordered_and_labeled() -> None:
    payload = _payload()
    products = []
    rates = []
    for scenario in ("low", "nominal", "high"):
        payload["rotational_tides"]["active_scenario"] = scenario
        config = rotational_tide_config_from_mapping(payload)
        assert config is not None
        giant = next(
            body
            for body in config.resolved_bodies()
            if body.name == "second_moon"
        )
        products.append(giant.love_number_k2 * giant.constant_time_lag_s)
        rates.append(giant.initial_spin_rate_rad_s)
        assert scenario in giant.scenario_label
        assert "no measured" in giant.provenance.lower() or scenario == "high"

    assert products[0] < products[1] < products[2]
    assert rates[0] < rates[1] < rates[2]


def test_zero_love_number_and_zero_lag_are_valid_disabled_limits() -> None:
    body = RotationalBodyConfig(
        name="second_moon",
        radius_km=2_514.0,
        polar_moment_factor=0.4,
        initial_spin_vector_rad_s=(0.0, 0.0, 1.0e-5),
        love_number_k2=0.0,
        constant_time_lag_s=0.0,
        scenario_label="disabled_limit",
        provenance="equation-level test",
    )
    config = NBodyTideConfig(structured_bodies=(body,))

    config.validate()


def test_unknown_scenario_field_is_rejected() -> None:
    payload = _payload()
    payload["rotational_tides"]["scenarios"]["nominal"]["surprise"] = True

    with pytest.raises(ValueError, match="unknown rotational-tide scenario"):
        rotational_tide_config_from_mapping(payload)


def test_reaction_incomplete_permanent_figure_cannot_be_enabled() -> None:
    figure = PermanentFigureConfig(
        enabled=True,
        j2=2.0e-4,
        c22=2.0e-5,
        scenario_label="test",
        provenance="test",
    )

    with pytest.raises(ValueError, match="reaction-aware attitude backend"):
        figure.validate()


def test_active_tides_cannot_drop_the_spin_reaction() -> None:
    with pytest.raises(ValueError, match="equal-and-opposite spin reaction"):
        NBodyTideConfig(enabled=True, evolve_spin=False).validate()

    NBodyTideConfig(enabled=False, evolve_spin=False).validate()
