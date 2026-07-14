"""Model-selection checks for the standout eclipse-track renderer."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import yaml

from chained_eclipse import climate_tracks


def _write_inputs(tmp_path, *, dynamics_model: str | None):
    catalog_dir = tmp_path / ("enhanced" if dynamics_model == "enhanced" else "baseline")
    catalog_dir.mkdir()
    climate = {
        "end_utc": "2026-07-11T00:00:00Z",
        "trajectory": {"sample_step_seconds": 3600.0},
        "notable_solar_events": [],
    }
    if dynamics_model is not None:
        climate["dynamics_model"] = dynamics_model
    climate_path = catalog_dir / "climate.json"
    climate_path.write_text(json.dumps(climate), encoding="utf-8")
    config_path = tmp_path / "system.yaml"
    config_path.write_text(yaml.safe_dump({"orbital_elements": {}}), encoding="utf-8")
    return climate_path, config_path


def test_legacy_catalog_uses_baseline_and_historical_default_directory(
    tmp_path, monkeypatch
) -> None:
    climate_path, config_path = _write_inputs(tmp_path, dynamics_model=None)
    constructions = []
    plotted = []

    class FakeBaselineEphemeris:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

    monkeypatch.setattr(climate_tracks, "CoupledEphemeris", FakeBaselineEphemeris)
    monkeypatch.setattr(climate_tracks, "load_ephemeris", lambda *_: object())
    monkeypatch.setattr(
        climate_tracks,
        "_plot_standout_tracks",
        lambda events, output_path, *, title: plotted.append((events, output_path, title)),
    )

    payload = climate_tracks.build_standout_tracks(
        climate_path=climate_path,
        config_path=config_path,
    )

    assert len(constructions) == 1
    assert constructions[0][1]["sample_step_seconds"] == 3600.0
    assert payload["dynamics_model"] == "baseline"
    assert payload["title"].endswith("Baseline dynamics")
    assert plotted[0][1] == climate_path.parent / "standout_tracks.png"
    assert (climate_path.parent / "standout_tracks.json").exists()


def test_enhanced_catalog_lazily_selects_enhanced_ephemeris(tmp_path, monkeypatch) -> None:
    climate_path, config_path = _write_inputs(tmp_path, dynamics_model="enhanced")
    constructions = []
    plotted = []

    class FakeEnhancedEphemeris:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

    fake_module = ModuleType("chained_eclipse.enhanced_ephemeris")
    fake_module.EnhancedEphemeris = FakeEnhancedEphemeris
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(climate_tracks, "load_ephemeris", lambda *_: object())
    monkeypatch.setattr(
        climate_tracks,
        "_plot_standout_tracks",
        lambda events, output_path, *, title: plotted.append((events, output_path, title)),
    )

    payload = climate_tracks.build_standout_tracks(
        climate_path=climate_path,
        config_path=config_path,
        per_body=3,
    )

    assert len(constructions) == 1
    assert payload["dynamics_model"] == "enhanced"
    assert payload["title"] == (
        "3 longest total-solar-eclipse tracks from each moon · Enhanced dynamics"
    )
    assert plotted[0][2] == payload["title"]
    assert plotted[0][1].parent == climate_path.parent
