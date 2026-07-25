"""Guard that third-party imports used by the package are declared dependencies.

``chained_eclipse.search`` and ``chained_eclipse.sensitivity`` both do
``from pyproj import Geod``. Before this test, ``pyproj`` was installed only as a
transitive dependency of ``cartopy``; if cartopy ever dropped or vendored it,
these imports would fail at runtime with no declared-dependency safety net.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.parametrize(
    "module_name",
    ["chained_eclipse.search", "chained_eclipse.sensitivity"],
)
def test_pyproj_dependent_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.skipif(not PYPROJECT.exists(), reason="running from an installed wheel")
def test_pyproj_is_a_declared_runtime_dependency() -> None:
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared = metadata["project"]["dependencies"]
    assert any(spec.split(">")[0].split("=")[0].strip() == "pyproj" for spec in declared), (
        "pyproj is imported by chained_eclipse.search and chained_eclipse.sensitivity "
        "and must be declared in [project] dependencies, not relied on transitively."
    )
