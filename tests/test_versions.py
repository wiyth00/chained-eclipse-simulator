"""Version single-sourcing checks.

``pyproject.toml`` is the only place the package version is written by hand.
``__version__`` must mirror the installed distribution, and ``CITATION.cff``
must be bumped in lockstep with releases. ``MODEL_VERSION`` is deliberately
independent (it tracks output-changing model revisions, not packaging), so it
is only checked for shape, never for equality with the package version.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import yaml

import chained_eclipse

CITATION_FILE = Path(__file__).resolve().parents[1] / "CITATION.cff"


def _installed_version() -> str:
    try:
        return version("chained-eclipse")
    except PackageNotFoundError:  # pragma: no cover - CI always installs
        pytest.skip("chained-eclipse is not installed")


def test_dunder_version_matches_installed_distribution() -> None:
    assert chained_eclipse.__version__ == _installed_version()


def test_citation_cff_version_matches_installed_distribution() -> None:
    citation = yaml.safe_load(CITATION_FILE.read_text(encoding="utf-8"))
    assert str(citation["version"]) == _installed_version(), (
        "CITATION.cff `version:` must be bumped in the same commit that changes "
        "the package version in pyproject.toml"
    )


def test_model_version_is_present_and_separate() -> None:
    assert isinstance(chained_eclipse.MODEL_VERSION, str)
    assert chained_eclipse.MODEL_VERSION
    # No equality assertion with __version__: the model version is allowed to
    # differ from the package version by design.
