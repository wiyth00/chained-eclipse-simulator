"""CLI flag, version-string, and resume-guard tests (no ephemeris kernel needed)."""

from __future__ import annotations

import os
import subprocess
import sys

from chained_eclipse import __version__
from chained_eclipse.cli import (
    _resume_identity_mismatches,
    _version_string,
    build_parser,
)
from chained_eclipse.constants import MODEL_VERSION


def test_parser_defaults_include_new_flags() -> None:
    args = build_parser().parse_args([])
    assert args.mode == "full"
    assert args.resume is False
    assert args.allow_unverified_ephemeris is False
    assert args.quiet is False
    assert args.verbose is False


def test_parser_accepts_new_flags() -> None:
    args = build_parser().parse_args(
        ["--mode", "full", "--resume", "--allow-unverified-ephemeris", "--verbose"]
    )
    assert args.resume is True
    assert args.allow_unverified_ephemeris is True
    assert args.verbose is True


def test_parser_rejects_quiet_with_verbose() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--quiet", "--verbose"])


def test_version_string_mentions_package_model_and_kernel() -> None:
    text = _version_string()
    assert __version__ in text
    assert MODEL_VERSION in text
    # Pins the informational kernel name to load_ephemeris's default kernel.
    assert "de440s.bsp" in text


def test_version_flag_via_subprocess() -> None:
    environment = dict(os.environ, MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, "-m", "chained_eclipse.cli", "--version"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0
    assert MODEL_VERSION in completed.stdout


def _identity() -> dict[str, str]:
    return {
        "model_version": "0.1.0",
        "config_sha256": "a" * 64,
        "kernel_sha256": "b" * 64,
    }


def test_resume_guard_accepts_matching_identity() -> None:
    identity = _identity()
    manifest = {**identity, "schema_version": "1.0", "design_targets": {}}
    assert _resume_identity_mismatches(manifest, identity) == []


def test_resume_guard_refuses_kernel_change() -> None:
    identity = _identity()
    manifest = {**identity, "kernel_sha256": "0" * 64}
    reasons = _resume_identity_mismatches(manifest, identity)
    assert len(reasons) == 1
    assert "kernel_sha256" in reasons[0]


def test_resume_guard_refuses_model_version_change() -> None:
    identity = _identity()
    manifest = {**identity, "model_version": "9.9.9"}
    reasons = _resume_identity_mismatches(manifest, identity)
    assert len(reasons) == 1
    assert "model_version" in reasons[0]


def test_resume_guard_refuses_empty_manifest() -> None:
    identity = _identity()
    reasons = _resume_identity_mismatches({}, identity)
    assert len(reasons) == 3
