"""Kernel integrity checks for the auto-downloaded JPL ephemeris.

``verify_kernel_integrity`` is exercised directly on small temporary files so
no real kernel, network access, or Skyfield loader is needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chained_eclipse import ephemeris as ephemeris_module
from chained_eclipse.ephemeris import (
    EphemerisIntegrityError,
    verify_kernel_integrity,
)


def _write_kernel(tmp_path: Path, payload: bytes = b"not a real kernel") -> Path:
    kernel_path = tmp_path / "de440s.bsp"
    kernel_path.write_bytes(payload)
    return kernel_path


def _sidecar(kernel_path: Path) -> Path:
    return kernel_path.with_name(kernel_path.name + ".sha256")


def test_first_load_records_sidecar_but_is_not_verified(tmp_path: Path) -> None:
    kernel_path = _write_kernel(tmp_path)
    digest, verified = verify_kernel_integrity(kernel_path)
    assert digest == hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    assert verified is False, "a fresh recording must not count as verification"
    assert _sidecar(kernel_path).read_text(encoding="utf-8").split()[0] == digest


def test_second_load_verifies_against_recorded_sidecar(tmp_path: Path) -> None:
    kernel_path = _write_kernel(tmp_path)
    verify_kernel_integrity(kernel_path)
    digest, verified = verify_kernel_integrity(kernel_path)
    assert verified is True
    assert digest == hashlib.sha256(kernel_path.read_bytes()).hexdigest()


def test_changed_kernel_fails_sidecar_check(tmp_path: Path) -> None:
    kernel_path = _write_kernel(tmp_path)
    verify_kernel_integrity(kernel_path)
    kernel_path.write_bytes(b"tampered or truncated kernel")
    with pytest.raises(EphemerisIntegrityError, match="changed on disk"):
        verify_kernel_integrity(kernel_path)


def test_allow_unverified_bypasses_sidecar_mismatch(tmp_path: Path) -> None:
    kernel_path = _write_kernel(tmp_path)
    verify_kernel_integrity(kernel_path)
    kernel_path.write_bytes(b"tampered or truncated kernel")
    digest, verified = verify_kernel_integrity(kernel_path, allow_unverified=True)
    assert verified is False
    assert digest == hashlib.sha256(kernel_path.read_bytes()).hexdigest()


def test_pinned_digest_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_path = _write_kernel(tmp_path)
    good = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        ephemeris_module.KERNEL_SHA256_PINS, "de440s.bsp", good.upper()
    )
    digest, verified = verify_kernel_integrity(kernel_path)
    assert digest == good
    assert verified is True


def test_pinned_mismatch_raises_even_with_matching_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_path = _write_kernel(tmp_path)
    verify_kernel_integrity(kernel_path)  # records a matching sidecar
    monkeypatch.setitem(
        ephemeris_module.KERNEL_SHA256_PINS, "de440s.bsp", "0" * 64
    )
    with pytest.raises(EphemerisIntegrityError, match="pinned"):
        verify_kernel_integrity(kernel_path)


def test_pin_beats_sidecar_for_verified_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_path = _write_kernel(tmp_path)
    good = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    monkeypatch.setitem(ephemeris_module.KERNEL_SHA256_PINS, "de440s.bsp", good)
    digest, verified = verify_kernel_integrity(kernel_path)
    assert (digest, verified) == (good, True)
    # Sidecar was still recorded for future unpinned loads.
    assert _sidecar(kernel_path).exists()
