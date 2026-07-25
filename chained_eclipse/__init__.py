"""Chained solar-eclipse search package.

Two distinct version identifiers live here on purpose:

- ``__version__`` is the *package* version, derived from the installed
  distribution metadata (single-sourced from ``pyproject.toml``). It moves
  with releases.
- ``MODEL_VERSION`` is the *model* version, defined in ``constants``. It is
  bumped when computed outputs change for fixed inputs (see the CHANGELOG
  policy) and is therefore allowed to differ from ``__version__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .constants import MODEL_VERSION

try:
    __version__ = version("chained-eclipse")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "0+unknown"

__all__ = ["MODEL_VERSION", "__version__"]
