# Chained Eclipse Simulator Agent Guide

## Mission

Maintain a scientifically explicit simulator for eclipses in an alternate
Earth system containing the real Moon and a massive second moon. Preserve the
distinction between:

- initial-condition architecture;
- integrated force and torque models;
- eclipse geometry and Earth orientation;
- visualizations; and
- diagnostic or validation products.

Read `docs/CODEX_HANDOFF.md` before changing the enhanced binary-moon model.

## Repository rules

- Preserve unrelated and pre-existing work. Never use destructive Git commands.
- Search with `rg` and reuse existing modules before creating parallel implementations.
- Keep physical units explicit. Core dynamics use kilometres, seconds, kilograms,
  and radians.
- Every new physical parameter must be configurable, validated, included in
  trajectory-cache identity, and serialized into strict JSON metadata.
- Document governing equations, frames, signs, conservative/dissipative behavior,
  and intentional omissions.
- Internal forces must balance. Internal orbital and spin torques must balance
  when the modeled physics requires total angular-momentum conservation.
- Do not describe the Newtonian REBOUND energy value as a conserved total energy
  when J2, relativity, spin, or dissipative tides are enabled.
- Do not silently invent well-constrained material properties for the giant moon.
  Label scenario assumptions and support sensitivity testing.
- Do not commit generated media or the general `outputs/` tree unless a compact
  audit has been deliberately copied under `docs/audits/`.

## Environment

Use the checked-in virtual environment directly. `uv run` may try to rebuild
REBOUNDx and should not be used when the environment is already functional.

REBOUNDx may require its neighboring REBOUND shared library on the loader path:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR=/tmp/chained-eclipse-mpl
```

Run targeted checks while iterating:

```bash
.venv/bin/ruff check chained_eclipse tests
.venv/bin/pytest -q \
  tests/test_enhanced_dynamics.py \
  tests/test_planetary_dynamics.py \
  tests/test_enhanced_ephemeris.py \
  tests/test_moon_architecture.py \
  tests/test_coupled_eclipse.py
```

Run the full suite before handoff:

```bash
.venv/bin/pytest -q
```

## Multi-agent work

- Keep the primary agent responsible for architecture, synthesis, integration,
  and final acceptance.
- Give subagents bounded deliverables and avoid concurrent edits to the same file.
- Prefer read-only exploration and independent derivation agents.
- Require redundant checks for important torque signs, scaling laws, and
  conservation claims.
- Wait for every requested agent in a wave and reconcile disagreements from
  equations and evidence rather than majority vote.
