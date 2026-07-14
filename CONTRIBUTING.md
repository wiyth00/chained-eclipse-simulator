# Contributing

Contributions that improve numerical validation, physical modeling,
performance, documentation, or reproducibility are welcome.

## Development setup

```bash
git clone https://github.com/wiyth00/chained-eclipse-simulator.git
cd chained-eclipse-simulator
uv sync --all-extras --no-editable
.venv/bin/pytest
.venv/bin/ruff check .
```

The first ephemeris-backed test or run downloads JPL DE440s into
`data/ephemeris/`. Downloaded kernels, trajectory caches, and generated output
are intentionally ignored by Git.

## Pull requests

- Keep physical assumptions and units explicit.
- Add or update tests for numerical or classification changes.
- Distinguish observed ephemerides, restricted counterfactual dynamics, and
  freely propagated alternate-system results.
- Report convergence or sensitivity evidence for precision claims.
- Do not commit generated kernels, trajectory caches, or large media files.

For substantial model changes, open an issue first so the intended scientific
boundary can be agreed before implementation.
