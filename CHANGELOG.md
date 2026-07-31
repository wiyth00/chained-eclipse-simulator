# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a scientific model, entries note whether a change alters computed
results. `MODEL_VERSION` in `chained_eclipse/constants.py` is bumped whenever
outputs change for a fixed input configuration.

## [Unreleased]

### Added

- Explicit inertial spin vectors and polar moments of inertia for Earth, the
  real Moon, and the giant moon, evolved by one coupled REBOUNDx `tides_spin`
  system.
- Configurable low, nominal, and high lunar/giant-moon Love-number, time-lag,
  inertia, and initial-spin scenarios with provenance and deliberate
  uncertainty labels.
- Orbital-plus-spin angular-momentum and complete REBOUNDx mechanical-energy
  diagnostics, three-body spin histories, synchronization ratios, and
  barycentric frame/recentering metadata.
- A permanent-figure configuration interface that records J2/C22 assumptions
  but fails fast if enabled before a reaction-aware attitude backend exists.
- Dependabot configuration for GitHub Actions and uv dependencies.
- Issue and pull request templates.
- This changelog.
- A three-times-scale, constant-density distant giant-moon scenario and a
  command-line optical/dynamical scaling diagnostic.
- A hierarchical binary-moon architecture, analytic Hill/hierarchy screens,
  a 1,000-year bound giant-moon configuration, and binary-aware stability
  diagnostics.

### Changed

- The enhanced bound-binary configuration now includes tides raised inside
  both moons by every active massive body, including the other moon, with
  equal-and-opposite orbital and spin reactions supplied by one `tides_spin`
  force instance.
- Enhanced trajectory cache identity now includes all rotational parameters
  and initial states, REBOUND/REBOUNDx versions, and the DE440s kernel SHA-256;
  cache writers use process-unique temporary files and persist the integrator's
  velocity samples for trustworthy osculating-element audits.
- The model version is now `0.2.0` because the enhanced bound-binary numerical
  trajectory changes when its new rotational-tide scenario is active.
- CI now only runs on pushes to `main` (plus all pull requests) and cancels
  superseded in-progress runs, instead of running twice per branch push.
- CI reports test coverage via `pytest-cov`, which was already declared as a
  dev dependency but not invoked.
- Configured second-moon radius and mass now flow through restricted searches,
  coupled solar and lunar eclipses, tides, maps, and animations. Baseline
  results are unchanged.
- Alternate scenario design runs no longer overwrite the checked-in baseline
  optimized orbit.
- Coupled eclipse, orbit-portrait, and stability commands now accept synthetic
  Jacobi outer/mutual moon states while preserving the original DE440s
  initialization by default.

## [0.1.0] - 2026-07-13

### Added

- Initial public release: ephemeris-driven two-moon eclipse search, coupled
  N-body dynamics via REBOUND/REBOUNDx, tidal and spin evolution, eclipse
  geometry and ground-track mapping, climate response modelling, and the
  associated reporting and animation entry points.

[Unreleased]: https://github.com/wiyth00/chained-eclipse-simulator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wiyth00/chained-eclipse-simulator/releases/tag/v0.1.0
