# Codex Handoff: Bound Binary Giant-Moon Simulator

## Starting point

This branch extends GitHub `main` commit
`ae34d43ea875f19d3cf6a397b09b84134e404625`. The work began as an eclipse
visualization experiment and evolved into a coupled alternate-system dynamics
model. The changes on this branch belong to one coherent simulator upgrade;
they were intentionally kept off `main` while the physical architecture was
being explored.

## Implemented architecture

The principal scenario is `config/bound_binary_giant.yaml`:

- the real Moon and giant moon orbit their mutual barycenter;
- their mutual semimajor axis is approximately 40,000 km;
- the moon-pair barycenter orbits Earth near 500,000 km;
- the giant has radius 2,514 km and mass approximately
  `2.225618e23 kg`;
- Jacobi initial conditions are converted to self-consistent Cartesian states;
- after the epoch, no moon follows a prescribed Keplerian track.

`chained_eclipse/moon_architecture.py` owns this hierarchy. Baseline propagation
uses fully coupled Sun–Earth–real-Moon–giant-Moon Newtonian integration.

The enhanced path additionally includes:

- seven active major-planet point-mass perturbers initialized from DE440s;
- Earth J2 through REBOUNDx `gravitational_harmonics`;
- full first-order post-Newtonian interactions through REBOUNDx `gr_full`;
- a vector constant-time-lag tide and coupled Earth spin through REBOUNDx
  `tides_spin`;
- differential Earth attitude relative to a real-Moon-only control;
- binary-aware trajectory caching, climate searches, and standout tracks.

The branch also contains:

- century-scale eclipse-climate support;
- Atlanta and Pacific double-eclipse time-lapse generators;
- a 2-D coupled eclipse animation;
- binary orbit portraits, shadow maps, and tide visualizations;
- tests for the hierarchy, enhanced ephemeris, planetary dynamics, and
  visualization pipelines.

## Validated results

The baseline bound binary remained stable in the existing 1,000-year experiment.
The previously executed 100-year baseline eclipse search covered 2026–2126.

The enhanced ten-year audit covered 2026-07-10 through 2036-07-10 with planets,
Earth J2, 1PN, tides, and Earth spin active:

- extra mean-solar day length: `+0.253303282 ms`;
- differential UT1: `-0.467178590 s`;
- differential longitude: `+0.001951907 deg`;
- differential pole position: `43.723706 arcsec`;
- moon–moon separation: `36,235.979–44,123.895 km`;
- moon-pair barycenter distance from Earth:
  `467,532.448–510,388.526 km`.

A one-year otherwise-identical no-J2 comparison found:

- pair-barycenter position difference: `11.286318 km`;
- orbit-normal difference: `1.185795 arcsec`;
- J2 contribution to differential day length: approximately
  `0.000005428 ms`.

The compact machine-readable audit is
`docs/audits/enhanced_physics_10y_audit.json`.

## Important interpretation

The alternate Earth attitude is the difference between:

1. the complete bound-binary enhanced system; and
2. a real-Solar-System reference with the real Moon initialized from DE440 and
   the second moon made massless.

That differential spin phase and pole rotation are composed onto Skyfield ITRS.
This is not a fitted future Earth-orientation solution.

## Highest-value missing physics

Implement these in scientifically bounded phases:

1. spin vectors and moments of inertia for both moons;
2. tides raised inside both moons by Earth and by each other;
3. Moon–Moon tidal forces and equal-and-opposite spin torques;
4. permanent lunar and giant-moon figures, including J2/C22 where justified;
5. spin-orbit synchronization and physical libration;
6. Earth J2 reaction torque and improved pole/obliquity evolution;
7. alternate-system barycentric initial-state refitting;
8. a future frequency-dependent Earth-tide interface.

Do not attempt to disguise the present constant-time-lag Earth tide as an ocean
circulation model. Continents, bathymetry, basin resonances, coastal amplification,
core–mantle coupling, atmosphere/ocean angular momentum, disruption, fragmentation,
and detailed eclipse optics remain outside the current model.

## Required validation for the next phase

- Verify pairwise internal force balance.
- Verify orbital-plus-spin angular-momentum balance in conservative limits.
- Verify dissipative tides remove mechanical energy rather than add it.
- Verify zero Love number or zero lag recovers the conservative limit.
- Test symmetry, disabled-feature backward compatibility, and invalid config.
- Include every new state and parameter in cache identity and metadata.
- Run equation-level tests before one-year and ten-year integrations.
- Compare new results numerically against the audit above.
