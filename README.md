# Chained Eclipse Simulator

[![CI](https://github.com/wiyth00/chained-eclipse-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/wiyth00/chained-eclipse-simulator/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2EA44F.svg)](LICENSE)

![Two-moon equilibrium tide visualization](docs/assets/two_moon_equilibrium_tides.png)

> A reproducible counterfactual-astronomy laboratory: real JPL ephemerides,
> numerical eclipse geometry, coupled N-body dynamics, and a hypothetical
> second moon. This is a scientific simulation—not an astronomical forecast.

This project searches for rapid pairs of solar eclipses involving the real Moon
and a hypothetical second moon.  It uses JPL DE440s ephemerides for the real
Sun, Earth, and Moon, three-dimensional shadow-cone geometry on a rotating
WGS84 Earth, numerical propagation for the second moon, and REBOUND for the
long-term coupled stability experiment.

The central scientific distinctions are deliberate:

- The **eclipse search** is an ephemeris-forced restricted model.  DE440s
  prescribes the real bodies and the added moon does not perturb them.
- The **baseline coupled model** is a self-consistent counterfactual four-body
  REBOUND integration initialized from DE440s. It includes the added moon's
  calculated mass, so the real Moon and Earth cannot remain on their DE440s
  trajectories.
- The **enhanced coupled model** adds all seven other major planets, Earth J2,
  REBOUNDx first-post-Newtonian gravity and a coupled tidal-spin model. It also
  integrates a matched massless-second-moon control to isolate the alternate
  Earth's attitude from the real-Earth orientation already represented by
  Skyfield.

These models answer different questions; none is silently treated as an exact
continuation of the real Solar System after the July 10, 2026 epoch.

## Headline result

Under the optimized enhanced configuration, an observer at 82.6859° N,
98.5547° W receives two separate totalities on 2026-08-12. The hypothetical
moon reaches maximum at 15:01:19 UTC; the real Moon follows at 20:20:35 UTC,
5 h 19 min 16 s later. Their central tracks pass within 9.82 km.

The 30-year enhanced run catalogs 695 solar and lunar eclipses while exposing
a large secular inclination exchange between the moons. The companion tide
visualization finds a strongest sampled two-moon equilibrium high of 0.827 m;
that is a tide-potential proxy, not a coastal water-level prediction.

See [Scientific results](docs/RESULTS.md) for the executed-run summary,
validation status, limitations, and release artifacts.

## Quick start

```bash
uv sync --all-extras --no-editable
uv run chained-eclipse --mode full
```

The first run downloads `de440s.bsp` into `data/ephemeris/`.  Results are
written under `outputs/`, including CSV/JSON data, an optimized configuration,
maps, timelines, diagnostic plots, a stability plot, and an HTML technical
report.

Large generated artifacts are intentionally excluded from Git history. The
reference movies and executed-run bundle are attached to the
[latest GitHub release](https://github.com/wiyth00/chained-eclipse-simulator/releases/latest).

## Distant giant-moon scenario

`config/distant_giant.yaml` replaces the 838 km moon at 180,000 km with a
Mercury-sized body whose radius and orbit are both scaled by three:

| Quantity | Baseline second moon | Distant giant |
|---|---:|---:|
| Radius | 838 km | 2,514 km |
| Semimajor axis | 180,000 km | 540,000 km |
| Perigee | 172,800 km | 518,400 km |
| Mass at 3,344 kg/m³ | 8.243 × 10²¹ kg | 2.226 × 10²³ kg |
| Mass in real Moons | 0.112 | 3.031 |
| Geocentric angular diameter at perigee | 0.5557° | 0.5557° |
| Restricted-model period | 8.80 days | 45.71 days |
| Massive two-body period | 8.79 days | 44.88 days |

The optical scaling is exact from Earth's center because angular radius depends
on `radius / distance`. A surface observer sees a small parallax difference
because Earth's own radius is not scaled. At constant density, mass grows as
the cube of the scale factor. The leading tide term `mass / distance³`
therefore stays almost unchanged even though the new moon is 27 times as
massive.

The dynamics do not share that symmetry. The giant moves the Earth–giant
barycenter about 19,401 km from Earth's center, well outside Earth, and its
initial separation from the real Moon is only about 1.32 mutual Hill radii.
That is a strong instability warning; the restricted eclipse design can still
construct the geometry, but the coupled REBOUND run decides whether the
counterfactual survives.

The saved solved orientation in `config/distant_giant_optimized.yaml` produces
two totalities at 65.2235° N, 25.2419° W on 2026-08-12. The real Moon reaches
maximum at 17:45:56.542 UTC and the giant moon at 17:57:58.743 UTC, a
12 min 2.2 s gap. Giant-moon totality lasts 393.9 s (6 min 33.9 s) at that
site. The totality intervals are distinct, although the long partial phases
overlap.

That 12-minute chain is an optics-first restricted-model design. It does not
survive when the giant's full mass is coupled from the July epoch. In the
four-body run, the giant produces a 830.3 s total eclipse near 33.4391° N,
78.5684° W on August 9 at 17:35:46 UTC. The perturbed real Moon produces its
own 338.9 s total eclipse near 49.3235° N, 41.4470° E on August 10 at
10:47:55 UTC. The gap is 17 h 12 min 9 s, so the coupled system no longer
qualifies as a chained eclipse.

The 10-year Newtonian four-body stability attempt does not make it through
year one. The moons' osculating radial ranges cross; they pass within about
16,192 km after 0.1002 years, and the real Moon crosses the modeled
Earth-system ejection boundary after 0.6977 years. This is a clean numerical
failure, not integration noise: the maximum relative energy error is
`7.9e-16`. The result is specific to the project's baseline coupled force model,
which omits major planets, Earth J2, tides, relativity, and lunar figure terms,
but those refinements are not plausibly large enough to rescue such a tightly
packed massive pair.

Inspect either constant-density physics or a fixed-mass optical control:

```bash
uv run chained-eclipse-moon-scaling --scale-factor 3
uv run chained-eclipse-moon-scaling --scale-factor 3 --mass-mode fixed-mass
```

Design the giant-moon eclipse into a separate output directory, without
replacing the checked-in baseline orbit:

```bash
uv run chained-eclipse --mode design \
  --config config/distant_giant.yaml \
  --output outputs/distant_giant

uv run chained-eclipse --mode stability \
  --config config/distant_giant.yaml \
  --output outputs/distant_giant \
  --stability-years 10
```

## Bound binary-moon solution

`config/bound_binary_giant.yaml` keeps the Mercury-sized moon and its
approximately solar-sized appearance, but changes the topology. The moons no
longer follow neighboring independent Earth orbits. Instead they form a tight
binary whose barycenter orbits Earth:

For circular coplanar independent orbits, the `2√3` mutual-Hill screen would
force this giant moon either inside about 149,071 km or outside about
991,228 km while the real Moon remains near 384,400 km. The outer solution is
beyond the roughly 744,465 km prograde solar-stability limit; the inner
solution makes the giant enormous in the sky. The binary hierarchy avoids
both conflicts.

| Quantity | Bound value | Why it matters |
|---|---:|---|
| Moon-pair barycenter semimajor axis | 500,000 km | 0.329 of the Earth-system Hill radius |
| Outer eccentricity and period | 0.01; 39.751 days | Keeps the pair well inside the prograde solar stability zone |
| Mutual semimajor axis | 40,000 km | Small compared with the binary's Hill region |
| Mutual eccentricity and period | 0.01; 4.139 days | Avoids both collision and weak binding |
| Binary Hill radius at outer periapsis | 126,069 km | Mutual apoapsis is only 0.320 of this radius |
| Hierarchy ratio | 12.375 | Exceeds the 9.552 Mardling–Aarseth screen |
| Initial minimum surface gap | 35,349 km | Safely outside physical contact |
| Giant apparent diameter range | 0.559–0.594° | Still comparable to the Sun |
| Real-Moon apparent diameter range | 0.372–0.429° | Usually too small for totality |

In the executed 1,000-year Newtonian four-body integration, both nested orbits
remain bound. The moon–moon separation never falls below 36,215 km, the mutual
eccentricity remains below 0.120, and the outer eccentricity remains below
0.065. Maximum relative energy error is `1.2e-15`. Individual Earth-centred
moon elements look violently corrugated because each moon carries the
4.14-day binary motion; stability is therefore judged from the Jacobi mutual
orbit and the moon-pair barycenter orbit.

For visual spectacle the saved case places both nested orbits in the ecliptic.
That coplanarity is not what supplies the binding, but it makes eclipses
frequent. From 2026-07-10 through 2027-07-10 the coupled model finds eight
annular real-Moon eclipses and eight total giant-moon eclipses, including six
pairs whose global maxima are within 12 hours. On 2027-06-22 their central
tracks pass within about 1.6 km, with the global maxima 2 h 48 min apart.

This is an alternate initial Solar System, not a continuation of the observed
Moon. The real Moon is deliberately moved from its DE440s state into the
synthetic binary at the epoch. The 1,000-year screen omits tides, Earth J2,
major planets, relativity, and lunar figure terms, so it establishes
short-to-intermediate-term gravitational boundedness rather than formation
plausibility or billion-year tidal survival.

```bash
uv run chained-eclipse-moon-architecture \
  --config config/bound_binary_giant.yaml

uv run chained-eclipse --mode stability \
  --config config/bound_binary_giant.yaml \
  --output outputs/bound_binary_giant \
  --stability-years 1000

uv run chained-eclipse-coupled \
  --config config/bound_binary_giant.yaml \
  --start 2026-07-10T00:00:00Z \
  --end 2027-07-10T00:00:00Z \
  --output-dir outputs/bound_binary_giant/coupled

uv run chained-eclipse-orbits \
  --config config/bound_binary_giant.yaml \
  --output outputs/bound_binary_giant/orbits/binary_moon_orbits.png \
  --days 50
```

## Reproducible modes

```bash
# Validate the real-Moon model against NASA/GSFC reference circumstances.
.venv/bin/chained-eclipse --mode validate

# Design the earliest deliberately aligned configuration.
.venv/bin/chained-eclipse --mode design

# Propagate the saved configuration without redesigning it per eclipse.
.venv/bin/chained-eclipse --mode fixed

# Run the 1,000-year coupled stability experiment.
.venv/bin/chained-eclipse --mode stability

# Render the solved 2026 event as a two-scale 3-D H.264 movie.
.venv/bin/python -m chained_eclipse.animation \
  --output outputs/animations/chained_eclipse_20260812_3d.mp4 \
  --frames 721 --fps 24 --lead-minutes 12 --trail-minutes 12 --dpi 140

# Render the detailed equirectangular shadow-footprint movie.
.venv/bin/python -m chained_eclipse.animation_2d \
  --output outputs/animations/chained_eclipse_20260812_2d_map.mp4

# Render the flat world-map preview for the bound binary-moon scenario.
.venv/bin/python -m chained_eclipse.coupled_animation_2d \
  --results outputs/bound_binary_giant/coupled/coupled_eclipses.json \
  --config config/bound_binary_giant.yaml \
  --pair-index 5 \
  --output outputs/bound_binary_giant/animations/20270622_world_map.mp4

# Follow both apparent lunar disks through Atlanta's twelve-hour partial phase.
.venv/bin/python -m chained_eclipse.atlanta_timelapse \
  --output outputs/bound_binary_giant/animations/20270622_atlanta_geometry.mp4

# Watch the August 2026 Pacific total-at-sunrise / annular-at-noon double eclipse.
.venv/bin/python -m chained_eclipse.pacific_double_timelapse \
  --output outputs/bound_binary_giant/animations/20260821_pacific_double.mp4
```

## 3-D animation

The movie combines a close view of the rotating WGS84 Earth and physical core
shadow cones with a true-centre orbital view.  Sun, Earth, and real-Moon states
come from DE440s; the second moon is re-integrated with the same DOP853,
Earth-J2, prescribed-Sun/Moon model used by the eclipse search.  The moon
markers in the wide view are enlarged so they remain visible, but their centre
positions, the Earth, shadow axes, and cone opening angles are physically
scaled.  The close camera follows the Greenland/Iceland corridor containing
the best common observing site.

## 2-D shadow map animation

The equirectangular North Atlantic map recomputes instantaneous topocentric
disk overlap on a 0.25-degree WGS84 grid for every frame.  It shows the true
partial-eclipse footprints, total/central cells, moving center points, recent
centerline trails, Earth day/night shading, and the common observing site over
detailed Natural Earth 50 m coastlines and borders.  The blue and orange
footprints can overlap because the two partial phases genuinely overlap even
though the two periods of totality at the common site are separate.

The coupled world-map command applies the same topocentric calculation to the
stable binary-moon architecture.  Its default quick-look render spans both
global events on June 22–23, 2027: blue is the real Moon, orange is the giant
moon, faint color shows each eclipse's full reach, and saturated color is the
instantaneous moving footprint.

The Atlanta timelapse holds the apparent Sun fixed while both lunar disks move
at their exact topocentric angular sizes.  It includes the full and close solar
fields, accumulated center trails, separate and combined obscuration, and a
timeline for the nested real- and giant-moon partial phases.

The Pacific double-eclipse timelapse adds a horizon-aware solar path.  It begins
with the giant moon's eclipse below the horizon, slows through ten minutes of
sunrise totality, crosses the clear-morning interval, then slows again through
the real Moon's nearly 24-minute annular phase near noon.  The clock uses
UTC+12 for the open-ocean common site.

## Later standalone second-moon eclipses

Continue the exact saved fixed system without redesigning the orbit:

```bash
.venv/bin/python -m chained_eclipse.standalone \
  --start 2026-08-13T00:00:00Z --end 2027-08-13T00:00:00Z

.venv/bin/python -m chained_eclipse.standalone_map \
  --grid-step-deg 0.1 --time-step-seconds 30
```

The first command enumerates standalone second-moon eclipses into JSON and
CSV.  The second produces the detailed ground track for the first later
central eclipse, including maximum-obscuration shading, the annularity band,
centerline, and greatest-eclipse point.

Plot all five total-eclipse centerlines from the May–July 2027 eclipse season:

```bash
.venv/bin/python -m chained_eclipse.total_tracks_2027
```

Zoom eclipse number four onto the United States and calculate Atlanta's exact
topocentric circumstances:

```bash
.venv/bin/python -m chained_eclipse.eclipse4_atlanta
```

These two one-off analyses are intentionally not installed as console
scripts; run them with `python -m` as shown.

Run the unit and reference tests with:

```bash
.venv/bin/pytest
```

## Precision statement

Real-eclipse timing and centerline coordinates are checked against detailed
NASA/GSFC Besselian-element pages.  Results for the hypothetical moon inherit
the verified geometric solver but not the observational precision of a real
ephemeris.  Long-horizon fixed-system event times are model predictions and
are sensitivity-tested; they are not astronomical forecasts.

## Fully coupled eclipse mode

The saved orbit can also be propagated in the self-consistent REBOUND
Sun–Earth–real-Moon–second-moon model, then searched directly for eclipses:

```bash
.venv/bin/python -m chained_eclipse.coupled_eclipse
.venv/bin/python -m chained_eclipse.coupled_figures
```

This baseline mode allows the second moon's calculated mass to perturb Earth
and the real Moon. It retains the exact saved initial state but does not retain
DE440s after the epoch. Results are written under `outputs/coupled/`. The
four-body control deliberately omits major planets, Earth J2, tides,
relativity, and lunar figure terms. In this mode the August 12 chain survives,
but moves to the Canadian high Arctic and widens to 5 hours 18 minutes 44
seconds; the earlier restricted-model Atlanta prediction does not survive.

## Lunar eclipses in the two-moon sky

Search for either moon passing through Earth's penumbra and umbra inside the
same coupled trajectory:

```bash
.venv/bin/python -m chained_eclipse.lunar_eclipse
```

The catalog is written under `outputs/coupled/lunar_eclipses/`. The visually
closest early pairing occurs on 2026-07-30: the second moon is totally eclipsed
for about 95 minutes while the nearly full real Moon sits only 7.35 degrees
away in the sky.

## Thirty-year eclipse climate

Propagate the fully coupled system through 2056, catalog every solar and lunar
eclipse, and render the inclination exchange and annual event rates:

```bash
.venv/bin/python -m chained_eclipse.eclipse_climate
.venv/bin/python -m chained_eclipse.climate_tracks
```

The first command writes the complete 2026-2056 catalog, notable-event table,
inclination samples, annual counts, and eclipse-climate chart under
`outputs/coupled/eclipse_climate_30y/`. The second plots the four longest
sampled total-solar-eclipse tracks from each moon. The default trajectory uses
one-hour interpolation knots and a ten-minute event scan; a five-minute scan
recovers the same 696-event catalog with no classification differences.

This counterfactual run exhibits a regular secular inclination exchange, not a
demonstrated chaotic instability. Near 2040 the real Moon's inclination falls
below one degree while the second moon approaches 19.5 degrees; their eclipse
rates respond in opposite directions. Long-range ground coordinates remain
conditional because this baseline catalog omits tides, major planets, Earth
J2, relativity, and a self-consistent alternate-Earth rotation history.

The bound binary-giant architecture can be extended to a century with the same
one-hour trajectory and ten-minute detector cadences:

```bash
.venv/bin/python -m chained_eclipse.eclipse_climate \
  --start 2026-07-10T00:00:00Z \
  --end 2126-07-10T00:00:00Z \
  --config config/bound_binary_giant.yaml \
  --output-dir outputs/bound_binary_giant/eclipse_climate_100y
```

This mode preserves the hierarchical binary initial conditions instead of
silently converting the two moons back to independent Earth-centered orbits.
Its century catalog and annual tables are written beside the plot.

## Enhanced dynamics mode

The enhanced climate run keeps the same optimized epoch state but replaces the
four-body control with a higher-fidelity counterfactual integration:

- Mercury and Venus are active planet-centre point masses. Mars through
  Neptune are active planetary-system barycentres with matching DE440 system
  gravitational parameters, so their unmodeled satellites are included in the
  system mass rather than silently discarded. Together with Earth, all eight
  major planets are active. Pluto is optional in the Python API and is off by
  default.
- DE440s supplies every real body's BCRS/ICRF state at the epoch only. After
  that instant, the Sun, planets, Earth and both moons evolve freely and
  self-consistently under REBOUND IAS15.
- REBOUNDx `gravitational_harmonics` applies Earth's J2 quadrupole using the
  configured equatorial radius and spin direction.
- REBOUNDx `gr_full` applies full first-order post-Newtonian interactions among
  all active bodies. This is materially stronger than a Sun-only Schwarzschild
  correction, but it still omits higher PN orders, frame dragging and the solar
  quadrupole.
- REBOUNDx `tides_spin` treats Earth as the deformable, spinning body and the
  other active bodies as point-mass tide raisers. Its constant time lag is
  normalized to reproduce 38.2 mm/year of circular real-Moon recession at
  384,400 km, and Earth's three-component spin vector is evolved inside the
  same N-body integration.

REBOUND and REBOUNDx are constrained to the compatible `>=4.6,<5.0` API range
in `pyproject.toml`. The quick-start install includes both packages and the
project's command-line entry points.

## Thirty-day two-moon tide animation

Render the direct interference of the real and hypothetical lunar tidal bulges:

```bash
.venv/bin/python -m chained_eclipse.tide_visualization
```

The equirectangular movie uses the enhanced N-body trajectory and evaluates the
exact tide-generating potential on a global one-degree grid every hour. The
usual degree-2 amplitudes are retained as per-frame diagnostics, while the map
also captures the small near-side/far-side asymmetry of the close second moon.
The plotted height is an instantaneous equilibrium open-ocean proxy. It is not a
coastal water-level forecast: the deliberately transparent model omits the solar
tide, continents, bathymetry, ocean inertia, resonance, friction, loading, and
self-attraction. A companion CSV and JSON manifest retain every frame's moon
subpoints, distances, individual amplitudes, bulge alignment, and global extrema.

### Differential Earth attitude

Ground longitude cannot remain tied to the real Earth's rotation after adding
a massive second moon. The enhanced trajectory therefore performs two
integrations:

1. the full alternate system, including a configured bound binary-moon
   architecture when present; and
2. a real-Solar-System control with the same epoch, planets, J2, `gr_full`, and
   `tides_spin`, with the second moon massless and the real Moon initialized
   from DE440.

The difference in their integrated Earth spin phase and spin-pole direction is
composed onto Skyfield's standard ITRS orientation. This retains the real-Earth
UT1/precession/nutation model as the zero-order reference while adding only the
counterfactual perturbation attributed to the second moon. It is a differential
attitude model, not a newly fitted future Earth-orientation series.

The Earth tide is the REBOUNDx vector constant-time-lag model, calibrated to
the present real-Moon recession rate. Both moons and the Sun raise tides on
Earth and exchange angular momentum with its evolving spin vector. Earth J2
acts on every active body, but its axis is held fixed: J2 changes the orbits
and eclipse geometry but this model does not yet apply the equal-and-opposite
J2 figure torque to Earth's spin.

### Run, map, and compare

Generate the enhanced 2026–2056 catalog and climate plot:

```bash
.venv/bin/python -m chained_eclipse.eclipse_climate \
  --dynamics-model enhanced \
  --output-dir outputs/coupled/eclipse_climate_30y_enhanced
```

Run the same enhanced stack for the stable giant/real-Moon binary:

```bash
.venv/bin/python -m chained_eclipse.eclipse_climate \
  --config config/bound_binary_giant.yaml \
  --dynamics-model enhanced \
  --output-dir outputs/bound_binary_giant/eclipse_climate_30y_enhanced
```

Recompute the standout total-eclipse tracks with that same enhanced trajectory:

```bash
.venv/bin/python -m chained_eclipse.climate_tracks \
  --climate outputs/coupled/eclipse_climate_30y_enhanced/climate.json
```

Compare the baseline and enhanced catalogs event by event:

```bash
.venv/bin/python -m chained_eclipse.enhanced_comparison \
  outputs/coupled/eclipse_climate_30y/climate.json \
  outputs/coupled/eclipse_climate_30y_enhanced/climate.json \
  --output-dir outputs/coupled/eclipse_climate_30y_enhanced/comparison \
  --max-match-days 7
```

The comparator performs one-to-one nearest-time matching separately for solar
and lunar eclipses from each moon. It writes `comparison.json`,
`matched_events.csv`, and a static plot of timing and global-maximum-point
displacements. It also reports added/removed events, type changes, count deltas,
and changes to rapid-pair and chained-eclipse classifications.
The seven-day assignment window is short enough to avoid cross-pairing adjacent
second-moon eclipse cycles late in the run. `added` and `removed` mean unmatched
inside that window; they are not automatically literal births or disappearances
of physical eclipses, especially for grazing partial and penumbral events.

Build the result-first enhanced Markdown and HTML report, including both saved
convergence comparisons and the original published-eclipse validation:

```bash
.venv/bin/python -m chained_eclipse.enhanced_report \
  --validation outputs/validation_report.json \
  --convergence outputs/coupled/eclipse_climate_30y_enhanced/convergence_detector_300s/comparison.json \
  --convergence outputs/coupled/eclipse_climate_30y_enhanced/convergence_trajectory_1y_1800s/comparison.json
```

The standalone secular tidal-magnitude audit is reproducible with:

```bash
.venv/bin/python -m chained_eclipse.tides_spin \
  --output-dir outputs/coupled/tidal_spin_audit
```

### Trajectory cache

Enhanced propagation is the expensive step because the full system and its
massless-second-moon attitude control must both be integrated. By default,
`EnhancedEphemeris` stores a compressed, content-addressed trajectory at:

```text
data/trajectories/enhanced_<20-character-sha256-prefix>.npz
```

The cache contains the sampled Sun/Earth/two-moon positions, full and control
Earth-spin vectors, and the Newtonian energy diagnostic. Its key includes the
orbital elements, end time, trajectory cadence, IAS15 tolerance, J2/relativity
and tide settings, Pluto toggle, force-model schema, and DE440s kernel name,
size, and modification time. Detector cadence is intentionally excluded
because it consumes but does not alter the trajectory. Repeating a climate or
track run with the same trajectory settings loads the cached arrays; changing
a keyed physical setting creates a different file instead of overwriting the
old integration.

Programmatic callers can set `cache_trajectory=False` or provide
`trajectory_cache_dir` to `EnhancedEphemeris`. If the force implementation or
REBOUND/REBOUNDx build changes without a cache-schema change, remove the
corresponding cached file before treating the rerun as independent; source and
library binaries themselves are not hashed into the key.

### Scientific limits of the enhanced run

The enhanced mode narrows the largest omissions, but it does not turn the
hypothetical system into a high-precision astronomical forecast:

- DE440s is an epoch initializer, not a continuing constraint or a refitted
  ephemeris for a Solar System containing the second moon.
- The planets are point masses; Mars through Neptune are system barycentres.
  Asteroids, trans-Neptunian objects, Pluto by default, and planetary figure
  terms other than Earth J2 are omitted.
- Earth J2 has a fixed coefficient and radius. Lunar and second-moon permanent
  figures, libration, deformation, and tides raised inside either moon are not
  modeled.
- `tides_spin` is a constant-time-lag equilibrium-tide approximation calibrated
  at the present lunar orbit. It is not a frequency-dependent ocean and
  solid-Earth tide model, and it omits changing Earth inertia and
  atmosphere/ocean angular momentum exchange.
- The differential attitude overlay is physically motivated but is not an
  observational UT1 or polar-motion prediction. J2 reaction torque, free
  nutation, and a fully refitted alternate-Earth orientation solution remain
  outside the model.
- Relativity stops at first post-Newtonian order. Solar frame dragging, the
  solar quadrupole, and higher-order terms are omitted.
- REBOUND's ordinary `Simulation.energy()` is only a Newtonian diagnostic here:
  it excludes the J2/1PN Hamiltonian contributions, while tides are genuinely
  dissipative. Its drift must not be labeled total energy error.
- Eclipse geometry still omits lunar limb topography and atmospheric
  enlargement of Earth's shadow during lunar eclipses. Grazing classifications
  remain more cadence-sensitive than central events.

Accordingly, late ground tracks and contact times are predictions of this
stated counterfactual model. The baseline-versus-enhanced comparison is the
right way to measure their model sensitivity; neither catalog is an
observational forecast for the real Earth.
