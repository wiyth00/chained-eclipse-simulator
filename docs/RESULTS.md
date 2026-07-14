# Scientific results

This page records the executed reference run bundled with version 0.1.0. The
counterfactual system begins at 2026-07-10 00:00 UTC and retains the optimized
second-moon state in `config/optimized_system.yaml`.

## Earliest enhanced-model chain

At 82.6859478° N, 98.5546979° W on 2026-08-12:

| Occulter | Local maximum (UTC) | Type | Totality |
|---|---:|---|---:|
| Hypothetical second moon | 15:01:19.388 | Total | 75.303 s |
| Real Moon | 20:20:35.603 | Total | 118.213 s |

The local maxima are separated by 5 h 19 min 16.2 s. The modeled central
tracks approach within 9.824 km. The event satisfies same-location limits of
12 h and 6 h, but not 3 h or 1 h. It satisfies regional-track limits of
1,000 km, 500 km, and 100 km.

## Thirty-year eclipse climate

The enhanced 2026–2056 catalog contains 695 events:

- 158 real-Moon solar eclipses
- 188 second-moon solar eclipses
- 162 real-Moon lunar eclipses
- 187 second-moon lunar eclipses

The real Moon's inclination spans 0.889°–5.370° while the second moon spans
4.259°–19.463°. The exchange is a large secular oscillation in this model, not
a demonstrated chaotic instability.

## Two-moon equilibrium tides

The 30-day visualization evaluates the exact tide-generating potential from
both moons on a rotating WGS84 Earth. At the strongest hourly sample,
2026-07-17 10:00 UTC, the degree-2 subpoint coefficients are 0.386 m for the
real Moon and 0.430 m for the second moon. The exact combined equilibrium high
is 0.827 m with a 1.235 m global peak-to-trough range.

This is not a coastal tide or flooding forecast. It omits the solar tide,
bathymetry, continents, ocean inertia, friction, resonance, loading, and
self-attraction so the two lunar contributions remain visually separable.

## Validation and numerical checks

- Four published NASA/GSFC solar eclipses validate within the project's stated
  60 s timing and 25 km central-path targets.
- A 600 s versus 300 s enhanced catalog scan preserves all 695 events and
  classifications; the largest maximum-time change is 0.019 s.
- A one-year 3,600 s versus 1,800 s trajectory-knot comparison preserves all
  36 tested events and types; maximum point displacement is 0.012 km.
- The version 0.1.0 test suite contains 68 passing tests.

## Interpretation boundary

DE440s supplies real-system initial conditions at the 2026 epoch. The enhanced
counterfactual then propagates freely with REBOUND IAS15, the other major
planets, Earth J2, first-order post-Newtonian gravity, and a constant-time-lag
Earth tidal-spin model. It is not a refitted observational ephemeris for an
Earth that actually possesses two moons.
