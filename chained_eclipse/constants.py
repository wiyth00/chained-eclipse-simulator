"""Physical constants and model conventions.

Distances are kilometres, masses are kilograms, and dynamical times are seconds
unless a symbol explicitly says otherwise.
"""

from __future__ import annotations

import math

MODEL_VERSION = "0.1.0"

# Geometric radii.  The nominal photospheric solar radius is the IAU 2015 value.
SUN_RADIUS_KM = 695_700.0
REAL_MOON_RADIUS_KM = 1_737.4
SECOND_MOON_RADIUS_KM = 838.0
SECOND_MOON_DENSITY_KG_M3 = 3_344.0

SECOND_MOON_MASS_KG = (
    4.0 / 3.0 * math.pi * (SECOND_MOON_RADIUS_KM * 1_000.0) ** 3
    * SECOND_MOON_DENSITY_KG_M3
)

EARTH_MASS_KG = 5.97217e24
REAL_MOON_MASS_KG = 7.342e22
SUN_MASS_KG = 1.98847e30

# WGS84 reference ellipsoid.
WGS84_A_KM = 6_378.137
WGS84_F = 1.0 / 298.257_223_563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# IERS 2010 / conventional dynamics values.
MU_EARTH_KM3_S2 = 398_600.435_507
MU_MOON_KM3_S2 = 4_902.800_118
MU_SUN_KM3_S2 = 132_712_440_041.939_38
EARTH_J2 = 1.082_626_68e-3
SPEED_OF_LIGHT_KM_S = 299_792.458
OBLIQUITY_J2000_DEG = 23.439_291_111

SECONDS_PER_DAY = 86_400.0
JULIAN_YEAR_DAYS = 365.25

