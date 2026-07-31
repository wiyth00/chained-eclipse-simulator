"""Conservation diagnostics for coupled orbital and spin dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RotationalDiagnosticSnapshot:
    """One instantaneous orbital-plus-spin conservation snapshot."""

    time_s: float
    orbital_angular_momentum_kg_km2_s: tuple[float, float, float]
    spin_angular_momentum_kg_km2_s: tuple[float, float, float]
    total_angular_momentum_kg_km2_s: tuple[float, float, float]
    mechanical_energy_kg_km2_s2: float
    energy_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rotational_diagnostic_snapshot(
    simulation: Any,
) -> RotationalDiagnosticSnapshot:
    """Return reaction-aware spin/orbit diagnostics in the simulation frame.

    The energy expression follows REBOUNDx's documented bookkeeping:

    * ``gr_full_hamiltonian`` replaces ``simulation.energy()`` when GR is on;
    * gravitational-harmonics potential is added when that effect is on; and
    * ``tides_spin_energy`` adds spin kinetic plus equilibrium quadrupole
      potentials.

    Dissipative positive-lag tides make this mechanical energy decrease.  With
    Earth J2 active, total angular momentum is diagnostic only because
    REBOUNDx gravitational harmonics omits the source-spin reaction torque.
    """

    orbital_raw = simulation.angular_momentum()
    orbital = np.asarray(
        (orbital_raw.x, orbital_raw.y, orbital_raw.z),
        dtype=float,
    )
    extras = getattr(simulation, "_extras_ref", None)
    if extras is None:
        spin = np.zeros(3, dtype=float)
    else:
        spin_raw = extras.spin_angular_momentum()
        spin = np.asarray((spin_raw.x, spin_raw.y, spin_raw.z), dtype=float)
    total = orbital + spin

    gr_force = getattr(simulation, "_gr_full_force_ref", None)
    terms: list[str]
    if gr_force is None:
        energy = float(simulation.energy())
        terms = ["newtonian_orbital_energy"]
    else:
        energy = float(extras.gr_full_hamiltonian(gr_force))
        terms = ["gr_full_hamiltonian"]
    if getattr(simulation, "_gravitational_harmonics_force_ref", None) is not None:
        energy += float(extras.gravitational_harmonics_potential())
        terms.append("gravitational_harmonics_potential")
    if getattr(simulation, "_tides_spin_force_ref", None) is not None:
        energy += float(extras.tides_spin_energy())
        terms.append("tides_spin_energy")

    values = np.concatenate((orbital, spin, total, np.asarray((energy,))))
    if not np.all(np.isfinite(values)) or not math.isfinite(float(simulation.t)):
        raise ValueError("rotational diagnostic state must be finite")
    return RotationalDiagnosticSnapshot(
        time_s=float(simulation.t),
        orbital_angular_momentum_kg_km2_s=tuple(float(value) for value in orbital),
        spin_angular_momentum_kg_km2_s=tuple(float(value) for value in spin),
        total_angular_momentum_kg_km2_s=tuple(float(value) for value in total),
        mechanical_energy_kg_km2_s2=energy,
        energy_terms=tuple(terms),
    )


def relative_vector_change(
    initial: tuple[float, float, float] | list[float],
    final: tuple[float, float, float] | list[float],
) -> float:
    """Return ``|final-initial| / |initial|`` for a nonzero vector."""

    first = np.asarray(initial, dtype=float)
    second = np.asarray(final, dtype=float)
    scale = float(np.linalg.norm(first))
    if first.shape != (3,) or second.shape != (3,) or scale == 0.0:
        raise ValueError("relative vector change requires finite nonzero vectors")
    return float(np.linalg.norm(second - first) / scale)


__all__ = [
    "RotationalDiagnosticSnapshot",
    "relative_vector_change",
    "rotational_diagnostic_snapshot",
]
