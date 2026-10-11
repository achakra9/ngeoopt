"""Containers passed back from energy backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StateInfo:
    """One electronic state at one geometry.

    ``signature`` is an optional AO-basis quantity (e.g. a transition density
    matrix) used for overlap-based root following. ``ao_overlap`` is the AO
    overlap matrix needed to compare signatures; backends only need to
    attach it to the first state of a result.
    """

    energy: float                      # total energy, Hartree
    index: int = 0                     # 0 = ground state, n = n-th root
    multiplicity: Optional[int] = None
    oscillator_strength: Optional[float] = None
    label: str = ""
    signature: Optional[np.ndarray] = None
    ao_overlap: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyResult:
    """Everything a backend returns for one geometry."""

    states: List[StateInfo]
    converged: bool = True
    wall_time: float = 0.0
    output_file: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def energy(self) -> float:
        """Ground-state (first) energy."""
        return self.states[0].energy

    @property
    def energies(self) -> np.ndarray:
        return np.array([s.energy for s in self.states])

    @property
    def ao_overlap(self) -> Optional[np.ndarray]:
        for s in self.states:
            if s.ao_overlap is not None:
                return s.ao_overlap
        return None

    @classmethod
    def from_energies(cls, energies, multiplicities=None, **kw) -> "EnergyResult":
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        mults = multiplicities if multiplicities is not None else [None] * len(energies)
        states = [StateInfo(float(e), i, m) for i, (e, m) in enumerate(zip(energies, mults))]
        return cls(states=states, **kw)

    @classmethod
    def coerce(cls, value) -> "EnergyResult":
        """Turn whatever a user function returns into an EnergyResult."""
        if isinstance(value, EnergyResult):
            return value
        if isinstance(value, StateInfo):
            return cls(states=[value])
        if isinstance(value, dict):
            if "states" in value:
                return cls(states=list(value["states"]),
                           extra={k: v for k, v in value.items() if k != "states"})
            if "energies" in value:
                return cls.from_energies(value["energies"], value.get("multiplicities"))
            if "energy" in value:
                return cls.from_energies([value["energy"]])
        # anything numpy-like: scalar or 1D list of state energies
        try:
            arr = np.asarray(_to_host(value), dtype=float)
        except Exception as exc:
            raise TypeError(f"Cannot interpret energy result of type {type(value)}") from exc
        return cls.from_energies(arr.ravel())


def _to_host(value):
    """Bring torch / cupy values to plain Python / numpy."""
    if hasattr(value, "detach"):  # torch
        return value.detach().cpu().numpy()
    if hasattr(value, "get") and type(value).__module__.startswith("cupy"):
        return value.get()
    return value
