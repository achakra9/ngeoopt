"""Molecule container used throughout the optimizer (coordinates in Bohr)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Union

import numpy as np

from ..units import BOHR_TO_ANGSTROM, length_factor
from .xyz import read_xyz, read_xyz_text, xyz_to_text

PathLike = Union[str, Path]

# Atomic masses for the elements most likely to show up; used only for
# mass-weighted quantities. Unknown elements fall back to 12.0.
_MASSES = {
    "H": 1.00784, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "I": 126.90, "Xe": 131.29, "Pt": 195.08, "Au": 196.97,
}

# Covalent radii in Angstrom (Alvarez 2008), used for the model Hessian
_COV_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ti": 1.60,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "I": 1.39, "Xe": 1.40, "Pt": 1.36, "Au": 1.36,
}


def atomic_mass(symbol: str) -> float:
    return _MASSES.get(symbol, 12.0)


def covalent_radius(symbol: str) -> float:
    """Covalent radius in Bohr."""
    return _COV_RADII.get(symbol, 1.5) / BOHR_TO_ANGSTROM


@dataclass
class Molecule:
    symbols: List[str]
    coords: np.ndarray  # (N, 3) in Bohr
    charge: int = 0
    multiplicity: int = 1
    comment: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError("Number of symbols does not match number of coordinates.")

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def coords_angstrom(self) -> np.ndarray:
        return self.coords * BOHR_TO_ANGSTROM

    @property
    def masses(self) -> np.ndarray:
        return np.array([atomic_mass(s) for s in self.symbols])

    def with_coords(self, coords: np.ndarray) -> "Molecule":
        return replace(self, coords=np.array(coords, dtype=float).reshape(-1, 3))

    def to_xyz(self, comment: str | None = None) -> str:
        return xyz_to_text(self.symbols, self.coords_angstrom,
                           comment=self.comment if comment is None else comment)

    @classmethod
    def from_xyz(cls, path: PathLike, charge: int = 0, multiplicity: int = 1,
                 units: str = "angstrom") -> "Molecule":
        xyz = read_xyz(path)
        return cls(xyz.symbols, xyz.coords * length_factor(units), charge,
                   multiplicity, xyz.comment)

    @classmethod
    def from_xyz_text(cls, text: str, charge: int = 0, multiplicity: int = 1,
                      units: str = "angstrom") -> "Molecule":
        xyz = read_xyz_text(text)
        return cls(xyz.symbols, xyz.coords * length_factor(units), charge,
                   multiplicity, xyz.comment)

    @classmethod
    def from_lines(cls, text: str, charge: int = 0, multiplicity: int = 1,
                   units: str = "angstrom") -> "Molecule":
        """Build from plain 'El x y z' lines (no atom-count header)."""
        rows = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
        symbols = [r[0] for r in rows]
        coords = np.array([[float(v) for v in r[1:4]] for r in rows])
        return cls(symbols, coords * length_factor(units), charge, multiplicity)


def rigid_body_modes(coords: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal basis (k, 3N) spanning translations and rotations.

    Uses unit weights unless masses are given. Linear molecules give 5 modes.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = coords.shape[0]
    w = np.ones(n) if masses is None else np.asarray(masses, dtype=float)
    com = (w[:, None] * coords).sum(0) / w.sum()
    r = coords - com
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        vecs.append(t.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        vecs.append(np.cross(axis, r).ravel())
    m = np.array(vecs).T
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    keep = s > 1e-6 * max(s.max(), 1.0)
    return u[:, keep].T


def project_rigid(vector: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Remove translational and rotational components from a (N,3) vector."""
    v = np.asarray(vector, dtype=float).ravel()
    basis = rigid_body_modes(coords)
    v = v - basis.T @ (basis @ v)
    return v.reshape(-1, 3)
