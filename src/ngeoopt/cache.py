"""On-disk cache of single-point results.

Each finished job is written as soon as it completes, keyed by a hash of the
geometry and the backend settings. If a run is killed in the middle of a
gradient, a restart only recomputes the displacements that never finished.

Entries are pickles written by ngeoopt itself; do not point the cache at a
directory you did not create.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry.molecule import Molecule
from .results import EnergyResult


def geometry_key(mol: Molecule, backend_tag: str, decimals: int = 9) -> str:
    h = hashlib.sha1()
    h.update(backend_tag.encode())
    h.update(f"{mol.charge}:{mol.multiplicity}:{','.join(mol.symbols)}".encode())
    h.update(np.round(mol.coords, decimals).tobytes())
    return h.hexdigest()


def backend_tag(backend) -> str:
    """Identify a backend by *all* of its settings, so changing e.g. conv_tol
    or the grid never reuses stale energies from the cache."""
    # drop memory addresses (e.g. a callable passed directly) so the tag is
    # stable across runs and restarts still find their cache
    text = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(backend))
    return f"{backend.describe()}#{hashlib.sha1(text.encode()).hexdigest()[:16]}"


class ResultCache:
    def __init__(self, directory, backend_tag: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = backend_tag

    def _path(self, mol: Molecule) -> Path:
        return self.dir / f"{geometry_key(mol, self.tag)}.pkl"

    def get(self, mol: Molecule) -> Optional[EnergyResult]:
        p = self._path(mol)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None  # truncated write from a killed job; recompute

    def put(self, mol: Molecule, result: EnergyResult) -> None:
        p = self._path(mol)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)

    def __len__(self) -> int:
        return len(list(self.dir.glob("*.pkl")))
