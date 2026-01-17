## Main driver to read XYZ coordinate files

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class XYZ:
    symbols: List[str]
    coords: np.ndarray  # shape (N, 3)
    comment: str = ""


def _normalize_symbol(sym: str) -> str:
    sym = sym.strip()
    if not sym:
        raise ValueError("Empty element symbol in XYZ.")
    return sym[0].upper() + sym[1:].lower()


def read_xyz(path: PathLike) -> XYZ:
    """
    Read an XYZ file from disk.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return read_xyz_text(text)


def read_xyz_text(text: str) -> XYZ:
    """
    Read XYZ content from a string.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("XYZ must contain at least two lines (N and comment).")

    try:
        n_atoms = int(lines[0].split()[0])
    except Exception:
        raise ValueError(f"Invalid atom count line: {lines[0]!r}")

    comment = lines[1]
    geom_lines = lines[2:]

    if len(geom_lines) < n_atoms:
        raise ValueError(
            f"XYZ claims {n_atoms} atoms but only {len(geom_lines)} geometry lines found."
        )

    symbols: List[str] = []
    coords = np.zeros((n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        parts = geom_lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid geometry line: {geom_lines[i]!r}")

        symbols.append(_normalize_symbol(parts[0]))
        try:
            coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except Exception:
            raise ValueError(f"Could not parse coordinates in line: {geom_lines[i]!r}")

    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or infinity.")

    return XYZ(symbols=symbols, coords=coords, comment=comment)


def xyz_to_text(
    symbols: Iterable[str],
    coords: np.ndarray,
    comment: str = ""
) -> str:
    """
    Convert symbols + coordinates to XYZ formatted text.
    """
    symbols = list(symbols)
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
    if len(symbols) != coords.shape[0]:
        raise ValueError("Number of symbols must match number of coordinate rows.")

    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        sym = _normalize_symbol(sym)
        lines.append(f"{sym:<2s} {x: .10f} {y: .10f} {z: .10f}")

    return "\n".join(lines) + "\n"


def write_xyz(
    path: PathLike,
    symbols: Iterable[str],
    coords: np.ndarray,
    comment: str = ""
) -> None:
    """
    Write an XYZ file to disk.
    """
    Path(path).write_text(
        xyz_to_text(symbols, coords, comment=comment),
        encoding="utf-8"
    )
