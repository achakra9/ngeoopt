"""Backend that calls a Python function for the energy.

The function receives the element symbols and the coordinates and returns
an energy, a list of state energies, a dict, or an ``EnergyResult``::

    def energy(symbols, coords, device="cpu", **kwargs):
        ...
        return e_hartree

If the function accepts a ``device`` argument it gets the worker's device
("cpu", "cuda:0" or "mps"), so torch-based models (ML potentials etc.) run on
the GPU of whichever worker picks up the job. torch / cupy return values are
copied back to the host automatically.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict

from ..geometry.molecule import Molecule
from ..results import EnergyResult
from ..units import BOHR_TO_ANGSTROM, energy_to_hartree
from .base import EnergyBackend, JobContext
from .template import load_callable


@dataclass
class FunctionBackend(EnergyBackend):
    function: Any = None                  # callable or 'module:function' / 'file.py:function'
    kwargs: Dict[str, Any] = field(default_factory=dict)
    coords_units: str = "bohr"            # bohr | angstrom
    energy_units: str = "hartree"
    pass_molecule: bool = False           # call f(molecule, **kw) instead of f(symbols, coords)

    name = "function"

    def __post_init__(self):
        if self.function is None:
            raise ValueError("function backend needs `function`")
        # resolve once to fail early on typos; strings stay strings so the
        # backend remains picklable for worker processes
        load_callable(self.function)

    def describe(self) -> str:
        f = self.function
        return f"function:{f if isinstance(f, str) else getattr(f, '__qualname__', f)}"

    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        fn = load_callable(self.function)
        kw = dict(self.kwargs)
        try:
            params = inspect.signature(fn).parameters
            takes_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
            if "device" in params or takes_var_kw:
                kw.setdefault("device", ctx.device)
            if "workdir" in params:
                kw.setdefault("workdir", ctx.workdir)
        except (TypeError, ValueError):
            pass
        if self.pass_molecule:
            out = fn(mol, **kw)
        else:
            coords = mol.coords * (BOHR_TO_ANGSTROM if self.coords_units.startswith("ang") else 1.0)
            out = fn(list(mol.symbols), coords.copy(), **kw)
        res = EnergyResult.coerce(out)
        if self.energy_units.lower() not in ("hartree", "au"):
            for s in res.states:
                s.energy = energy_to_hartree(s.energy, self.energy_units)
        return res
