"""Q-Chem backend.

Every job runs as ``qchem -nt <threads> job.in job.out`` in its own directory
with a private ``QCSCRATCH``, so many displacements can run concurrently.

Outputs are read by the built-in parser (``qchem_parser``): SCF/DFT,
TDDFT/TDA/CIS, SF-TDDFT and MRSF-TDDFT. Correlated methods (MP2/CC/EOM) are
not parsed yet; use ``energy_regex`` / ``states_regex`` or a ``parser`` for
those, exactly as for the generic template backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..geometry.molecule import Molecule
from ..results import EnergyResult
from .base import JobContext
from .qchem_parser import check_correlated, parse_qchem_output
from .template import TemplateBackend, geometry_block

log = logging.getLogger(__name__)

_GENERATED = "<generated>"


def rem_block(rem: Dict[str, Any]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)
    width = max((len(k) for k in rem), default=0)
    return "$rem\n" + "\n".join(f"   {k:<{width}s}  {fmt(v)}" for k, v in rem.items()) + "\n$end"


def _rem_value(rem: Dict[str, Any], key: str):
    for k, v in rem.items():
        if k.lower() == key:
            return v
    return None


@dataclass
class QChemBackend(TemplateBackend):
    rem: Dict[str, Any] = field(default_factory=dict)
    extra_sections: str = ""               # appended verbatim ($basis, $solvent, ...)
    command: str = "qchem -nt {nthreads} {input} {output}"
    template: Optional[str] = _GENERATED
    input_name: str = "job.in"
    output_name: str = "job.out"
    scratch_env: Optional[str] = "QCSCRATCH"
    success_regex: Optional[str] = "Thank you very much for using Q-Chem"
    sym_ignore: bool = True                # never let Q-Chem symmetrize displaced geometries
    signatures: bool = True                # amplitude / state-DM descriptors for root following

    name = "qchem"

    def __post_init__(self):
        if self.template_file:
            self.template = None           # read the file in TemplateBackend
        elif self.template == _GENERATED and not self.rem:
            raise ValueError("qchem backend needs a `rem` dict or a `template_file`")
        if self.rem and self.sym_ignore and _rem_value(self.rem, "sym_ignore") is None \
                and _rem_value(self.rem, "symmetry") is None:
            # a 0.005 Bohr displacement can fall inside Q-Chem's symmetry
            # tolerance and be symmetrized away, which breaks the stencil
            self.rem = {**self.rem, "sym_ignore": True}
        super().__post_init__()
        if self.rem and self.energy_regex is None and self.parser is None:
            check_correlated(self.rem)
        self._warn_tolerances()

    def _warn_tolerances(self) -> None:
        if not self.rem:
            return
        scf = _rem_value(self.rem, "scf_convergence")
        if scf is None or int(scf) < 9:
            log.warning("Q-Chem SCF_CONVERGENCE=%s; use 9-10 for finite differences "
                        "(energy noise / step = gradient noise)", scf or "default")
        cis = _rem_value(self.rem, "cis_convergence")
        if _rem_value(self.rem, "cis_n_roots") is not None and (cis is None or int(cis) < 7):
            log.warning("Q-Chem CIS_CONVERGENCE=%s; use >= 7 for excited-state finite "
                        "differences", cis or "default")
        method = str(_rem_value(self.rem, "method") or "").lower()
        if method not in ("", "hf") and _rem_value(self.rem, "xc_grid") is None:
            log.warning("Q-Chem uses the SG-1 grid by default; set XC_GRID 3 (or finer) "
                        "for smooth DFT finite differences")

    def make_input(self, mol: Molecule, ctx: JobContext) -> str:
        if self.template != _GENERATED:
            return super().make_input(mol, ctx)
        mol_block = (f"$molecule\n{mol.charge} {mol.multiplicity}\n"
                     f"{geometry_block(mol, 'angstrom')}\n$end")
        parts = [mol_block, rem_block(self.rem)]
        if self.extra_sections.strip():
            parts.append(self.extra_sections.strip())
        return "\n\n".join(parts) + "\n"

    def parse_output(self, text: str, ctx: JobContext) -> EnergyResult:
        if self.parser is not None or self.energy_regex is not None:
            return super().parse_output(text, ctx)
        return parse_qchem_output(text, signatures=self.signatures)

    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        res = super().compute(mol, ctx)
        for s in res.states:
            if s.extra.get("scf_reference") and s.multiplicity is None:
                s.multiplicity = mol.multiplicity
        return res
