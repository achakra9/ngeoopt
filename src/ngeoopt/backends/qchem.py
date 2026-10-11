"""Q-Chem backend.

Input generation and job execution are in place: every job runs as
``qchem -nt <threads> job.in job.out`` in its own directory with a private
``QCSCRATCH``, so many displacements can run concurrently.

Built-in output parsing (SCF/DFT, TDDFT/CIS, SF/MRSF-TDDFT, MP2/CC, EOM) is
NOT written yet -- it will be added against reference Q-Chem outputs. Until
then, pass ``energy_regex`` / ``states_regex`` or your own ``parser`` exactly
as for the generic template backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..geometry.molecule import Molecule
from ..results import EnergyResult
from .base import JobContext
from .template import TemplateBackend, geometry_block


def rem_block(rem: Dict[str, Any]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)
    width = max((len(k) for k in rem), default=0)
    return "$rem\n" + "\n".join(f"   {k:<{width}s}  {fmt(v)}" for k, v in rem.items()) + "\n$end"


@dataclass
class QChemBackend(TemplateBackend):
    rem: Dict[str, Any] = field(default_factory=dict)
    extra_sections: str = ""               # appended verbatim ($basis, $solvent, ...)
    command: str = "qchem -nt {nthreads} {input} {output}"
    template: Optional[str] = "<generated>"
    input_name: str = "job.in"
    output_name: str = "job.out"
    scratch_env: Optional[str] = "QCSCRATCH"

    name = "qchem"

    def __post_init__(self):
        if self.template == "<generated>" and not self.template_file and not self.rem:
            raise ValueError("qchem backend needs a `rem` dict or a `template_file`")
        super().__post_init__()

    def make_input(self, mol: Molecule, ctx: JobContext) -> str:
        if self.template_file or self.template != "<generated>":
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
        raise NotImplementedError(
            "Built-in Q-Chem output parsing is not implemented yet. Set `energy_regex` "
            "(and `states_regex` for excited states) or a custom `parser` in the backend "
            "options for now.")
