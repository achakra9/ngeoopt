"""Generic backend for any program that prints an energy.

You supply:
  * an input template with ``{{placeholders}}`` (double braces, so they never
    clash with $-sections, %blocks or single braces of real input formats),
  * the command line to run it,
  * how to read the energy back: a regex, or your own parser function.

Placeholders available in the template:
  {{geometry}}      "El  x  y  z" lines (units: ``geometry_units``)
  {{charge}} {{multiplicity}} {{natoms}} {{nthreads}} {{label}} {{workdir}}
  {{xyz}}           a complete XYZ file (count line, comment, coordinates)
  any key of ``variables``

Placeholders in ``command``: {input} {output} {nthreads} {workdir} {label}
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..geometry.molecule import Molecule
from ..results import EnergyResult
from ..units import BOHR_TO_ANGSTROM, energy_to_hartree
from .base import EnergyBackend, JobContext

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_callable(spec: Union[str, Callable]) -> Callable:
    """Resolve 'package.module:function' or '/path/file.py:function'."""
    if callable(spec):
        return spec
    if ":" not in spec:
        raise ValueError(f"callable spec must look like 'module:function', got {spec!r}")
    mod_name, func_name = spec.rsplit(":", 1)
    if mod_name.endswith(".py") or os.sep in mod_name:
        path = Path(mod_name).expanduser().resolve()
        mspec = importlib.util.spec_from_file_location(path.stem, path)
        if mspec is None or mspec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(mspec)
        sys.modules.setdefault(path.stem, module)
        mspec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_name)
    return getattr(module, func_name)


def render(template: str, values: Dict[str, Any]) -> str:
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
        return str(values[key])
    return _PLACEHOLDER.sub(sub, template)


def geometry_block(mol: Molecule, units: str = "angstrom", fmt: str = "{:<3s} {: .10f} {: .10f} {: .10f}") -> str:
    x = mol.coords * (BOHR_TO_ANGSTROM if units.lower().startswith("ang") else 1.0)
    return "\n".join(fmt.format(s, *c) for s, c in zip(mol.symbols, x))


@dataclass
class TemplateBackend(EnergyBackend):
    command: str = ""
    template: Optional[str] = None
    template_file: Optional[str] = None
    input_name: str = "job.inp"
    output_name: str = "job.out"
    geometry_units: str = "angstrom"
    variables: Dict[str, Any] = field(default_factory=dict)
    energy_regex: Optional[str] = None
    energy_match: str = "last"            # last | first
    states_regex: Optional[str] = None    # extra state energies, every match is a state
    states_are_excitations: bool = False  # states_regex gives excitation energies
    states_units: Optional[str] = None    # defaults to energy_units
    energy_units: str = "hartree"
    parser: Any = None                    # 'module:function' -> f(text, workdir) -> energies/EnergyResult
    success_regex: Optional[str] = None   # must be found in the output, else the job failed
    env: Dict[str, str] = field(default_factory=dict)
    scratch_env: Optional[str] = None     # env var pointed at a private scratch dir
    copy_files: List[str] = field(default_factory=list)
    timeout: Optional[float] = None       # seconds
    shell: bool = True

    name = "template"

    def __post_init__(self):
        if self.template is None and self.template_file:
            self.template = Path(self.template_file).expanduser().read_text()
        if self.template is None:
            raise ValueError("template backend needs `template` or `template_file`")
        if not self.command:
            raise ValueError("template backend needs `command`")
        if self.energy_regex is None and self.parser is None and type(self) is TemplateBackend:
            raise ValueError("template backend needs `energy_regex` or `parser`")
        self.copy_files = [str(Path(f).expanduser().resolve()) for f in self.copy_files]

    def describe(self) -> str:
        return f"{self.name}:{self.command.split()[0] if self.command else '?'}"

    # -- steps that subclasses may override -------------------------------
    def make_input(self, mol: Molecule, ctx: JobContext) -> str:
        values = {
            "geometry": geometry_block(mol, self.geometry_units),
            "xyz": mol.to_xyz(comment=ctx.label),
            "charge": mol.charge,
            "multiplicity": mol.multiplicity,
            "natoms": mol.natoms,
            "nthreads": ctx.nthreads,
            "label": ctx.label,
            "workdir": str(ctx.workdir),
            **self.variables,
        }
        return render(self.template, values)

    def run_program(self, ctx: JobContext) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.env.items()})
        env["OMP_NUM_THREADS"] = str(ctx.nthreads)
        if self.scratch_env:
            scr = ctx.workdir / "scratch"
            scr.mkdir(exist_ok=True)
            env[self.scratch_env] = str(scr)
        cmd = self.command.format(input=self.input_name, output=self.output_name,
                                  nthreads=ctx.nthreads, workdir=str(ctx.workdir),
                                  label=ctx.label)
        return subprocess.run(cmd if self.shell else cmd.split(), shell=self.shell,
                              cwd=ctx.workdir, env=env, capture_output=True, text=True,
                              timeout=self.timeout)

    def parse_output(self, text: str, ctx: JobContext) -> EnergyResult:
        if self.parser is not None:
            fn = load_callable(self.parser)
            return EnergyResult.coerce(fn(text, ctx.workdir))
        matches = re.findall(self.energy_regex, text, flags=re.MULTILINE)
        if not matches:
            raise RuntimeError(f"energy_regex {self.energy_regex!r} found nothing in "
                               f"{ctx.workdir / self.output_name}")
        m = matches[-1] if self.energy_match == "last" else matches[0]
        e0 = energy_to_hartree(float(m[0] if isinstance(m, tuple) else m), self.energy_units)
        energies = [e0]
        if self.states_regex:
            unit = self.states_units or self.energy_units
            for m in re.findall(self.states_regex, text, flags=re.MULTILINE):
                v = energy_to_hartree(float(m[0] if isinstance(m, tuple) else m), unit)
                energies.append(e0 + v if self.states_are_excitations else v)
        res = EnergyResult.from_energies(energies)
        res.states[0].extra["scf_reference"] = bool(self.states_regex)
        return res

    # -- main entry --------------------------------------------------------
    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        for f in self.copy_files:
            shutil.copy(f, ctx.workdir)
        (ctx.workdir / self.input_name).write_text(self.make_input(mol, ctx))
        proc = self.run_program(ctx)
        out_path = ctx.workdir / self.output_name
        text = out_path.read_text(errors="replace") if out_path.exists() else proc.stdout
        if proc.returncode != 0:
            raise RuntimeError(f"command exited with {proc.returncode} in {ctx.workdir}\n"
                               f"stderr: {proc.stderr[-2000:]}")
        if self.success_regex and not re.search(self.success_regex, text):
            raise RuntimeError(f"success_regex not found in {out_path}")
        res = self.parse_output(text, ctx)
        res.output_file = str(out_path)
        return res
