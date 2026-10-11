"""YAML/dict configuration -> fully built run objects.

Every section maps onto a settings dataclass, and unknown keywords are
rejected with the list of valid ones so typos never pass silently.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .findiff import FDSettings
from .geometry.molecule import Molecule
from .optimizer import ConvergenceCriteria, OptimizerSettings
from .tracking import TrackingSettings

SECTIONS = ("task", "molecule", "backend", "state", "gradient", "optimizer",
            "parallel", "output")


@dataclass
class ParallelSettings:
    executor: str = "process"        # serial | process | mpi
    workers: Any = "auto"            # int or 'auto'
    threads_per_worker: int = 1
    device: str = "auto"             # auto | cpu | cuda | cuda:N | mps
    gpus: Optional[list] = None      # restrict to these CUDA ids
    workers_per_gpu: int = 1
    retries: int = 0                 # re-run a failed single point this many times


@dataclass
class OutputSettings:
    workdir: str = "ngeoopt_run"
    checkpoint: str = "ngeoopt_state.npz"
    trajectory: Optional[str] = "trajectory.xyz"
    final_xyz: str = "final.xyz"
    keep_files: bool = False         # keep per-job scratch directories
    restart: bool = True             # resume from checkpoint/cache if present
    cache: bool = True
    log_file: Optional[str] = "ngeoopt.log"
    log_level: str = "info"


@dataclass
class MoleculeSettings:
    xyz: Optional[str] = None
    geometry: Optional[str] = None   # inline "El x y z" lines
    charge: int = 0
    multiplicity: int = 1
    units: str = "angstrom"


@dataclass
class RunConfig:
    task: str = "optimize"           # optimize | gradient | energy
    molecule: MoleculeSettings = field(default_factory=MoleculeSettings)
    backend: Dict[str, Any] = field(default_factory=dict)
    state: TrackingSettings = field(default_factory=TrackingSettings)
    gradient: FDSettings = field(default_factory=FDSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    parallel: ParallelSettings = field(default_factory=ParallelSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    base_dir: Path = field(default_factory=Path.cwd)
    given: Dict[str, set] = field(default_factory=dict)   # keywords set explicitly, per section

    def build_molecule(self) -> Molecule:
        m = self.molecule
        if m.xyz:
            p = Path(m.xyz)
            if not p.is_absolute():
                p = self.base_dir / p
            return Molecule.from_xyz(p, m.charge, m.multiplicity, m.units)
        if m.geometry:
            return Molecule.from_lines(m.geometry, m.charge, m.multiplicity, m.units)
        raise ValueError("molecule section needs `xyz` or `geometry`")


def _build(cls, data: Optional[dict], section: str):
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown keyword(s) in [{section}]: {sorted(unknown)}; "
                         f"valid: {sorted(names)}")
    return cls(**data)


def _convergence(spec) -> ConvergenceCriteria:
    if spec is None:
        return ConvergenceCriteria()
    if isinstance(spec, str):
        return ConvergenceCriteria.preset(spec)
    spec = dict(spec)
    preset = spec.pop("preset", None)
    if preset:
        return ConvergenceCriteria.preset(preset, **spec)
    return _build(ConvergenceCriteria, spec, "optimizer.convergence")


def config_from_dict(d: Dict[str, Any], base_dir: Optional[Path] = None) -> RunConfig:
    d = dict(d or {})
    unknown = set(d) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}; valid: {list(SECTIONS)}")
    opt = dict(d.get("optimizer") or {})
    conv = _convergence(opt.pop("convergence", None))
    optimizer = _build(OptimizerSettings, opt, "optimizer")
    optimizer.convergence = conv
    backend = dict(d.get("backend") or {})
    if "name" not in backend:
        raise ValueError("backend section needs `name` (pyscf | qchem | template | function)")
    task = d.get("task", "optimize")
    if task not in ("optimize", "gradient", "energy"):
        raise ValueError("task must be optimize | gradient | energy")
    return RunConfig(
        task=task,
        molecule=_build(MoleculeSettings, d.get("molecule"), "molecule"),
        backend=backend,
        state=_build(TrackingSettings, d.get("state"), "state"),
        gradient=_build(FDSettings, d.get("gradient"), "gradient"),
        optimizer=optimizer,
        parallel=_build(ParallelSettings, d.get("parallel"), "parallel"),
        output=_build(OutputSettings, d.get("output"), "output"),
        base_dir=Path(base_dir or Path.cwd()),
        given={k: set((d.get(k) or {}).keys()) for k in SECTIONS
               if isinstance(d.get(k), dict)},
    )


def load_config(path) -> RunConfig:
    path = Path(path)
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data, base_dir=path.parent.resolve())


EXAMPLE_CONFIG = """\
# ngeoopt input -- every keyword is optional except molecule and backend.name
task: optimize                 # optimize | gradient | energy

molecule:
  xyz: water.xyz               # or: geometry: |  (inline "El x y z" lines)
  charge: 0
  multiplicity: 1
  units: angstrom

backend:
  name: pyscf                  # pyscf | qchem | template | function
  method: b3lyp                # hf | <any xc> | mp2 | ccsd | ccsd(t) | casscf | casci
  basis: 6-31g*
  excited: null                # tddft | tda | cis | tdhf | eom-ccsd
  nroots: 5
  spin_states: singlet         # singlet | triplet | both
  conv_tol: 1.0e-10
  grids_level: 4
  density_fit: false
  use_gpu: auto                # auto | never | always (gpu4pyscf on CUDA)

state:                         # which state to optimize and how to follow it
  root: 0                      # 0 = ground / lowest state of `multiplicity`
  multiplicity: null
  method: auto                 # auto | overlap | character | energy | index
  min_overlap: 0.5
  follow: true

gradient:
  scheme: central              # forward | backward | central | central4
  step: 0.005                  # Bohr
  project_rigid: true
  frozen_atoms: []             # 0-based atom indices

optimizer:
  method: rfo                  # rfo | qn | sd
  max_cycles: 100
  hessian_init: lindh          # lindh | unit
  hessian_update: bfgs         # bfgs | sr1 | psb | bofill | none
  trust_radius: 0.3
  trust_min: 0.001
  trust_max: 1.0
  convergence: gaussian        # loose | gaussian | qchem | tight | verytight, or a dict

parallel:
  executor: process            # serial | process | mpi
  workers: auto
  threads_per_worker: 1
  device: auto                 # auto | cpu | cuda | cuda:N | mps
  gpus: null
  workers_per_gpu: 1
  retries: 0

output:
  workdir: ngeoopt_run
  keep_files: false
  restart: true
  log_level: info
"""
