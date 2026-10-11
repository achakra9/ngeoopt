"""High-level entry points tying config, backend, executor and optimizer together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import __version__
from .backends import EnergyBackend, make_backend
from .cache import ResultCache, backend_tag
from .config import RunConfig, config_from_dict, load_config
from .findiff import FDSettings, FiniteDifferenceGradient, GradientResult
from .geometry.molecule import Molecule
from .optimizer import OptimizationResult, Optimizer, OptimizerSettings
from .parallel import make_executor
from .tracking import StateTracker, TrackingSettings

log = logging.getLogger("ngeoopt")


def setup_logging(level: str = "info", log_file: Optional[Path] = None) -> list:
    """Attach console (+ file) handlers to the 'ngeoopt' logger; returns them."""
    root = logging.getLogger("ngeoopt")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    return list(root.handlers[-2:] if log_file else root.handlers[-1:])


def teardown_logging(handlers) -> None:
    root = logging.getLogger("ngeoopt")
    for h in handlers:
        root.removeHandler(h)
        h.close()
    root.propagate = True


def _resolve(path: str, base: Path) -> str:
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def _resolve_spec(spec, base: Path):
    """'file.py:func' -> absolute file path; 'package.module:func' unchanged."""
    if not isinstance(spec, str) or ":" not in spec:
        return spec
    mod, func = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        return f"{_resolve(mod, base)}:{func}"
    return spec


def resolve_backend_paths(opts: dict, base: Path) -> dict:
    """Make file paths in the backend section relative to the input file."""
    for key in ("template_file", "input_file"):
        if opts.get(key):
            opts[key] = _resolve(opts[key], base)
    if opts.get("copy_files"):
        opts["copy_files"] = [_resolve(f, base) for f in opts["copy_files"]]
    for key in ("parser", "function"):
        if key in opts:
            opts[key] = _resolve_spec(opts[key], base)
    return opts


@dataclass
class RunResult:
    task: str
    molecule: Molecule
    energy: float
    gradient: Optional[np.ndarray] = None
    optimization: Optional[OptimizationResult] = None
    gradient_result: Optional[GradientResult] = None


def run(config: Union[str, Path, dict, RunConfig]) -> RunResult:
    """Run a calculation from a YAML path, a dict or a RunConfig."""
    if isinstance(config, (str, Path)):
        cfg = load_config(config)
    elif isinstance(config, dict):
        cfg = config_from_dict(config)
    else:
        cfg = config
    out = cfg.output
    workdir = Path(out.workdir)
    if not workdir.is_absolute():
        workdir = cfg.base_dir / workdir
    workdir.mkdir(parents=True, exist_ok=True)
    handlers = setup_logging(out.log_level, workdir / out.log_file if out.log_file else None)
    try:
        return _run(cfg, workdir)
    finally:
        teardown_logging(handlers)


def _run(cfg: RunConfig, workdir: Path) -> RunResult:
    out = cfg.output
    log.info("ngeoopt %s  task=%s  workdir=%s", __version__, cfg.task, workdir)

    mol = cfg.build_molecule()
    bopts = resolve_backend_paths(dict(cfg.backend), cfg.base_dir)
    backend = make_backend(bopts.pop("name"), **bopts)
    log.info("backend: %s", backend.describe())
    if backend.name == "pyscf" and cfg.parallel.device in ("auto", "mps"):
        from .device import detect
        if detect().kind == "mps":
            log.info("PySCF has no Apple Metal backend; energies run on CPU cores "
                     "(MPS is used only by torch-based function backends)")

    if backend.name == "qchem" and backend.parser is None and backend.energy_regex is None:
        # Q-Chem prints excited-state energies with 8 decimals: 5e-9 Eh rounding
        # becomes ~1e-6 Eh/Bohr gradient noise at h = 0.005
        if "step" not in cfg.given.get("gradient", set()):
            cfg.gradient.step = 0.01
            log.info("Q-Chem: finite-difference step set to 0.01 Bohr (energies are printed "
                     "with 8 decimals); set gradient.step to override")
        if cfg.optimizer.convergence.max_force < 1e-4:
            log.warning("Q-Chem energies carry ~5e-9 Eh rounding: gradients are good to "
                        "~5e-7 Eh/Bohr at best, so tight/verytight criteria may not be reachable")

    cache = ResultCache(workdir / "cache", backend_tag(backend)) if out.cache else None
    par = cfg.parallel
    with make_executor(par.executor, par.workers, par.threads_per_worker, par.device,
                       par.gpus, par.workers_per_gpu, scratch=workdir / "jobs",
                       keep_files=out.keep_files, retries=par.retries) as ex:
        tracker = StateTracker(cfg.state)
        fd = FiniteDifferenceGradient(backend, ex, tracker, cfg.gradient, cache)
        if cfg.task == "energy":
            state, _ = fd.energy(mol, label="energy")
            log.info("energy = %.12f Eh (root %d)", state.energy, state.index)
            return RunResult("energy", mol, state.energy)
        if cfg.task == "gradient":
            gr = fd.compute(mol, prefix="gradient")
            log.info("energy = %.12f Eh, max|g| = %.3e", gr.energy, gr.max_force)
            np.savetxt(workdir / "gradient.txt", gr.gradient, fmt="% .10e",
                       header="Cartesian gradient, Hartree/Bohr")
            return RunResult("gradient", mol, gr.energy, gr.gradient, gradient_result=gr)
        opt = Optimizer(fd, cfg.optimizer, workdir=workdir, checkpoint=out.checkpoint,
                        trajectory=out.trajectory)
        res = opt.run(mol, restart=out.restart)
    (workdir / out.final_xyz).write_text(
        res.molecule.to_xyz(comment=f"E= {res.energy:.10f} converged={res.converged}"))
    log.info("final energy %.12f Eh after %d cycles (%s)", res.energy, res.cycles,
             "converged" if res.converged else "NOT converged")
    return RunResult("optimize", res.molecule, res.energy, res.gradient, optimization=res)


def optimize(mol: Molecule, backend: EnergyBackend, *, workdir: str = "ngeoopt_run",
             executor: str = "process", workers="auto", threads_per_worker: int = 1,
             device: str = "auto", state: Optional[TrackingSettings] = None,
             gradient: Optional[FDSettings] = None,
             optimizer: Optional[OptimizerSettings] = None,
             restart: bool = True, cache: bool = True) -> OptimizationResult:
    """Python API: optimize ``mol`` with any backend object."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    rc = ResultCache(wd / "cache", backend_tag(backend)) if cache else None
    with make_executor(executor, workers, threads_per_worker, device,
                       scratch=wd / "jobs") as ex:
        fd = FiniteDifferenceGradient(backend, ex, StateTracker(state), gradient, rc)
        return Optimizer(fd, optimizer, workdir=wd).run(mol, restart=restart)


def gradient(mol: Molecule, backend: EnergyBackend, *, executor: str = "process",
             workers="auto", threads_per_worker: int = 1, device: str = "auto",
             settings: Optional[FDSettings] = None,
             state: Optional[TrackingSettings] = None,
             scratch: str = "ngeoopt_scratch") -> GradientResult:
    """Python API: one finite-difference gradient."""
    with make_executor(executor, workers, threads_per_worker, device, scratch=scratch) as ex:
        fd = FiniteDifferenceGradient(backend, ex, StateTracker(state), settings)
        return fd.compute(mol)
