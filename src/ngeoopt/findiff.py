"""Finite-difference Cartesian gradients from energy-only backends."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backends.base import EnergyBackend
from .cache import ResultCache
from .geometry.molecule import Molecule, project_rigid
from .parallel.executors import BaseExecutor, Job
from .results import EnergyResult, StateInfo
from .tracking import StateTracker

log = logging.getLogger(__name__)

# stencil: list of (multiple of h, weight); gradient = sum(w * E(x + m h)) / h
STENCILS: Dict[str, List[Tuple[int, float]]] = {
    "forward": [(1, 1.0), (0, -1.0)],
    "backward": [(0, 1.0), (-1, -1.0)],
    "central": [(1, 0.5), (-1, -0.5)],
    "central4": [(2, -1.0 / 12), (1, 8.0 / 12), (-1, -8.0 / 12), (-2, 1.0 / 12)],
}

_AXES = "xyz"


@dataclass
class FDSettings:
    scheme: str = "central"
    step: float = 0.005                       # Bohr
    project_rigid: bool = True                # remove translation/rotation noise
    frozen_atoms: Sequence[int] = field(default_factory=list)   # 0-based
    use_cache: bool = True

    def __post_init__(self):
        if self.scheme not in STENCILS:
            raise ValueError(f"scheme must be one of {sorted(STENCILS)}")
        if self.step <= 0:
            raise ValueError("finite-difference step must be positive")


@dataclass
class GradientResult:
    energy: float
    gradient: np.ndarray          # (N, 3) Hartree/Bohr
    state: StateInfo
    center: EnergyResult
    njobs: int
    ncached: int
    wall_time: float
    tracking: dict = field(default_factory=dict)

    @property
    def max_force(self) -> float:
        return float(np.abs(self.gradient).max())

    @property
    def rms_force(self) -> float:
        return float(np.sqrt(np.mean(self.gradient ** 2)))


def displacement_plan(mol: Molecule, settings: FDSettings):
    """List of (label, atom, axis, multiple) for every displaced geometry."""
    frozen = set(settings.frozen_atoms)
    plan = []
    for a in range(mol.natoms):
        if a in frozen:
            continue
        for k in range(3):
            for m, _ in STENCILS[settings.scheme]:
                if m == 0:
                    continue
                sign = "+" if m > 0 else "-"
                plan.append((f"a{a:03d}{_AXES[k]}{sign}{abs(m)}", a, k, m))
    return plan


class FiniteDifferenceGradient:
    def __init__(self, backend: EnergyBackend, executor: BaseExecutor,
                 tracker: Optional[StateTracker] = None,
                 settings: Optional[FDSettings] = None,
                 cache: Optional[ResultCache] = None):
        self.backend = backend
        self.executor = executor
        self.tracker = tracker or StateTracker()
        self.settings = settings or FDSettings()
        self.cache = cache if self.settings.use_cache else None

    def energy(self, mol: Molecule, label: str = "sp") -> Tuple[StateInfo, EnergyResult]:
        """Single point at ``mol`` with tracking applied (no gradient)."""
        res = self._run([Job(label, mol)])[0]
        match = self.tracker.advance(res)
        return match.state, res

    def _run(self, jobs: List[Job]) -> List[EnergyResult]:
        results: List[Optional[EnergyResult]] = [None] * len(jobs)
        todo = []
        for i, j in enumerate(jobs):
            hit = self.cache.get(j.mol) if self.cache else None
            if hit is not None:
                results[i] = hit
            else:
                todo.append(i)
        self._ncached = len(jobs) - len(todo)
        if todo:
            def store(k, res):
                if self.cache is not None:
                    self.cache.put(jobs[todo[k]].mol, res)
            fresh = self.executor.map(self.backend, [jobs[i] for i in todo], on_result=store)
            for i, r in zip(todo, fresh):
                results[i] = r
        return results  # type: ignore[return-value]

    def compute(self, mol: Molecule, prefix: str = "grad",
                expected_energy: Optional[float] = None) -> GradientResult:
        """Gradient of the tracked state at ``mol``.

        ``expected_energy`` (optional) is the predicted energy of the target
        state here; it guides energy/character-based root matching.
        """
        t0 = time.perf_counter()
        s = self.settings
        h = s.step
        plan = displacement_plan(mol, s)
        jobs = [Job(f"{prefix}/center", mol)]
        x0 = mol.coords
        for label, a, k, m in plan:
            x = x0.copy()
            x[a, k] += m * h
            jobs.append(Job(f"{prefix}/{label}", mol.with_coords(x)))

        log.info("%s: %d single points (%s, h=%.4g bohr, %d workers)", prefix, len(jobs),
                 s.scheme, h, self.executor.nworkers)
        results = self._run(jobs)
        center, disp = results[0], results[1:]

        # pick the target state at the centre, then match every displacement to it
        cmatch = self.tracker.advance(center, expected_energy)
        local = self.tracker.child(cmatch.state, center)
        matches, stats = local.select_many(disp)

        stencil = dict(STENCILS[s.scheme])
        grad = np.zeros_like(x0)
        if 0 in stencil:
            for a in range(mol.natoms):
                if a not in set(s.frozen_atoms):
                    grad[a] += stencil[0] * cmatch.state.energy
        for (label, a, k, m), mt in zip(plan, matches):
            grad[a, k] += stencil[m] * mt.state.energy
        grad /= h
        if s.project_rigid and not s.frozen_atoms:
            grad = project_rigid(grad, x0)

        unconverged = sum(m.state.extra.get("converged") is False
                          for m in [cmatch] + list(matches))
        if unconverged:
            log.warning("%s: tracked state not converged in %d single point(s); "
                        "increase td_max_cycle or nroots", prefix, unconverged)
        stats["unconverged"] = int(unconverged)
        if stats.get("swaps"):
            log.info("%s: root ordering changed in %d displacement(s); tracked by %s",
                     prefix, stats["swaps"], stats["method"])
        return GradientResult(energy=cmatch.state.energy, gradient=grad, state=cmatch.state,
                              center=center, njobs=len(jobs), ncached=self._ncached,
                              wall_time=time.perf_counter() - t0, tracking=stats)
