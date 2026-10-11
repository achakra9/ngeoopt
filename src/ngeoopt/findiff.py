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
    # SCF of every displaced geometry starts from the centre-geometry density
    # (and each new centre from the previous one): keeps broken-symmetry /
    # open-shell solutions on one branch. Only for backends with supports_guess;
    # costs one extra sequential single point per gradient.
    guess_from_center: bool = True
    # second differences above this (Hartree/Bohr^2) mean the tracked energy
    # jumped inside the stencil (two different states were differenced)
    max_curvature: float = 5.0

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
        self.last_guess = None      # density of the latest centre geometry

    @property
    def _use_guess(self) -> bool:
        return self.settings.guess_from_center and getattr(self.backend, "supports_guess", False)

    def _remember_guess(self, res: EnergyResult):
        if self._use_guess and res.extra.get("guess") is not None:
            self.last_guess = res.extra["guess"]

    def energy(self, mol: Molecule, label: str = "sp") -> Tuple[StateInfo, EnergyResult]:
        """Single point at ``mol`` with tracking applied (no gradient)."""
        res = self._run([Job(label, mol, self.last_guess if self._use_guess else None)])[0]
        self._remember_guess(res)
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

    def _check_continuity(self, plan, matches, center: EnergyResult, target: StateInfo,
                          prefix: str) -> List[str]:
        """Flag coordinates whose +h/-h energies imply an unphysical curvature.

        Two causes are told apart:
          * the matched root changed inside the stencil -> two states were
            probably differenced (tracking problem)
          * same root on both sides -> genuine, very strong coupling to a nearby
            state (e.g. a pseudo-Jahn-Teller point); the surface is real but
            finite differences of this step size are unreliable there
        """
        h = self.settings.step
        e0 = target.energy
        e = {(a, k, m): mt for (_, a, k, m), mt in zip(plan, matches)}
        swapped, coupled = [], []
        for (a, k, m), mt in e.items():
            if m != 1 or (a, k, -1) not in e:
                continue
            mm = e[(a, k, -1)]
            curv = (mt.state.energy + mm.state.energy - 2.0 * e0) / h ** 2
            if abs(curv) <= self.settings.max_curvature:
                continue
            tag = f"atom {a} {_AXES[k]} ({curv:+.1f} Eh/bohr^2)"
            (swapped if (mt.swapped or mm.swapped) else coupled).append(tag)
        if swapped:
            log.warning("%s: tracked energy jumps along %d coordinate(s) where the matched "
                        "root changed: %s. Two different states were probably differenced; "
                        "check the root following or use a smaller step.", prefix,
                        len(swapped), ", ".join(swapped[:6]) + (" ..." if len(swapped) > 6 else ""))
        if coupled:
            others = [abs(s.energy - e0) for s in center.states
                      if s is not target and not s.extra.get("scf_reference")]
            gap = min(others) * 1000 if others else float("nan")
            log.warning("%s: very large curvature along %d coordinate(s) on the same state "
                        "(nearest state %.2f mEh away): strong coupling, e.g. a pseudo-Jahn-"
                        "Teller point; finite differences are unreliable here (%s)", prefix,
                        len(coupled), gap,
                        ", ".join(coupled[:4]) + (" ..." if len(coupled) > 4 else ""))
        return swapped + coupled

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
        x0 = mol.coords
        center_job = Job(f"{prefix}/center", mol, self.last_guess if self._use_guess else None)
        disp_jobs = []
        for label, a, k, m in plan:
            x = x0.copy()
            x[a, k] += m * h
            disp_jobs.append(Job(f"{prefix}/{label}", mol.with_coords(x)))

        log.info("%s: %d single points (%s, h=%.4g bohr, %d workers%s)", prefix,
                 len(disp_jobs) + 1, s.scheme, h, self.executor.nworkers,
                 ", SCF guess from centre" if self._use_guess else "")
        if self._use_guess:
            # centre first; its density seeds every displaced SCF
            center = self._run([center_job])[0]
            ncached = self._ncached
            self._remember_guess(center)
            for j in disp_jobs:
                j.guess = self.last_guess
            disp = self._run(disp_jobs)
            self._ncached += ncached
        else:
            results = self._run([center_job] + disp_jobs)
            center, disp = results[0], results[1:]
        jobs = [center_job] + disp_jobs

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
        stats["discontinuous"] = self._check_continuity(plan, matches, center, cmatch.state, prefix)
        if s.project_rigid and not s.frozen_atoms:
            grad = project_rigid(grad, x0)

        unconverged = sum(m.state.extra.get("converged") is False
                          for m in [cmatch] + list(matches))
        if unconverged:
            log.warning("%s: tracked state not converged in %d single point(s); "
                        "increase td_max_cycle or nroots", prefix, unconverged)
        stats["unconverged"] = int(unconverged)
        s2 = [m.state.extra.get("s2") for m in [cmatch] + list(matches)]
        s2 = [v for v in s2 if v is not None]
        if s2:
            stats["s2_range"] = (float(min(s2)), float(max(s2)))
            if max(s2) - min(s2) > 0.1:
                log.warning("%s: <S^2> of the tracked state varies %.3f-%.3f across the "
                            "stencil; the gradient mixes spin character (use a smaller step "
                            "or check the state)", prefix, min(s2), max(s2))
        if stats.get("swaps"):
            log.info("%s: root ordering changed in %d displacement(s); tracked by %s",
                     prefix, stats["swaps"], stats["method"])
        return GradientResult(energy=cmatch.state.energy, gradient=grad, state=cmatch.state,
                              center=center, njobs=len(jobs), ncached=self._ncached,
                              wall_time=time.perf_counter() - t0, tracking=stats)
