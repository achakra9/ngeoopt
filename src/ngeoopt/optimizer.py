"""Quasi-Newton geometry optimizer driven by finite-difference gradients.

Cartesian coordinates, RFO or plain quasi-Newton steps inside a trust
radius, rigid-body motion projected out, model (Lindh) initial Hessian.
The full optimizer state is checkpointed after every cycle so a killed job
restarts where it stopped; together with the single-point cache, a restart
mid-gradient only redoes displacements that had not finished.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .findiff import FiniteDifferenceGradient, GradientResult
from .geometry.molecule import Molecule, rigid_body_modes
from .hessian import UPDATES, lindh_hessian

log = logging.getLogger(__name__)

# max force, rms force, max step, rms step (Hartree/Bohr, Bohr), energy (Hartree)
CONVERGENCE_PRESETS = {
    "loose":     dict(max_force=2.5e-3, rms_force=1.7e-3, max_step=1.0e-2, rms_step=6.7e-3, energy=1e-5),
    "gaussian":  dict(max_force=4.5e-4, rms_force=3.0e-4, max_step=1.8e-3, rms_step=1.2e-3, energy=1e-6),
    "qchem":     dict(max_force=3.0e-4, rms_force=3.0e-4, max_step=1.2e-3, rms_step=1.2e-3, energy=1e-6),
    "tight":     dict(max_force=1.5e-5, rms_force=1.0e-5, max_step=6.0e-5, rms_step=4.0e-5, energy=1e-8),
    "verytight": dict(max_force=2.0e-6, rms_force=1.0e-6, max_step=6.0e-6, rms_step=4.0e-6, energy=1e-9),
}


@dataclass
class ConvergenceCriteria:
    max_force: float = 4.5e-4
    rms_force: float = 3.0e-4
    max_step: float = 1.8e-3
    rms_step: float = 1.2e-3
    energy: float = 1e-6
    # 'force_and_any': forces AND (steps OR energy change)   [Gaussian/Q-Chem style]
    # 'all'          : every criterion
    # 'force'        : forces only
    mode: str = "force_and_any"

    @classmethod
    def preset(cls, name: str, **overrides) -> "ConvergenceCriteria":
        if name not in CONVERGENCE_PRESETS:
            raise ValueError(f"unknown convergence preset {name!r}; "
                             f"choose from {sorted(CONVERGENCE_PRESETS)}")
        return cls(**{**CONVERGENCE_PRESETS[name], **overrides})

    def check(self, grad, step, de) -> dict:
        g = np.asarray(grad).ravel()
        s = np.asarray(step).ravel() if step is not None else None
        flags = {
            "max_force": float(np.abs(g).max()) <= self.max_force,
            "rms_force": float(np.sqrt(np.mean(g ** 2))) <= self.rms_force,
            "max_step": s is not None and float(np.abs(s).max()) <= self.max_step,
            "rms_step": s is not None and float(np.sqrt(np.mean(s ** 2))) <= self.rms_step,
            "energy": de is not None and abs(de) <= self.energy,
        }
        forces = flags["max_force"] and flags["rms_force"]
        steps = flags["max_step"] and flags["rms_step"]
        if self.mode == "all":
            ok = forces and steps and flags["energy"]
        elif self.mode == "force":
            ok = forces
        else:
            ok = forces and (steps or flags["energy"])
        flags["converged"] = ok
        return flags


@dataclass
class OptimizerSettings:
    method: str = "rfo"              # rfo | qn | sd
    max_cycles: int = 100
    hessian_init: str = "lindh"      # lindh | unit
    hessian_unit: float = 0.5        # Hartree/Bohr^2 for 'unit'
    hessian_update: str = "bfgs"     # bfgs | sr1 | psb | bofill | none
    trust_radius: float = 0.3        # Bohr (norm of the full step)
    trust_min: float = 1e-3
    trust_max: float = 1.0
    reject_energy_rise: float = 1e-5  # Hartree; larger rises are rejected and retried
    convergence: ConvergenceCriteria = field(default_factory=ConvergenceCriteria)

    def __post_init__(self):
        if self.method not in ("rfo", "qn", "sd"):
            raise ValueError("optimizer method must be rfo | qn | sd")
        if self.hessian_update not in UPDATES:
            raise ValueError(f"hessian_update must be one of {sorted(UPDATES)}")
        if self.hessian_init not in ("lindh", "unit"):
            raise ValueError("hessian_init must be lindh | unit")


@dataclass
class CycleRecord:
    cycle: int
    energy: float
    max_force: float
    rms_force: float
    step_norm: float
    trust: float
    accepted: bool
    root: int
    njobs: int
    ncached: int
    wall_time: float
    converged: bool = False


@dataclass
class OptimizationResult:
    molecule: Molecule
    energy: float
    gradient: np.ndarray
    converged: bool
    cycles: int
    history: List[CycleRecord]


class Optimizer:
    def __init__(self, gradient: FiniteDifferenceGradient,
                 settings: Optional[OptimizerSettings] = None,
                 workdir: str | Path = ".", checkpoint: str = "ngeoopt_state.npz",
                 trajectory: Optional[str] = "trajectory.xyz",
                 frozen_atoms=()):
        self.fd = gradient
        self.settings = settings or OptimizerSettings()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.workdir / checkpoint
        self.trajectory_path = self.workdir / trajectory if trajectory else None
        self.frozen = sorted(set(frozen_atoms) | set(gradient.settings.frozen_atoms))

    # -- linear algebra ----------------------------------------------------
    def _active_basis(self, coords) -> np.ndarray:
        """Orthonormal basis (3N, m) of allowed displacements."""
        n3 = coords.size
        if self.frozen:
            mask = np.ones((coords.shape[0], 3), bool)
            mask[self.frozen] = False
            return np.eye(n3)[:, mask.ravel()]
        rigid = rigid_body_modes(coords)
        P = np.eye(n3) - rigid.T @ rigid
        w, v = np.linalg.eigh(P)
        return v[:, w > 0.5]

    def _step(self, coords, g, H, trust):
        U = self._active_basis(coords)
        gs = U.T @ g.ravel()
        Hs = U.T @ H @ U
        m = self.settings.method
        if m == "sd":
            dx = -gs
        elif m == "qn":
            w, v = np.linalg.eigh(Hs)
            w = np.where(np.abs(w) < 1e-4, 1e-4, np.abs(w))  # force a descent direction
            dx = -v @ ((v.T @ gs) / w)
        else:  # rational function optimization
            n = gs.size
            aug = np.zeros((n + 1, n + 1))
            aug[:n, :n] = Hs
            aug[:n, n] = aug[n, :n] = gs
            w, v = np.linalg.eigh(aug)
            vec = v[:, 0]
            if abs(vec[-1]) < 1e-12:
                dx = -gs
            else:
                dx = vec[:n] / vec[-1]
        norm = np.linalg.norm(dx)
        if norm > trust:
            dx *= trust / norm
        pred = float(gs @ dx + 0.5 * dx @ Hs @ dx)
        return (U @ dx).reshape(coords.shape), pred

    # -- checkpoint --------------------------------------------------------
    def _save(self, st: dict) -> None:
        arrays = {k: np.asarray(v) for k, v in st.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in st.items() if not isinstance(v, np.ndarray) and k != "tracker"}
        tr = st.get("tracker") or {}
        for k, v in (tr.get("arrays") or {}).items():
            arrays[f"tracker_{k}"] = v
        meta["tracker"] = {k: v for k, v in tr.items() if k != "arrays"}
        meta["history"] = [asdict(h) for h in st["history"]]
        tmp = self.checkpoint_path.with_suffix(".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(self.checkpoint_path)
        # human-readable sidecar
        self.checkpoint_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    def load_checkpoint(self) -> Optional[dict]:
        if not self.checkpoint_path.exists():
            return None
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            st = dict(meta)
            tracker_arrays = {}
            for k in data.files:
                if k == "meta":
                    continue
                if k.startswith("tracker_"):
                    tracker_arrays[k[len("tracker_"):]] = np.array(data[k])
                else:
                    st[k] = np.array(data[k])
        st["tracker"] = {**meta.get("tracker", {}), "arrays": tracker_arrays}
        st["history"] = [CycleRecord(**h) for h in meta.get("history", [])]
        return st

    def _write_frame(self, mol: Molecule, cycle: int, energy: float) -> None:
        if self.trajectory_path is None:
            return
        with open(self.trajectory_path, "a") as fh:
            fh.write(mol.to_xyz(comment=f"cycle {cycle} E= {energy:.10f}"))

    # -- main loop ---------------------------------------------------------
    def run(self, mol: Molecule, restart: bool = True) -> OptimizationResult:
        s = self.settings
        st = self.load_checkpoint() if restart else None
        if st is not None and st.get("symbols") != list(mol.symbols):
            log.warning("checkpoint %s belongs to a different molecule; ignoring it",
                        self.checkpoint_path)
            st = None
        if st is not None:
            log.info("restarting from %s at cycle %d", self.checkpoint_path, st["cycle"])
            self.fd.tracker.load_state_dict(st["tracker"])
            if st.get("scf_guess") is not None:
                self.fd.last_guess = st["scf_guess"]
            if st.get("converged"):
                log.info("checkpoint is already converged")
                best = mol.with_coords(st["x_best"])
                return OptimizationResult(best, st["e_best"], st["g_best"], True,
                                          st["cycle"], st["history"])
        else:
            if self.trajectory_path and self.trajectory_path.exists() and not restart:
                self.trajectory_path.unlink()
            if s.hessian_init == "lindh":
                H = lindh_hessian(mol.symbols, mol.coords)
                H += 1e-4 * np.eye(H.shape[0])
            else:
                H = s.hessian_unit * np.eye(3 * mol.natoms)
            st = {"symbols": list(mol.symbols), "cycle": 0, "x_next": mol.coords.copy(),
                  "H": H, "trust": s.trust_radius, "history": [], "converged": False,
                  "e_best": None, "pred": None}

        H = st["H"]
        trust = float(st["trust"])
        history: List[CycleRecord] = st["history"]
        x_best = st.get("x_best")
        g_best = st.get("g_best")
        e_best = st.get("e_best")
        pred = st.get("pred")
        x = np.array(st["x_next"])
        cycle = int(st["cycle"])
        gr: Optional[GradientResult] = None
        converged = False

        while cycle < s.max_cycles:
            cycle += 1
            t0 = time.perf_counter()
            cur = mol.with_coords(x)
            # where the tracked state should be according to the quadratic model;
            # also remember the tracker so a rejected step does not move it
            expected = None if e_best is None or pred is None else e_best + pred
            tracker_before = self.fd.tracker.state_dict()
            gr = self.fd.compute(cur, prefix=f"cycle{cycle:03d}", expected_energy=expected)
            g, e = gr.gradient, gr.energy
            step_taken = None if x_best is None else x - x_best
            accepted = True
            if x_best is not None:
                ds = (x - x_best).ravel()
                H = UPDATES[s.hessian_update](H, ds, (g - g_best).ravel())
                de = e - e_best
                rho = de / pred if pred and abs(pred) > 1e-14 else 1.0
                snorm = np.linalg.norm(ds)
                if de > s.reject_energy_rise and trust > s.trust_min:
                    accepted = False
                    self.fd.tracker.load_state_dict(tracker_before)
                    trust = max(s.trust_min, 0.25 * snorm)
                    log.info("cycle %d: energy rose by %.2e, step rejected; trust -> %.3g",
                             cycle, de, trust)
                elif rho > 0.75 and snorm > 0.8 * trust:
                    trust = min(s.trust_max, 2.0 * trust)
                elif rho < 0.25:
                    trust = max(s.trust_min, 0.5 * snorm)
            de_conv = None if e_best is None else e - e_best
            if accepted:
                x_best, g_best, e_best = x.copy(), g.copy(), e
                self._write_frame(cur, cycle, e)

            flags = s.convergence.check(g_best, step_taken if accepted else None, de_conv)
            converged = accepted and flags["converged"]
            rec = CycleRecord(cycle, e, gr.max_force, gr.rms_force,
                              0.0 if step_taken is None else float(np.linalg.norm(step_taken)),
                              trust, accepted, gr.state.index, gr.njobs, gr.ncached,
                              time.perf_counter() - t0, converged)
            history.append(rec)
            log.info("cycle %3d  E=%.10f  dE=%s  maxF=%.2e  rmsF=%.2e  step=%.2e  "
                     "trust=%.3f  root=%d  %s", cycle, e,
                     "   n/a   " if de_conv is None else f"{de_conv:+.2e}",
                     gr.max_force, gr.rms_force, rec.step_norm, trust, gr.state.index,
                     "CONVERGED" if converged else "")

            if not converged:
                dx, pred = self._step(x_best, g_best, H, trust)
                x = x_best + dx
            st.update(cycle=cycle, x_next=x, x_best=x_best, g_best=g_best, e_best=e_best,
                      H=H, trust=trust, pred=pred, history=history, converged=converged,
                      tracker=self.fd.tracker.state_dict())
            if self.fd.last_guess is not None:
                st["scf_guess"] = np.asarray(self.fd.last_guess)
            self._save(st)
            if converged:
                break

        if not converged:
            log.warning("optimization did not converge in %d cycles", s.max_cycles)
        final = mol.with_coords(x_best)
        return OptimizationResult(final, float(e_best), g_best, converged, cycle, history)
