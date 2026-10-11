"""Root following for excited-state finite differences and optimizations.

Every displaced geometry returns a list of states, and near a crossing the
ordering of roots can change between displacements. If the gradient takes
"root 2" at every point blindly, it can difference two different states and
the result is garbage. The tracker picks, at each geometry, the state that
best matches a reference state:

* ``overlap``   -- |<ref|candidate>| from AO-basis signatures (transition
                   densities). Most reliable; needs backend support (PySCF).
* ``character`` -- multiplicity must match, then the smallest combination
                   of energy change and oscillator-strength change.
* ``energy``    -- closest energy with matching multiplicity.
* ``index``     -- always the same root number. No tracking; only for
                   debugging or when you know the ordering cannot change.
* ``auto``      -- overlap when signatures exist, otherwise character.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .results import EnergyResult, StateInfo

log = logging.getLogger(__name__)

METHODS = ("auto", "overlap", "character", "energy", "index")


@dataclass
class TrackingSettings:
    root: int = 0                       # 0 = ground state (or lowest of `multiplicity`)
    multiplicity: Optional[int] = None  # restrict the root count to one spin manifold
    method: str = "auto"
    min_overlap: float = 0.5            # warn below this overlap
    overlap_tie: float = 0.05           # overlaps closer than this -> break tie by energy
    energy_scale: float = 0.01          # Hartree; weight of dE in 'character' score
    osc_scale: float = 0.05             # weight of d(f) in 'character' score
    follow: bool = True                 # update the reference after each accepted step

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"tracking method must be one of {METHODS}, got {self.method!r}")


@dataclass
class Match:
    state: StateInfo
    position: int        # position in EnergyResult.states
    score: float         # overlap (higher better) or distance (lower better)
    method: str
    swapped: bool = False


def signature_overlap(a: np.ndarray, b: np.ndarray, s: Optional[np.ndarray]) -> float:
    """Normalised |<a|b>| for AO-basis matrices (or spin-stacked (2,n,n))."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return 0.0
    if a.ndim == 2:
        a, b = a[None], b[None]

    def dot(x, y):
        if s is None:
            return float(sum(np.vdot(xi, yi).real for xi, yi in zip(x, y)))
        return float(sum(np.einsum("ij,jk,lk,il->", xi, s, yi, s, optimize=True)
                         for xi, yi in zip(x, y)))

    na, nb = dot(a, a), dot(b, b)
    if na <= 0 or nb <= 0:
        return 0.0
    return abs(dot(a, b)) / np.sqrt(na * nb)


class StateTracker:
    def __init__(self, settings: Optional[TrackingSettings] = None):
        self.settings = settings or TrackingSettings()
        self.reference: Optional[StateInfo] = None
        self.reference_overlap_matrix: Optional[np.ndarray] = None
        self.history: List[dict] = []

    # -- helpers ---------------------------------------------------------
    def _candidates(self, result: EnergyResult) -> List[Tuple[int, StateInfo]]:
        mult = self.settings.multiplicity
        cands = list(enumerate(result.states))
        if mult is not None:
            filtered = [(i, s) for i, s in cands if s.multiplicity in (None, mult)]
            # the ground state of a closed-shell TDDFT run carries its own multiplicity
            cands = filtered or cands
        return cands

    def _method(self, cand_states) -> str:
        m = self.settings.method
        if m != "auto":
            return m
        ref = self.reference
        if ref is not None and ref.signature is not None and all(
                s.signature is not None for _, s in cand_states
                if not s.extra.get("scf_reference")):
            return "overlap"
        return "character"

    # -- public API ------------------------------------------------------
    def initialize(self, result: EnergyResult) -> StateInfo:
        """Choose the target state at the starting geometry."""
        cands = self._candidates(result)
        root = self.settings.root
        if root >= len(cands):
            raise ValueError(
                f"root {root} requested but backend returned only {len(cands)} matching "
                f"state(s); increase nroots")
        pos, state = cands[root]
        self.set_reference(state, result)
        log.info("tracking target: position %d (E=%.10f, mult=%s, f=%s)", pos,
                 state.energy, state.multiplicity, state.oscillator_strength)
        return state

    def set_reference(self, state: StateInfo, result: Optional[EnergyResult] = None):
        self.reference = state
        if result is not None and result.ao_overlap is not None:
            self.reference_overlap_matrix = result.ao_overlap

    def select(self, result: EnergyResult) -> Match:
        """Pick the state in ``result`` that corresponds to the reference."""
        if self.reference is None:
            st = self.initialize(result)
            return Match(st, result.states.index(st), 1.0, "init")
        ref = self.reference
        cands = self._candidates(result)
        if len(result.states) == 1:
            return Match(result.states[0], 0, 1.0, "single")
        method = self._method(cands)

        # Root numbers are NOT trusted: ordering changes along an optimization.
        # The one exception is a genuine SCF reference state (no excitation
        # signature, e.g. the ground state of a TDDFT run), which is always
        # listed first. Spin-flip / MRSF "ground" states carry signatures and
        # are tracked like any other root.
        if ref.signature is None and ref.extra.get("scf_reference"):
            for pos, s in cands:
                if s.extra.get("scf_reference"):
                    return Match(s, pos, 1.0, "reference")

        # an excited target must never collapse onto the SCF reference
        excited = [c for c in cands if not c[1].extra.get("scf_reference")]
        if excited and method != "index":
            cands = excited

        if method == "overlap" and ref.signature is None:
            method = "character"   # nothing to overlap with; fall back to character

        if method == "index":
            for pos, s in cands:
                if s.index == ref.index:
                    return Match(s, pos, 1.0, method)
            pos, s = cands[min(self.settings.root, len(cands) - 1)]
            return Match(s, pos, 1.0, method)

        if method == "overlap":
            smat = self.reference_overlap_matrix
            scored = []
            for pos, s in cands:
                if s.signature is None:
                    continue
                if (ref.multiplicity is not None and s.multiplicity is not None
                        and ref.multiplicity != s.multiplicity):
                    continue
                scored.append((signature_overlap(ref.signature, s.signature, smat), pos, s))
            if not scored:
                raise RuntimeError("overlap tracking requested but no state carries a signature")
            scored.sort(key=lambda t: -t[0])
            ov, pos, s = scored[0]
            # near-degenerate overlaps (strongly mixed states): prefer the closer energy
            if len(scored) > 1 and scored[0][0] - scored[1][0] < self.settings.overlap_tie:
                close = [t for t in scored if scored[0][0] - t[0] < self.settings.overlap_tie]
                ov, pos, s = min(close, key=lambda t: abs(t[2].energy - ref.energy))
            if ov < self.settings.min_overlap:
                log.warning("best state overlap %.3f below threshold %.2f (label %s)",
                            ov, self.settings.min_overlap, result.extra.get("label"))
            return Match(s, pos, ov, method, swapped=s.index != ref.index)

        # energy / character: smallest distance
        best = (np.inf, None, None)
        for pos, s in cands:
            d = abs(s.energy - ref.energy) / self.settings.energy_scale
            if method == "character":
                if (ref.multiplicity is not None and s.multiplicity is not None
                        and ref.multiplicity != s.multiplicity):
                    continue
                if ref.oscillator_strength is not None and s.oscillator_strength is not None:
                    d += abs(s.oscillator_strength - ref.oscillator_strength) / self.settings.osc_scale
            if d < best[0]:
                best = (d, pos, s)
        d, pos, s = best
        if s is None:
            raise RuntimeError("no candidate state with matching multiplicity")
        return Match(s, pos, float(d), method, swapped=s.index != ref.index)

    def child(self, state: StateInfo, result: EnergyResult) -> "StateTracker":
        """A tracker with the same settings, referenced to ``state``."""
        t = StateTracker(self.settings)
        t.reference = state
        t.reference_overlap_matrix = (result.ao_overlap if result.ao_overlap is not None
                                      else self.reference_overlap_matrix)
        return t

    def select_many(self, results: List[EnergyResult]) -> Tuple[List[Match], dict]:
        matches = [self.select(r) for r in results]
        stats = {
            "swaps": int(sum(m.swapped for m in matches)),
            "method": matches[0].method if matches else None,
        }
        if matches and matches[0].method == "overlap":
            stats["min_overlap"] = float(min(m.score for m in matches))
        return matches, stats

    def advance(self, result: EnergyResult) -> Match:
        """After an accepted step: match at the new geometry and update the reference."""
        old_index = self.reference.index if self.reference is not None else None
        m = self.select(result)
        if self.settings.follow:
            self.set_reference(m.state, result)
        self.history.append({"index": m.state.index, "energy": m.state.energy,
                             "score": m.score, "method": m.method})
        if m.swapped:
            log.info("root flip: target state is now root %d (was %s)",
                     m.state.index, old_index)
        return m

    # -- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict:
        ref = self.reference
        d = {"history": self.history}
        if ref is not None:
            d["reference"] = {"energy": ref.energy, "index": ref.index,
                              "multiplicity": ref.multiplicity,
                              "oscillator_strength": ref.oscillator_strength,
                              "label": ref.label,
                              "scf_reference": bool(ref.extra.get("scf_reference"))}
            d["arrays"] = {}
            if ref.signature is not None:
                d["arrays"]["signature"] = np.asarray(ref.signature)
            if self.reference_overlap_matrix is not None:
                d["arrays"]["ao_overlap"] = np.asarray(self.reference_overlap_matrix)
        return d

    def load_state_dict(self, d: dict) -> None:
        self.history = list(d.get("history", []))
        r = d.get("reference")
        if r:
            arrays = d.get("arrays", {})
            self.reference = StateInfo(energy=r["energy"], index=r["index"],
                                       multiplicity=r["multiplicity"],
                                       oscillator_strength=r["oscillator_strength"],
                                       label=r.get("label", ""),
                                       signature=arrays.get("signature"),
                                       extra={"scf_reference": r.get("scf_reference", False)})
            self.reference_overlap_matrix = arrays.get("ao_overlap")
