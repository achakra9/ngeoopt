"""Root following for excited-state finite differences and optimizations.

Every displaced geometry returns a list of states, and the ordering of roots
changes along a coordinate (near crossings, and along whole optimization
paths). The tracker therefore never trusts the root number on its own: at each
geometry it scores every candidate against the reference state using all the
evidence the backend provides, and picks the lowest score:

    term                        cost (dimensionless, ~1 = "clearly different")
    wavefunction / amplitudes   (1 - |<ref|cand>|) / overlap_scale
    energy                      |E - E_expected| / energy_scale
    <S^2>                       |d<S^2>| / s2_scale
    oscillator strength         |df| / (osc_scale + 0.2 max f)
    transition dipole           || |mu| - |mu_ref| || / (tdip_scale + 0.2 max |mu|)
    root number                 index_weight * |d index|   (light tie-breaker)

Terms whose data a backend does not provide simply drop out. E_expected is the
optimizer's predicted energy of the target at the new geometry (or the
reference energy inside a gradient stencil). Transition dipoles are compared
component-wise in absolute value because their sign is arbitrary.

Hard guards on top of the score:
  * an excited target never collapses onto the SCF reference state
  * candidates whose <S^2> jumps by more than s2_max_change are excluded
  * candidates of a different (clean) multiplicity are excluded

``method`` selects the weights:
  auto / composite  all terms (default)
  overlap           overlap first, energy only to break near-ties
  character         energy, <S^2>, f, transition dipole (no wavefunction)
  energy            energy only (follows the adiabatic surface at true crossings)
  index             the root number only: no tracking, explicit opt-in
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from .results import EnergyResult, StateInfo

log = logging.getLogger(__name__)

METHODS = ("auto", "composite", "overlap", "character", "energy", "index")

#          overlap energy  s2   osc  tdip index
_WEIGHTS = {
    "composite": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "overlap":   (1.0, 0.02, 0.0, 0.0, 0.0, 0.0),
    "character": (0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    "energy":    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
}


@dataclass
class TrackingSettings:
    root: int = 0                       # 0 = ground state (or lowest of `multiplicity`)
    multiplicity: Optional[int] = None  # restrict the root count to one spin manifold
    method: str = "auto"
    min_overlap: float = 0.5            # warn below this overlap
    overlap_scale: float = 0.1          # 1 - overlap of 0.1 costs 1 unit
    energy_scale: float = 0.01          # Hartree, between optimization cycles
    stencil_energy_scale: float = 0.002  # Hartree, displaced geometry vs its centre
                                         # (a 0.005 Bohr step moves a state by micro-Hartrees)
    osc_scale: float = 0.05
    tdip_scale: float = 0.1             # a.u.
    s2_scale: float = 0.2
    index_weight: float = 0.05          # per position; only breaks ties
    s2_max_change: float = 0.5          # reject matches whose <S^2> jumps more than this
    s2_tolerance: float = 0.25          # |<S^2> - S(S+1)| above this = spin-contaminated
    follow: bool = True                 # update the reference after each accepted step

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"tracking method must be one of {METHODS}, got {self.method!r}")


@dataclass
class Match:
    state: StateInfo
    position: int        # position in EnergyResult.states
    score: float         # composite cost (lower is better); 1.0 for trivial matches
    method: str
    swapped: bool = False
    overlap: Optional[float] = None
    terms: dict = field(default_factory=dict)


def _s2(state: StateInfo) -> Optional[float]:
    v = state.extra.get("s2")
    return None if v is None else float(v)


def _tdip(state: StateInfo) -> Optional[np.ndarray]:
    v = state.extra.get("trans_mom")
    return None if v is None else np.abs(np.asarray(v, dtype=float))


def is_contaminated(state: StateInfo, tol: float) -> bool:
    """True when <S^2> is printed and is not close to S(S+1) for 2S+1 = multiplicity."""
    s2 = _s2(state)
    if s2 is None:
        return False
    if state.multiplicity is None:
        return True
    s = (state.multiplicity - 1) / 2.0
    return abs(s2 - s * (s + 1)) > tol


def signature_overlap(a, b, s: Optional[np.ndarray]) -> float:
    """Normalised |<a|b>|.

    * arrays: AO-basis matrices (or spin-stacked (2,n,n)), metric ``s``
    * dicts : sparse descriptors {feature: value} from text outputs (e.g. Q-Chem
      squared amplitudes, per-atom hole/electron populations); cosine over the
      union of features, missing features count as zero
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = sum(v * v for v in a.values())
        nb = sum(v * v for v in b.values())
        return abs(dot) / np.sqrt(na * nb) if na > 0 and nb > 0 else 0.0
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

    # -- public API ------------------------------------------------------
    def initialize(self, result: EnergyResult) -> StateInfo:
        """Choose the target state at the starting geometry."""
        mult = self.settings.multiplicity
        if mult is None:
            cands = list(enumerate(result.states))
        else:
            # count only states that are clean members of the requested manifold;
            # spin-contaminated SF roots (multiplicity None) are not "the n-th singlet"
            cands = [(i, st) for i, st in enumerate(result.states) if st.multiplicity == mult
                     and not is_contaminated(st, self.settings.s2_tolerance)]
        root = self.settings.root
        if root >= len(cands):
            raise ValueError(
                f"root {root} requested but backend returned only {len(cands)} matching "
                f"state(s){' with clean multiplicity ' + str(mult) if mult else ''}; "
                f"increase nroots")
        pos, state = cands[root]
        self.set_reference(state, result)
        log.info("tracking target: position %d (E=%.10f, mult=%s, <S^2>=%s, f=%s)", pos,
                 state.energy, state.multiplicity, _s2(state), state.oscillator_strength)
        return state

    def set_reference(self, state: StateInfo, result: Optional[EnergyResult] = None):
        self.reference = state
        if result is not None and result.ao_overlap is not None:
            self.reference_overlap_matrix = result.ao_overlap

    def select(self, result: EnergyResult, expected_energy: Optional[float] = None) -> Match:
        """Pick the state in ``result`` that corresponds to the reference.

        ``expected_energy`` is where the target state should be at this geometry
        (e.g. the optimizer's quadratic prediction). Energy-based matching then
        compares against it instead of the reference geometry's energy, which
        matters after real optimization steps where the state's own energy
        changes by more than the gap to its neighbours.
        """
        if self.reference is None:
            st = self.initialize(result)
            return Match(st, result.states.index(st), 1.0, "init")
        ref = self.reference
        st_ = self.settings
        e_ref = ref.energy if expected_energy is None else expected_energy
        cands = self._candidates(result)
        if len(result.states) == 1:
            return Match(result.states[0], 0, 1.0, "single")
        method = "composite" if st_.method == "auto" else st_.method

        # the SCF reference of a spin-conserving run is identified by its flag,
        # never by position; spin-flip "ground" states are ordinary roots
        if ref.signature is None and ref.extra.get("scf_reference"):
            for pos, s in cands:
                if s.extra.get("scf_reference"):
                    return Match(s, pos, 1.0, "reference")

        if method == "index":
            for pos, s in cands:
                if s.index == ref.index:
                    return Match(s, pos, 1.0, method)
            pos, s = cands[min(st_.root, len(cands) - 1)]
            return Match(s, pos, 1.0, method)

        # hard guards ------------------------------------------------------
        excited = [c for c in cands if not c[1].extra.get("scf_reference")]
        cands = excited or cands
        if ref.multiplicity is not None:
            same = [c for c in cands if c[1].multiplicity in (None, ref.multiplicity)]
            cands = same or cands
        s2_ref = _s2(ref)
        if s2_ref is not None:
            steady = [c for c in cands if _s2(c[1]) is None
                      or abs(_s2(c[1]) - s2_ref) <= st_.s2_max_change]
            if steady:
                cands = steady
            else:
                log.warning("every candidate changes <S^2> by more than %.2f (label %s)",
                            st_.s2_max_change, result.extra.get("label"))

        # composite score --------------------------------------------------
        w_ov, w_e, w_s2, w_f, w_mu, w_idx = _WEIGHTS[method]
        smat = self.reference_overlap_matrix
        mu_ref = _tdip(ref)
        best = None
        for pos, s in cands:
            terms = {}
            ov = None
            if w_ov and ref.signature is not None and s.signature is not None:
                ov = signature_overlap(ref.signature, s.signature, smat)
                terms["overlap"] = w_ov * (1.0 - ov) / st_.overlap_scale
            if w_e:
                terms["energy"] = w_e * abs(s.energy - e_ref) / st_.energy_scale
            if w_s2 and s2_ref is not None and _s2(s) is not None:
                terms["s2"] = w_s2 * abs(_s2(s) - s2_ref) / st_.s2_scale
            if w_f and ref.oscillator_strength is not None and s.oscillator_strength is not None:
                f0, f1 = ref.oscillator_strength, s.oscillator_strength
                terms["osc"] = w_f * abs(f1 - f0) / (st_.osc_scale + 0.2 * max(f0, f1))
            mu = _tdip(s)
            if w_mu and mu_ref is not None and mu is not None:
                big = max(np.linalg.norm(mu_ref), np.linalg.norm(mu))
                terms["tdip"] = w_mu * float(np.linalg.norm(mu - mu_ref)) / (st_.tdip_scale + 0.2 * big)
            if w_idx:
                terms["index"] = w_idx * st_.index_weight * abs(s.index - ref.index)
            cost = float(sum(terms.values()))
            if best is None or cost < best[0]:
                best = (cost, pos, s, ov, terms)
        if best is None:
            raise RuntimeError("no candidate state left after the multiplicity / <S^2> guards")
        cost, pos, s, ov, terms = best
        if ov is not None and ov < st_.min_overlap:
            log.warning("best state overlap %.3f below threshold %.2f (label %s)",
                        ov, st_.min_overlap, result.extra.get("label"))
        log.debug("matched %s: cost %.3f %s", s.label, cost,
                  {k: round(v, 3) for k, v in terms.items()})
        return Match(s, pos, cost, method, swapped=s.index != ref.index, overlap=ov, terms=terms)

    def child(self, state: StateInfo, result: EnergyResult) -> "StateTracker":
        """A tracker with the same settings, referenced to ``state``."""
        # inside a finite-difference stencil energies barely move: weigh them harder
        t = StateTracker(replace(self.settings, energy_scale=self.settings.stencil_energy_scale))
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
        ovs = [m.overlap for m in matches if m.overlap is not None]
        if ovs:
            stats["min_overlap"] = float(min(ovs))
        return matches, stats

    def advance(self, result: EnergyResult, expected_energy: Optional[float] = None) -> Match:
        """Match at a new geometry and move the reference there (if ``follow``)."""
        old_index = self.reference.index if self.reference is not None else None
        m = self.select(result, expected_energy)
        if self.settings.follow:
            self.set_reference(m.state, result)
        self.history.append({"index": m.state.index, "energy": m.state.energy,
                             "score": m.score, "method": m.method, "s2": _s2(m.state),
                             "overlap": m.overlap})
        if is_contaminated(m.state, self.settings.s2_tolerance):
            log.warning("tracked state is spin-contaminated: <S^2> = %.3f (mult %s)",
                        _s2(m.state), m.state.multiplicity)
        if m.swapped:
            log.info("root flip: target state is now root %d (was %s)",
                     m.state.index, old_index)
        return m

    # -- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict:
        ref = self.reference
        d = {"history": list(self.history)}
        if ref is not None:
            d["reference"] = {"energy": ref.energy, "index": ref.index,
                              "multiplicity": ref.multiplicity,
                              "oscillator_strength": ref.oscillator_strength,
                              "label": ref.label,
                              "scf_reference": bool(ref.extra.get("scf_reference")),
                              "s2": _s2(ref),
                              "trans_mom": None if _tdip(ref) is None else [float(v) for v in _tdip(ref)]}
            d["arrays"] = {}
            if isinstance(ref.signature, dict):
                d["reference"]["signature_dict"] = {k: float(v) for k, v in ref.signature.items()}
            elif ref.signature is not None:
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
                                       signature=r.get("signature_dict") or arrays.get("signature"),
                                       extra={"scf_reference": r.get("scf_reference", False),
                                              "s2": r.get("s2"), "trans_mom": r.get("trans_mom")})
            self.reference_overlap_matrix = arrays.get("ao_overlap")
