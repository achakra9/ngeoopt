"""Parser for Q-Chem output files.

Covers (written against Q-Chem 7.0 devel outputs):
  * SCF / DFT ground states
  * TDDFT, TDDFT/TDA, CIS excited states (the last "Excitation Energies" block,
    i.e. full TDDFT when both TDA and RPA are printed)
  * SF-TDDFT / SF-CIS and MRSF-TDDFT spin-flip states

Energies
  The SCF energy is taken from the converged SCF iteration line (10 decimals)
  rather than "Total energy =" (8 decimals). Excited-state total energies are
  only printed with 8 decimals, which limits finite-difference precision.

States
  * Spin-conserving TDDFT/CIS: [SCF reference (scf_reference=True), roots...]
  * Spin-flip / MRSF: only the SF states. The high-spin reference is not a
    state of the SF manifold (the "ground state" is usually SF state 1).
  Multiplicity comes from "Multiplicity:" when printed, else from <S**2>
  (None when the state is spin-contaminated).

Signatures for root following (phase-free, so independent of the arbitrary
signs Q-Chem gives orbitals in separate jobs):
  * "amp": squared CI amplitudes keyed by the excitation, e.g. D(14)->S(2) alpha
  * "dm" : hole / electron Mulliken populations per atom from the state and
           difference density matrices (only with state_analysis = true)
  Transition-density based quantities (TDM populations, Omega matrices, NTO
  decompositions) are deliberately not used.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..results import EnergyResult, StateInfo
from ..spin import multiplicity_from_s2  # noqa: F401  (re-exported)

_FLOAT = r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?"

_SCF_ITER = re.compile(rf"^\s*\d+\s+({_FLOAT})\s+\S+\s+Convergence criterion met", re.M)
_TOTAL_E = re.compile(rf"^\s*Total energy(?: in the final basis set)?\s*=\s*({_FLOAT})", re.M)
_BLOCK_HDR = re.compile(r"^\s*(.*Excitation Energies)\s*$", re.M)
_STATE_HDR = re.compile(rf"^\s*Excited state\s+(\d+):\s+excitation energy \(eV\)\s*=\s*({_FLOAT})", re.M)
_STATE_E = re.compile(rf"Total energy for state\s+(\d+):\s+({_FLOAT})")
_MULT = re.compile(r"Multiplicity:\s*(\w+)")
_S2 = re.compile(rf"<S\*\*2>\s*:\s*({_FLOAT})")
_STRENGTH = re.compile(rf"Strength\s*:\s*({_FLOAT})")
_TMOM = re.compile(rf"Trans\. Mom\.:\s*({_FLOAT})\s*X\s*({_FLOAT})\s*Y\s*({_FLOAT})\s*Z")
_AMP = re.compile(rf"([DSV])\(\s*(\d+)\)\s*-->\s*([DSV])\(\s*(\d+)\)\s+amplitude\s*=\s*({_FLOAT})\s*(alpha|beta)?")
_DAVIDSON = re.compile(r"^\s*\d+\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s*(Roots Converged)?\s*$", re.M)

_MULT_NAMES = {"singlet": 1, "doublet": 2, "triplet": 3, "quartet": 4, "quintet": 5}
_POST_HF = ("mp2", "ri-mp2", "ccsd", "ccsd(t)", "eom", "cc2", "qcisd", "adc", "omp2")


class QChemParseError(RuntimeError):
    pass


@dataclass
class QChemState:
    number: int
    excitation_ev: float
    energy: Optional[float] = None
    multiplicity: Optional[int] = None
    s2: Optional[float] = None
    strength: Optional[float] = None
    trans_mom: Optional[Tuple[float, float, float]] = None
    amplitudes: Dict[str, float] = field(default_factory=dict)


def electron_count(text: str) -> Optional[int]:
    m = re.search(r"There are\s+(\d+) alpha and\s+(\d+) beta electrons", text)
    return int(m.group(1)) + int(m.group(2)) if m else None


def last_job(text: str) -> str:
    """Q-Chem batch outputs contain several jobs; keep the last one."""
    parts = re.split(r"^\s*Welcome to Q-Chem", text, flags=re.M)
    return parts[-1] if len(parts) > 1 else text


def scf_energy(text: str) -> float:
    m = _SCF_ITER.findall(text)
    if m:
        return float(m[-1])
    if re.search(r"SCF failed to converge|Convergence failure", text, re.I):
        raise QChemParseError("SCF did not converge")
    m = _TOTAL_E.findall(text)
    if m:
        return float(m[-1])
    raise QChemParseError("no SCF energy found")


def excited_states(text: str) -> Tuple[Optional[str], List[QChemState]]:
    """States of the last 'Excitation Energies' block (header, states)."""
    headers = list(_BLOCK_HDR.finditer(text))
    if not headers:
        return None, []
    hdr = headers[-1]
    body = text[hdr.end():]
    first = _STATE_HDR.search(body)
    if first is None:
        return hdr.group(1).strip(), []
    # the block ends at the first dashed rule after the states begin
    stop = re.search(r"^\s*-{20,}\s*$", body[first.start():], re.M)
    if stop:
        body = body[:first.start() + stop.start()]
    starts = list(_STATE_HDR.finditer(body))
    states = []
    for i, m in enumerate(starts):
        chunk = body[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(body)]
        st = QChemState(number=int(m.group(1)), excitation_ev=float(m.group(2)))
        if (e := _STATE_E.search(chunk)):
            st.energy = float(e.group(2))
        if (mm := _MULT.search(chunk)):
            st.multiplicity = _MULT_NAMES.get(mm.group(1).lower())
        if (s2 := _S2.search(chunk)):
            st.s2 = float(s2.group(1))
        if (f := _STRENGTH.search(chunk)):
            st.strength = float(f.group(1))
        if (t := _TMOM.search(chunk)):
            st.trans_mom = tuple(float(v) for v in t.groups())
        for a in _AMP.finditer(chunk):
            key = f"{a.group(1)}{int(a.group(2))}>{a.group(3)}{int(a.group(4))}{(a.group(6) or '')[:1]}"
            st.amplitudes[key] = st.amplitudes.get(key, 0.0) + float(a.group(5)) ** 2
        states.append(st)
    return hdr.group(1).strip(), states


def davidson_unconverged(text: str) -> int:
    """Roots left unconverged in the last Davidson table (0 if all converged)."""
    rows = _DAVIDSON.findall(text)
    if not rows:
        return 0
    conv, left, flag = rows[-1]
    return 0 if flag else int(left)


def state_dm_populations(text: str) -> Dict[int, Dict[str, float]]:
    """Hole/electron Mulliken populations per atom from the difference density
    matrices printed by state_analysis, keyed by Q-Chem state number.

    Only headings of the form '... State <n> :' are used (the format seen in
    MRSF outputs); other layouts are ignored rather than guessed.
    """
    sec = re.search(r"Analysis of Unrelaxed Density Matrices(.*?)"
                    r"(?:Transition Density Matrix Analysis|SA-NTO Decomposition|={20,})",
                    text, re.S)
    if not sec:
        return {}
    body = sec.group(1)
    heads = list(re.finditer(r"^\s+\S.*?State\s+(\d+)\s*(?:\(.*?\))?\s*:\s*$", body, re.M))
    out: Dict[int, Dict[str, float]] = {}
    for i, h in enumerate(heads):
        chunk = body[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(body)]
        tab = re.search(r"Mulliken Population Analysis \(State/Difference DM\).*?\n\s*-{10,}\n"
                        r"(.*?)\n\s*-{10,}", chunk, re.S)
        if not tab:
            continue
        feats = {}
        for row in tab.group(1).strip().splitlines():
            p = row.split()
            if len(p) < 8:
                continue
            for name, v in zip(("ha", "hb", "ea", "eb"), p[4:8]):
                feats[f"{name}{int(p[0])}"] = abs(float(v))
        if feats:
            out[int(h.group(1))] = feats
    return out


def _normalized(d: Dict[str, float]) -> Dict[str, float]:
    n = math.sqrt(sum(v * v for v in d.values()))
    return {k: v / n for k, v in d.items()} if n > 0 else {}


def build_signature(amplitudes: Dict[str, float], dm: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Concatenate unit-normalised feature groups; the cosine of two such
    signatures is the average of the per-group similarities."""
    sig = {f"amp:{k}": v for k, v in _normalized(amplitudes).items()}
    if dm:
        sig.update({f"dm:{k}": v for k, v in _normalized(dm).items()})
    return sig


def parse_qchem_output(text: str, signatures: bool = True) -> EnergyResult:
    if "Thank you very much for using Q-Chem" not in text:
        raise QChemParseError("Q-Chem did not terminate normally")
    text = last_job(text)
    e_scf = scf_energy(text)
    header, states = excited_states(text)
    extra = {"scf_energy": e_scf, "qchem_block": header}
    if not states:
        ref = StateInfo(energy=e_scf, index=0, label="SCF", extra={"scf_reference": True})
        return EnergyResult(states=[ref], extra=extra)

    spin_flip = bool(re.search(r"\bSF-|MRSF|Spin.?Flip", header or "", re.I))
    left = davidson_unconverged(text)
    nelec = electron_count(text)
    dm = state_dm_populations(text) if signatures else {}

    result_states: List[StateInfo] = []
    if not spin_flip:
        result_states.append(StateInfo(energy=e_scf, label="S0",
                                       extra={"scf_reference": True, "converged": True}))
    by_energy = sorted(states, key=lambda s: s.excitation_ev)
    unconverged = {s.number for s in by_energy[len(by_energy) - left:]} if left else set()
    for st in states:
        energy = st.energy if st.energy is not None else e_scf + st.excitation_ev / 27.211386245988
        mult = st.multiplicity if st.multiplicity is not None else (
            multiplicity_from_s2(st.s2, nelec) if st.s2 is not None else None)
        sig = None
        if signatures and (st.amplitudes or st.number in dm):
            sig = build_signature(st.amplitudes, dm.get(st.number))
        result_states.append(StateInfo(
            energy=energy, multiplicity=mult, oscillator_strength=st.strength,
            label=f"{'SF' if spin_flip else 'ES'}{st.number}", signature=sig,
            extra={"qchem_state": st.number, "excitation_ev": st.excitation_ev, "s2": st.s2,
                   "converged": st.number not in unconverged,
                   "trans_mom": st.trans_mom}))
    result_states.sort(key=lambda s: (not s.extra.get("scf_reference", False), s.energy))
    for i, s in enumerate(result_states):
        s.index = i
    extra.update(spin_flip=spin_flip, unconverged_roots=left)
    return EnergyResult(states=result_states, converged=left == 0, extra=extra)


def check_correlated(rem: dict) -> None:
    method = str({k.lower(): v for k, v in rem.items()}.get("method", "")).lower()
    if any(method.startswith(p) for p in _POST_HF):
        raise NotImplementedError(
            f"built-in parsing for METHOD {method} is not implemented yet (no reference "
            f"outputs); set energy_regex / states_regex or a parser for now")
