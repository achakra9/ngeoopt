"""PySCF energy backend (CPU, or NVIDIA GPU via gpu4pyscf).

Ground states: HF, any DFT functional PySCF knows, MP2, CCSD, CCSD(T),
(SA-)CASSCF / CASCI. Excited states: TDDFT/TDHF, TDA/CIS, EOM-EE-CCSD.

For TDDFT/TDA/CIS every state carries its AO transition density as an
overlap signature, so root following uses wavefunction overlaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..device import to_numpy
from ..geometry.molecule import Molecule
from ..results import EnergyResult, StateInfo
from .base import EnergyBackend, JobContext

log = logging.getLogger(__name__)

_WF_METHODS = {"hf", "mp2", "ccsd", "ccsd(t)", "casscf", "casci"}
_EXCITED = {None, "tddft", "tda", "cis", "tdhf", "eom-ccsd"}


@dataclass
class PySCFBackend(EnergyBackend):
    method: str = "hf"                 # hf | <xc> (b3lyp, pbe0, wb97x...) | mp2 | ccsd | ccsd(t) | casscf | casci
    basis: Any = "sto-3g"              # str or per-element dict
    reference: str = "auto"            # auto | rhf | uhf | rohf
    excited: Optional[str] = None      # None | tddft | tda | cis | tdhf | eom-ccsd
    nroots: int = 3
    spin_states: str = "singlet"       # singlet | triplet | both (restricted TD only)
    ecp: Any = None
    conv_tol: float = 1e-10            # tight: finite differences amplify SCF noise
    conv_tol_grad: Optional[float] = None   # default: 1e-7 for excited states
    max_cycle: int = 200
    grids_level: int = 4
    density_fit: Any = False           # False | True | auxbasis name
    td_conv_tol: float = 1e-8
    td_max_cycle: int = 200
    max_memory: int = 4000             # MB per worker
    ncas: Optional[int] = None         # CASSCF active orbitals
    nelecas: Any = None                # CASSCF active electrons (int or (na, nb))
    cas_nroots: int = 1                # state-averaged roots for CASSCF/CASCI
    solvent: Optional[Dict[str, Any]] = None   # {"method": "pcm"|"ddcosmo", "eps": 78.4}
    signatures: bool = True            # attach transition densities for overlap tracking
    use_gpu: str = "auto"              # auto | never | always
    scf_attrs: Dict[str, Any] = field(default_factory=dict)   # extra attributes set on mf
    verbose: int = 0

    name = "pyscf"
    supports_signatures = True

    def __post_init__(self):
        self.method = self.method.lower()
        if self.excited is not None:
            self.excited = self.excited.lower()
            if self.excited in ("none", ""):
                self.excited = None
        if self.excited not in _EXCITED:
            raise ValueError(f"excited must be one of {sorted(e for e in _EXCITED if e)}")
        if self.spin_states not in ("singlet", "triplet", "both"):
            raise ValueError("spin_states must be singlet | triplet | both")
        if self.method in ("casscf", "casci") and (self.ncas is None or self.nelecas is None):
            raise ValueError("casscf/casci need ncas and nelecas")

    def describe(self) -> str:
        ex = f"/{self.excited}({self.nroots})" if self.excited else ""
        return f"pyscf:{self.method}/{self.basis}{ex}"

    # -- building blocks ---------------------------------------------------
    def _mol(self, mol: Molecule, ctx: JobContext):
        from pyscf import gto
        m = gto.Mole()
        m.atom = [(s, tuple(c)) for s, c in zip(mol.symbols, mol.coords)]
        m.unit = "Bohr"
        m.basis = self.basis
        if self.ecp:
            m.ecp = self.ecp
        m.charge = mol.charge
        m.spin = mol.multiplicity - 1
        m.max_memory = self.max_memory
        m.verbose = self.verbose
        if ctx.keep_files and self.verbose:
            m.output = str(ctx.workdir / "pyscf.log")
        m.build()
        return m

    def _is_dft(self) -> bool:
        return self.method not in _WF_METHODS

    def _scf(self, m):
        from pyscf import dft, scf
        ref = self.reference
        if ref == "auto":
            ref = "rhf" if m.spin == 0 else "uhf"
        if self._is_dft():
            cls = {"rhf": dft.RKS, "uhf": dft.UKS, "rohf": dft.ROKS}[ref]
            mf = cls(m)
            mf.xc = self.method
            mf.grids.level = self.grids_level
        else:
            mf = {"rhf": scf.RHF, "uhf": scf.UHF, "rohf": scf.ROHF}[ref](m)
        if self.density_fit:
            aux = None if self.density_fit is True else self.density_fit
            mf = mf.density_fit(auxbasis=aux)
        if self.solvent:
            kind = self.solvent.get("method", "pcm").lower()
            mf = mf.PCM() if kind == "pcm" else mf.ddCOSMO()
            if "eps" in self.solvent:
                mf.with_solvent.eps = self.solvent["eps"]
        mf.conv_tol = self.conv_tol
        if self.conv_tol_grad:
            mf.conv_tol_grad = self.conv_tol_grad
        elif self.excited or self.method in ("casscf", "casci"):
            # excitation energies depend linearly on orbital errors, so the
            # default sqrt(conv_tol) leaves ~1e-8 Eh noise in every displacement
            mf.conv_tol_grad = 1e-7
        mf.max_cycle = self.max_cycle
        for k, v in self.scf_attrs.items():
            setattr(mf, k, v)
        return mf

    def _maybe_gpu(self, mf, ctx: JobContext):
        if self.use_gpu == "never" or not ctx.device.startswith("cuda"):
            if self.use_gpu == "always":
                raise RuntimeError(f"use_gpu=always but worker device is {ctx.device}")
            return mf, False
        try:
            import gpu4pyscf  # noqa: F401
            return mf.to_gpu(), True
        except Exception as exc:
            if self.use_gpu == "always":
                raise
            log.warning("gpu4pyscf unavailable (%s); running PySCF on CPU", exc)
            return mf, False

    # -- excited states ----------------------------------------------------
    def _td_states(self, mf, m, label: str = "") -> list:
        restricted = not isinstance(mf.mo_coeff, (tuple, list)) and np.ndim(to_numpy(mf.mo_coeff)) == 2
        kinds = ["singlet", "triplet"] if (self.spin_states == "both" and restricted) \
            else [self.spin_states]
        states = []
        for kind in kinds:
            if self.excited in ("tda", "cis"):
                td = mf.TDA()
            else:
                td = mf.TDDFT() if self._is_dft() else mf.TDHF()
            td.nstates = self.nroots
            td.conv_tol = self.td_conv_tol
            td.max_cycle = self.td_max_cycle
            if restricted:
                td.singlet = kind == "singlet"
            td.kernel()
            e_exc = np.atleast_1d(to_numpy(td.e))
            conv = np.atleast_1d(to_numpy(td.converged)).astype(bool)
            if conv.size != e_exc.size:
                conv = np.full(e_exc.size, bool(conv.all()))
            if not conv.all():
                log.debug("%s: %s root(s) %s not converged", 
                            label, kind, [int(i) + 1 for i in np.where(~conv)[0]])
            try:
                osc = np.atleast_1d(to_numpy(td.oscillator_strength()))
            except Exception:
                osc = [None] * len(e_exc)
            mult = (1 if kind == "singlet" else 3) if restricted else None
            for r, (de, xy) in enumerate(zip(e_exc, td.xy)):
                sig = self._tdm(mf, xy) if self.signatures else None
                f = None if osc[r] is None or kind == "triplet" else float(osc[r])
                states.append(StateInfo(energy=float(to_numpy(mf.e_tot) + de),
                                        multiplicity=mult, oscillator_strength=f,
                                        label=f"{kind[0].upper()}{r + 1}", signature=sig,
                                        extra={"excitation_energy": float(de),
                                               "converged": bool(conv[r])}))
        states.sort(key=lambda s: s.energy)
        return states

    @staticmethod
    def _tdm(mf, xy):
        """AO-basis (X+Y) transition density; (2, nao, nao) when unrestricted."""
        mo = to_numpy(mf.mo_coeff)
        occ = to_numpy(mf.mo_occ)
        x, y = xy
        if isinstance(x, tuple):  # unrestricted: ((xa, xb), (ya, yb))
            out = []
            for s in range(2):
                xs = to_numpy(x[s]) + (to_numpy(y[s]) if not np.isscalar(y) and y is not None
                                       and not np.isscalar(y[s]) else 0)
                c = mo[s]
                o = occ[s] > 0
                out.append(c[:, o] @ xs @ c[:, ~o].T)
            return np.array(out)
        x = to_numpy(x)
        t = x + (to_numpy(y) if not np.isscalar(y) else 0)
        o = occ > 0
        return mo[:, o] @ t @ mo[:, ~o].T

    # -- main entry --------------------------------------------------------
    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        from pyscf import lib
        lib.num_threads(ctx.nthreads)
        m = self._mol(mol, ctx)
        mf = self._scf(m)
        mf, on_gpu = self._maybe_gpu(mf, ctx)
        mf.kernel()
        converged = bool(mf.converged)
        if not converged:
            log.warning("SCF not converged for %s", ctx.label)
        e_scf = float(to_numpy(mf.e_tot))
        ref_state = StateInfo(energy=e_scf, multiplicity=mol.multiplicity, label="S0",
                              extra={"scf_reference": True})
        states = [ref_state]
        meth = self.method

        if meth in ("mp2", "ccsd", "ccsd(t)") or self.excited == "eom-ccsd":
            mf_cpu = mf.to_cpu() if on_gpu else mf
            if meth == "mp2":
                from pyscf import mp
                pt = mp.MP2(mf_cpu)
                pt.kernel()
                ref_state.energy = float(pt.e_tot)
            else:
                from pyscf import cc
                ccobj = cc.CCSD(mf_cpu)
                ccobj.conv_tol = min(1e-9, self.conv_tol * 10)
                ccobj.kernel()
                converged &= bool(ccobj.converged)
                e = float(ccobj.e_tot)
                if meth == "ccsd(t)":
                    e += float(ccobj.ccsd_t())
                ref_state.energy = e
                if self.excited == "eom-ccsd":
                    ee = np.atleast_1d(ccobj.eomee_ccsd_singlet(nroots=self.nroots)[0])
                    for r, de in enumerate(ee):
                        states.append(StateInfo(energy=float(ccobj.e_tot + de), multiplicity=1,
                                                label=f"EOM{r + 1}",
                                                extra={"excitation_energy": float(de)}))
        elif meth in ("casscf", "casci"):
            from pyscf import mcscf
            mf_cpu = mf.to_cpu() if on_gpu else mf
            mc = (mcscf.CASSCF if meth == "casscf" else mcscf.CASCI)(mf_cpu, self.ncas, self.nelecas)
            if self.cas_nroots > 1:
                if meth == "casscf":
                    mc = mc.state_average_([1.0 / self.cas_nroots] * self.cas_nroots)
                else:
                    mc.fcisolver.nroots = self.cas_nroots
            mc.conv_tol = max(self.conv_tol, 1e-10)
            mc.kernel()
            converged &= bool(getattr(mc, "converged", True))
            e_states = getattr(mc, "e_states", None) if self.cas_nroots > 1 else None
            e_states = np.atleast_1d(mc.e_tot if e_states is None else e_states)
            states = [StateInfo(energy=float(e), multiplicity=mol.multiplicity, label=f"CAS{r}")
                      for r, e in enumerate(e_states)]
        elif self.excited is not None:
            states += self._td_states(mf, m, ctx.label)

        for i, s in enumerate(states):
            s.index = i
        if self.signatures and any(s.signature is not None for s in states):
            states[0].ao_overlap = m.intor("int1e_ovlp")
        return EnergyResult(states=states, converged=converged,
                            extra={"gpu": on_gpu, "scf_energy": e_scf})
