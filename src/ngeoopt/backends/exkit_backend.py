"""exkit backend: any exkit method as an energy source.

exkit (excited states and correlation on PySCF / Q-Chem references) has no
analytic gradients for most of its methods -- SF-CIS/TDA, SF-CISd, MRSF,
MD-SF, pp/hh-TDA, DEA/DIP, IP/EA, RI-MP2, double hybrids ... -- so ngeoopt
drives it by finite differences with its own root following.

Each single point mirrors exkit's own numerical-geometry energy evaluation
(exkit.geomopt.EnergyCalculator):
  * the molecule and SCF are built with exkit.driver.build_mol / build_scf
    (no geometry symmetrization, which would undo a small displacement)
  * the SCF starts from the centre-geometry density (ctx.guess) so broken
    symmetry / open-shell solutions stay on one branch
  * numpy's global RNG is reseeded before every job (exkit's CASSCF initial guess
    is random; otherwise displaced jobs can converge to different SA-CASSCF solutions)
  * the Davidson tolerance is tightened to <= 1e-7 and max_cycle >= 200, and the
    SCF orbital-gradient tolerance to 1e-7 (unless the input sets scf_conv_tol_grad)

States returned:
  * spin-conserving methods (reference in the state manifold): the SCF
    reference flagged scf_reference, then the roots
  * spin-flip and other methods: the roots only (exkit state 1 = first root)
  * ground-state methods (MP2, double hybrids, ...): one state
Every root carries <S^2> (multiplicity None when spin-contaminated); for
spin-conserving methods also the transition dipole and oscillator strength; and, where
the method exposes excitation blocks, an AO-basis excitation density per
block, sum Co X Cv^T. These are invariant to orbital sign flips between
separate SCF runs, so overlap-based root following works across geometries.
Methods with doubles (SF-CISd) get no signature: their near-degenerate states can
share almost the same singles, so a singles-only overlap is misleading; they are
tracked by <S^2> and the predicted energy instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..geometry.molecule import Molecule
from ..results import EnergyResult, StateInfo
from ..spin import multiplicity_from_s2
from .base import EnergyBackend, JobContext

log = logging.getLogger(__name__)

# exkit options that ngeoopt always controls
_FORCED = dict(verbose=0, scf_verbose=0, analysis=False, banner=False, cubes=False,
               resource_report=False, heartbeat=0.0, optimize="none", frequency="none",
               symmetry="c1", tpa=False, td_dynamics=False, exciton=False)


def _read_exkit_input(path: str) -> Dict[str, Any]:
    from exkit.inputs import _read_text
    raw = _read_text(Path(path).expanduser().read_text())
    raw.pop("geometry", None)
    return raw


@dataclass
class ExkitBackend(EnergyBackend):
    input: Dict[str, Any] = field(default_factory=dict)   # exkit keywords (method, basis, nstates, ...)
    input_file: Optional[str] = None     # an exkit .inp file; its geometry is replaced
    signatures: bool = True              # AO excitation densities for overlap tracking
    davidson_conv_tol: float = 1e-7      # upper bound on exkit conv_tol
    davidson_max_cycle: int = 200        # lower bound on exkit max_cycle
    scf_conv_tol_grad: float = 1e-7      # used unless the exkit input sets scf_conv_tol_grad
    require_scf_converged: bool = True   # fail the job (and retry) instead of using it
    seed: Optional[int] = 0              # reseed numpy's global RNG before every job
    cas_conv_tol: float = 1e-10          # CAS methods, unless the input sets cas_conv_tol

    name = "exkit"
    supports_signatures = True
    supports_guess = True

    def __post_init__(self):
        try:
            import exkit  # noqa: F401
        except ImportError as exc:
            raise ImportError("backend 'exkit' needs exkit installed in this environment") from exc
        if self.input_file:
            self.input_file = str(Path(self.input_file).expanduser().resolve())
        raw = self._raw()
        for k in ("optimize", "frequency"):
            if str(raw.get(k, "none")).lower() not in ("none", "no", "false", "0"):
                log.warning("exkit %s = %s ignored: ngeoopt does the optimization", k, raw[k])
        integrals = str(raw.get("integrals", "pyscf")).lower()
        if integrals not in ("pyscf", "qchem"):
            raise ValueError("exkit backend needs integrals = pyscf or qchem")
        # validate keywords once, in the parent, so typos fail before any job starts
        self._options(Molecule(["H", "H"], np.array([[0, 0, 0], [0, 0, 1.4]])),
                      JobContext(Path(".")), validate_only=True)

    def describe(self) -> str:
        raw = self._raw()
        return f"exkit:{raw.get('method', 'cis')}/{raw.get('basis', 'sto-3g')}"

    # -- options -------------------------------------------------------------
    def _raw(self) -> Dict[str, Any]:
        raw = _read_exkit_input(self.input_file) if self.input_file else {}
        raw.update({k.lower(): v for k, v in self.input.items()})
        return raw

    def _options(self, mol: Molecule, ctx: JobContext, validate_only: bool = False) -> dict:
        from exkit.inputs import parse_input
        raw = self._raw()
        for key, val in (("charge", mol.charge), ("spin", mol.multiplicity - 1)):
            if key in raw and int(raw[key]) != val and not validate_only:
                log.warning("exkit %s = %s overridden by the molecule (%s)", key, raw[key], val)
            raw[key] = val
        raw["geometry"] = "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}"
                                    for s, (x, y, z) in zip(mol.symbols, mol.coords))
        raw["units"] = "bohr"
        raw.update(_FORCED)
        raw["outdir"] = str(ctx.workdir)
        o = parse_input(raw)
        if not raw.get("scf_conv_tol_grad"):
            # excitation energies depend linearly on orbital errors; PySCF's default
            # sqrt(scf_conv_tol) leaves ~1e-8 Eh noise in every displaced root
            o["scf_conv_tol_grad"] = self.scf_conv_tol_grad
        if "cas_conv_tol" in o and "cas_conv_tol" not in raw:
            # SA-CASSCF energies at exkit's default 1e-8 scatter by ~2e-7 Eh between
            # runs: ~2e-5 Eh/Bohr of gradient noise at h = 0.005
            o["cas_conv_tol"] = self.cas_conv_tol
        o["conv_tol"] = min(o["conv_tol"], self.davidson_conv_tol)
        o["max_cycle"] = max(o["max_cycle"], self.davidson_max_cycle)
        if ctx.device.startswith("cuda") and "device" not in raw:
            o["device"] = "cuda"
        if ctx.nthreads:
            o["threads"] = ctx.nthreads
        return o

    # -- signatures ----------------------------------------------------------
    @staticmethod
    def _excitation_densities(method, X) -> Optional[list]:
        """Per state: (nblocks, nao, nao) AO excitation densities, or None."""
        if not all(hasattr(method, a) for a in ("blocks", "Co", "Cv", "unpack")):
            return None
        nS = getattr(method, "nS", None)
        if nS is not None and np.asarray(X).shape[0] > nS:
            # SF-CISd and similar: states that differ mainly in their doubles have
            # near-identical singles, so a singles-only signature misidentifies
            # them (seen for twisted ethylene). No signature -> <S^2>/energy tracking.
            return None
        try:
            amps = method.unpack(np.asarray(X))
            out = []
            for k in range(np.asarray(X).shape[1]):
                blocks = []
                for n, b in enumerate(method.blocks):
                    c = np.asarray(amps[n][k])
                    w = 0.5 if getattr(b, "tri", False) else 1.0
                    blocks.append(w * (np.asarray(method.Co[n]) @ c @ np.asarray(method.Cv[n]).T))
                out.append(np.array(blocks))
            return out
        except Exception as exc:  # method-specific layouts: fall back to <S^2>/energy
            log.debug("no exkit excitation densities for %s: %s", type(method).__name__, exc)
            return None

    # -- main entry ----------------------------------------------------------
    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        from exkit import driver
        from exkit.backend import get_backend
        from exkit.reference import Reference
        from exkit.registry import get_method
        from exkit.twoel import want_gpu4pyscf

        o = self._options(mol, ctx)
        if self.seed is not None:
            # exkit's CASSCF path draws from numpy's global RNG (initial CI guess);
            # without a fixed state a worker that already ran other jobs can land on
            # a different SA-CASSCF solution (seen with exkit.run called repeatedly)
            np.random.seed(self.seed)
        entry = get_method(o["method"])
        cls = entry["cls"]
        if not getattr(cls, "supports_geomopt", True):
            raise NotImplementedError(f"exkit method {o['method']} does not support geometry "
                                      f"derivatives (supports_geomopt = False)")
        if getattr(cls, "make_reference", None) is not None:
            raise NotImplementedError("periodic (gamma-*/k-*) exkit methods are not supported")
        xc = driver.resolve_xc(o, entry)
        backend = get_backend(o["backend"], o["device"])
        gpu = want_gpu4pyscf(backend, o["gpu4pyscf"])

        dm0 = ctx.guess
        if o["integrals"] == "pyscf":
            emol = driver.build_mol(o, cls)
            if dm0 is not None and np.shape(dm0)[-1] != emol.nao_nr():
                dm0 = None
            mf = driver.build_scf(emol, o, xc, gpu=gpu, dm0=None if gpu else dm0, cls=cls)
            ref = Reference(mf)
        else:  # qchem integrals: exkit writes and runs Q-Chem in this job's directory
            o["qchem_output"], o["qchem_check"] = None, False
            ref = driver.build_reference(o, entry, xc, gpu)
            mf = ref.mf
        scf_ok = bool(getattr(mf, "converged", True))
        if not scf_ok and self.require_scf_converged:
            raise RuntimeError(f"exkit SCF did not converge for {ctx.label} (use exkit scf_* "
                               f"controls, e.g. scf_algorithm = soscf, scf_level_shift)")
        guess = np.asarray(mf.make_rdm1())
        e_ref = float(ref.e_tot)
        nelec = int(ref.mol.nelectron)
        extra = {"scf_energy": e_ref, "guess": guess, "exkit_method": o["method"]}

        common = dict(backend=backend, jk=o["jk"], auxbasis=o["auxbasis"],
                      gpu4pyscf="yes" if gpu else "no", aux_chunk=o["aux_chunk"],
                      nfrz_occ=o["nfrz_occ"], nfrz_virt=o["nfrz_virt"],
                      max_memory=o["max_memory"])
        opts = {k: o[k] for k in getattr(cls, "extra_options", {})}
        if getattr(cls, "is_ground_state", False):
            m = cls(ref, **common, **opts)
            e = float(m.kernel(verbose=0).e_tot)
            return EnergyResult(states=[StateInfo(e, 0, mol.multiplicity, label="GS")],
                                converged=scf_ok, extra=extra)

        m = cls(ref, cvs_orbitals=o["cvs_orbitals"], **common, **opts)
        res = m.kernel(nroots=o["nstates"], solver=o["solver"], conv_tol=o["conv_tol"],
                       max_cycle=o["max_cycle"], verbose=0)
        e_tot = np.asarray(res.e_tot, dtype=float)
        pt = getattr(res, "pt", None)
        if pt is not None:                 # IP/EA with ipea_pt: the corrected surface
            e_tot = np.asarray(pt["e_tot"], dtype=float)
            extra["pt"] = pt.get("kind")
        s2 = np.asarray(res.s2, dtype=float) if res.s2 is not None else np.full(e_tot.size, np.nan)
        dens = self._excitation_densities(m, res.X) if self.signatures else None
        conv = bool(getattr(res, "converged", True))
        omega = np.asarray(res.omega, dtype=float)
        mus = [None] * e_tot.size
        if dens is not None and cls.ground_in_space:
            # transition dipoles <0|r|K> from the excitation densities (spin-conserving
            # methods only; checked against PySCF CIS: same mu and f = 2/3 w |mu|^2)
            r_ao = ref.mol.intor("int1e_r")
            mus = [sum(np.einsum("xij,ij->x", r_ao, Tb) for Tb in T) for T in dens]

        states = []
        if cls.ground_in_space:
            ref_s2 = float(ref.spin_square())
            states.append(StateInfo(e_ref, multiplicity=mol.multiplicity, label="S0",
                                    extra={"scf_reference": True, "s2": ref_s2,
                                           "converged": True, "exkit_state": 0}))
        for k in range(e_tot.size):
            sk = None if not np.isfinite(s2[k]) else float(s2[k])
            mu = mus[k]
            states.append(StateInfo(
                float(e_tot[k]),
                multiplicity=None if sk is None else multiplicity_from_s2(sk, nelec),
                oscillator_strength=None if mu is None else float(2.0 / 3.0 * omega[k] * mu @ mu),
                label=f"R{k + 1}", signature=None if dens is None else dens[k],
                extra={"s2": sk, "converged": conv, "exkit_state": k + 1,
                       "excitation_energy": float(omega[k]),
                       "trans_mom": None if mu is None else tuple(float(v) for v in mu)}))
        states.sort(key=lambda s: (not s.extra.get("scf_reference", False), s.energy))
        for i, s in enumerate(states):
            s.index = i
        if dens is not None:
            states[0].ao_overlap = np.asarray(ref.S)
        return EnergyResult(states=states, converged=conv and scf_ok, extra=extra)
