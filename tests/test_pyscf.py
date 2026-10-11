"""PySCF backend checked against PySCF's own analytic gradients."""

import numpy as np
import pytest

from conftest import HAS_PYSCF

pytestmark = [pytest.mark.pyscf,
              pytest.mark.skipif(not HAS_PYSCF, reason="pyscf not installed")]

from ngeoopt import (FDSettings, FiniteDifferenceGradient, Molecule, OptimizerSettings,  # noqa: E402
                     StateTracker, TrackingSettings, optimize)
from ngeoopt.parallel import make_executor  # noqa: E402

FORMALDEHYDE = "C 0 0 -0.53\nO 0 0 0.68\nH 0 0.94 -1.12\nH 0 -0.94 -1.12"


def pyscf_mol(mol, basis):
    from pyscf import gto
    return gto.M(atom=[(s, tuple(c)) for s, c in zip(mol.symbols, mol.coords)],
                 unit="Bohr", basis=basis, charge=mol.charge, spin=mol.multiplicity - 1,
                 verbose=0)


def fd(mol, backend, tmp_path, state=None, scheme="central4", workers=4):
    from ngeoopt.backends.pyscf_backend import PySCFBackend  # noqa: F401
    with make_executor("process", nworkers=workers, device="cpu",
                       scratch=tmp_path) as ex:
        return FiniteDifferenceGradient(backend, ex, StateTracker(state),
                                        FDSettings(scheme=scheme)).compute(mol)


def test_rhf_gradient(tmp_path, water):
    from pyscf import scf
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    gr = fd(water, PySCFBackend(method="hf", basis="sto-3g"), tmp_path)
    mf = scf.RHF(pyscf_mol(water, "sto-3g"))
    mf.conv_tol = 1e-11
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    assert gr.energy == pytest.approx(mf.e_tot, abs=1e-9)
    assert np.abs(gr.gradient - g).max() < 2e-6


def test_uks_doublet_gradient(tmp_path):
    from pyscf import dft
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    oh = Molecule.from_lines("O 0 0 0\nH 0 0 0.99", multiplicity=2)
    gr = fd(oh, PySCFBackend(method="pbe", basis="6-31g", grids_level=3), tmp_path)
    mf = dft.UKS(pyscf_mol(oh, "6-31g"))
    mf.xc = "pbe"
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    assert np.abs(gr.gradient - g).max() < 2e-5


def test_mp2_energy_matches(tmp_path, water):
    from pyscf import mp, scf
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    from ngeoopt.backends.base import JobContext
    res = PySCFBackend(method="mp2", basis="sto-3g").compute(water, JobContext(tmp_path))
    mf = scf.RHF(pyscf_mol(water, "sto-3g")).run(conv_tol=1e-11)
    assert res.energy == pytest.approx(mp.MP2(mf).run().e_tot, abs=1e-9)


def test_tda_s1_gradient_with_overlap_tracking(tmp_path):
    from pyscf import scf, tdscf
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    mol = Molecule.from_lines(FORMALDEHYDE)
    be = PySCFBackend(method="hf", basis="6-31g", excited="cis", nroots=4)
    gr = fd(mol, be, tmp_path, TrackingSettings(root=1, multiplicity=1), scheme="central",
            workers=6)
    assert gr.tracking["method"] == "overlap"
    assert gr.tracking["min_overlap"] > 0.99
    mf = scf.RHF(pyscf_mol(mol, "6-31g")).run(conv_tol=1e-12, conv_tol_grad=1e-8)
    td = tdscf.TDA(mf)
    td.nstates = 4
    td.conv_tol = 1e-10
    td.kernel()
    g = td.nuc_grad_method().kernel(state=1)
    assert gr.energy == pytest.approx(mf.e_tot + td.e[0], abs=1e-8)
    assert np.abs(gr.gradient - g).max() < 5e-5


def test_triplet_states_have_multiplicity(tmp_path, water):
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    from ngeoopt.backends.base import JobContext
    be = PySCFBackend(method="hf", basis="sto-3g", excited="tda", nroots=2, spin_states="both")
    res = be.compute(water, JobContext(tmp_path))
    mults = [s.multiplicity for s in res.states[1:]]
    assert sorted(mults) == [1, 1, 3, 3]
    assert res.states[0].extra["scf_reference"]
    assert res.ao_overlap is not None
    assert np.all(np.diff(res.energies[1:]) >= 0)   # sorted by energy


def test_optimize_water_hf(tmp_path):
    from pyscf import scf
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    start = Molecule.from_lines("O 0 0 0.2\nH 0 0.9 -0.5\nH 0 -0.7 -0.4")
    res = optimize(start, PySCFBackend(method="hf", basis="sto-3g"), workdir=tmp_path,
                   workers=4, device="cpu", restart=False,
                   gradient=FDSettings(scheme="central"),
                   optimizer=OptimizerSettings())
    assert res.converged and res.cycles < 12
    mf = scf.RHF(pyscf_mol(res.molecule, "sto-3g")).run(conv_tol=1e-11)
    assert np.abs(mf.nuc_grad_method().kernel()).max() < 3e-4
    assert res.energy == pytest.approx(-74.96590, abs=1e-5)


def test_bad_options():
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    with pytest.raises(ValueError):
        PySCFBackend(excited="adc2")
    with pytest.raises(ValueError):
        PySCFBackend(method="casscf")
    with pytest.raises(RuntimeError):
        from ngeoopt.backends.base import JobContext
        from pathlib import Path
        PySCFBackend(use_gpu="always")._maybe_gpu(None, JobContext(Path("."), device="cpu"))
