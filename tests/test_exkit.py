"""exkit backend: energies vs exkit.run, FD gradients vs analytic, SF/MRSF
optimizations, geometry-transferable signatures."""

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_PYSCF

HAS_EXKIT = importlib.util.find_spec("exkit") is not None
pytestmark = pytest.mark.skipif(not (HAS_EXKIT and HAS_PYSCF), reason="needs exkit + pyscf")

from ngeoopt import (FDSettings, FiniteDifferenceGradient, JobContext, Molecule,  # noqa: E402
                     StateTracker, TrackingSettings, gradient, optimize, run)
from ngeoopt.parallel import make_executor  # noqa: E402
from ngeoopt.tracking import signature_overlap  # noqa: E402


def backend(**inp):
    from ngeoopt.backends.exkit_backend import ExkitBackend
    return ExkitBackend(input=inp)


@pytest.fixture
def ch2():
    return Molecule.from_lines("C 0 0 0.1\nH 0 0.99 -0.55\nH 0 -0.99 -0.55", multiplicity=3)


def exkit_reference(mol, inp, tmp_path):
    import exkit
    geo = "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(mol.symbols, mol.coords))
    np.random.seed(0)                       # same RNG state as the backend's jobs
    with contextlib.redirect_stdout(io.StringIO()):
        return exkit.run(dict(inp, geometry=geo, units="bohr", spin=mol.multiplicity - 1,
                              analysis=False, verbose=0, banner=False, conv_tol=1e-7,
                              max_cycle=200, outdir=str(tmp_path / "ref")))


@pytest.mark.parametrize("method,molname", [("cis", "water"), ("sf-tda", "ch2"),
                                            ("mrsf-tda", "ch2")])
def test_energies_match_exkit_run(method, molname, water, ch2, tmp_path):
    mol = water if molname == "water" else ch2
    inp = {"method": method, "basis": "6-31g", "nstates": 4}
    res = backend(**inp).compute(mol, JobContext(tmp_path / "job"))
    ref = exkit_reference(mol, inp, tmp_path)
    roots = [s for s in res.states if not s.extra.get("scf_reference")]
    np.testing.assert_allclose([s.energy for s in roots], np.asarray(ref.e_tot), atol=1e-9)
    # reference in the manifold only for spin-conserving CIS
    assert res.states[0].extra.get("scf_reference", False) == (method == "cis")
    assert all(s.extra["s2"] is not None for s in roots)
    assert all(np.ndim(s.signature) == 3 for s in roots)
    assert res.extra["guess"].shape[-1] == 13


def test_cis_s1_gradient_matches_pyscf_analytic(water, tmp_path):
    from pyscf import gto, scf, tdscf
    g = gradient(water, backend(method="cis", basis="6-31g", nstates=4), workers=6,
                 device="cpu", scratch=tmp_path, state=TrackingSettings(root=1, multiplicity=1))
    assert g.tracking["method"] == "composite" and g.tracking["min_overlap"] > 0.999
    m = gto.M(atom=[(s, tuple(c)) for s, c in zip(water.symbols, water.coords)],
              unit="Bohr", basis="6-31g", verbose=0)
    mf = scf.RHF(m).run(conv_tol=1e-12, conv_tol_grad=1e-8)
    td = tdscf.TDA(mf)
    td.nstates, td.conv_tol = 3, 1e-10
    td.kernel()
    assert g.energy == pytest.approx(mf.e_tot + td.e[0], abs=1e-8)
    assert np.abs(g.gradient - td.nuc_grad_method().kernel(state=1)).max() < 1.5e-5


def test_signatures_transfer_between_geometries(ch2, tmp_path):
    """Separate SCFs at two geometries (arbitrary orbital phases): each root's
    AO excitation density overlaps ~1 with itself and little with the others."""
    be = backend(method="mrsf-tda", basis="6-31g", nstates=4)
    a = be.compute(ch2, JobContext(tmp_path / "a"))
    x = ch2.coords.copy()
    x[1, 1] += 0.02
    b = be.compute(ch2.with_coords(x), JobContext(tmp_path / "b"))
    S = a.ao_overlap
    ov = np.array([[signature_overlap(p.signature, q.signature, S) for q in b.states]
                   for p in a.states])
    assert np.all(np.diag(ov) > 0.99)
    off = ov[~np.eye(len(ov), dtype=bool)]
    assert off.max() < 0.5


@pytest.mark.parametrize("method", ["mrsf-tda", "sf-tda"])
def test_spin_flip_s1_optimization(method, ch2, tmp_path):
    res = optimize(ch2, backend(method=method, basis="6-31g", nstates=6), workers=8,
                   device="cpu", workdir=tmp_path, restart=False,
                   state=TrackingSettings(root=1, multiplicity=1))
    assert res.converged and res.cycles <= 12
    x = res.molecule.coords
    u, v = x[1] - x[0], x[2] - x[0]
    angle = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    assert 135 < angle < 150                   # open-shell singlet of CH2 is wide
    assert np.abs(res.gradient).max() < 4.5e-4


def test_centre_density_seeds_displacements(ch2, tmp_path):
    be = backend(method="sf-tda", basis="sto-3g", nstates=3)
    with make_executor("serial", device="cpu", scratch=tmp_path) as ex:
        fd = FiniteDifferenceGradient(be, ex, StateTracker(TrackingSettings(root=0)),
                                      FDSettings(guess_from_center=True))
        fd.compute(ch2)
        assert fd.last_guess is not None and fd.last_guess.shape == (2, 7, 7)


def test_unconverged_scf_fails_the_job(ch2, tmp_path):
    be = backend(method="sf-tda", basis="6-31g", nstates=2, scf_max_cycle=1)
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        be.compute(ch2, JobContext(tmp_path))


def test_bad_exkit_keyword_fails_early():
    with pytest.raises(KeyError, match="nstate"):
        backend(method="cis", nstate=3)


def test_unsupported_method(tmp_path, water):
    with pytest.raises(NotImplementedError):
        backend(method="ip-eom-mp2", basis="sto-3g", nstates=2).compute(water, JobContext(tmp_path))


def test_input_file_and_yaml(tmp_path, ch2):
    inp = tmp_path / "ch2.inp"
    inp.write_text("method = sf-tda\nbasis = sto-3g\nnstates = 3\n$geometry\nHe 0 0 0\n$end\n")
    cfg = {"task": "energy",
           "molecule": {"geometry": "C 0 0 0.1\nH 0 0.99 -0.55\nH 0 -0.99 -0.55",
                        "multiplicity": 3},
           "backend": {"name": "exkit", "input_file": str(inp)},
           "state": {"root": 1, "multiplicity": 1},
           "parallel": {"executor": "serial"},
           "output": {"workdir": str(tmp_path / "run")}}
    res = run(cfg)
    ref = backend(method="sf-tda", basis="sto-3g", nstates=3).compute(
        ch2, JobContext(tmp_path / "x"))
    singlets = [s for s in ref.states if s.multiplicity == 1]
    assert res.energy == pytest.approx(singlets[1].energy, abs=1e-10)


def test_cis_transition_dipoles_match_pyscf(water, tmp_path):
    from ngeoopt.backends.pyscf_backend import PySCFBackend
    ex = backend(method="cis", basis="6-31g", nstates=4).compute(water, JobContext(tmp_path / "a"))
    py = PySCFBackend(method="hf", basis="6-31g", excited="cis", nroots=2).compute(
        water, JobContext(tmp_path / "b"))
    ex_s = [s for s in ex.states if s.multiplicity == 1 and not s.extra.get("scf_reference")]
    for a, b in zip(ex_s, py.states[1:]):
        assert a.energy == pytest.approx(b.energy, abs=1e-8)
        assert a.oscillator_strength == pytest.approx(b.oscillator_strength, abs=1e-6)
        np.testing.assert_allclose(np.abs(a.extra["trans_mom"]), np.abs(b.extra["trans_mom"]),
                                   atol=1e-5)
    triplets = [s for s in ex.states if s.multiplicity == 3]
    assert all(s.oscillator_strength == pytest.approx(0.0, abs=1e-10) for s in triplets)


def test_spin_flip_and_doubles_evidence(ch2, tmp_path):
    sf = backend(method="sf-tda", basis="sto-3g", nstates=3).compute(ch2, JobContext(tmp_path / "a"))
    assert all(s.extra["trans_mom"] is None and s.oscillator_strength is None for s in sf.states)
    assert all(s.signature is not None for s in sf.states)
    cisd = backend(method="sf-cisd", basis="sto-3g", nstates=3).compute(ch2, JobContext(tmp_path / "b"))
    assert all(s.signature is None for s in cisd.states)   # no singles-only overlaps
    assert all(s.extra["s2"] is not None for s in cisd.states)


@pytest.mark.parametrize("method,molname,extra", [
    ("tda", "water", {}), ("sf-cis", "ch2", {}), ("mrsf-cis", "ch2", {}),
    ("mrsf-tddft", "ch2", {}), ("md-sf-tda", "ch2", {}), ("sf-cisd", "ch2", {}),
    ("pp-tda", "water", {}), ("hh-tda", "water", {}), ("dea", "water", {}),
    ("ip-cis", "water", {}), ("ea-tda", "water", {}), ("mp2", "water", {}),
    ("b2plyp", "water", {})])
def test_method_sweep_matches_exkit_run(method, molname, extra, water, ch2, tmp_path):
    mol = water if molname == "water" else ch2
    inp = dict({"method": method, "basis": "sto-3g", "nstates": 3}, **extra)
    res = backend(**inp).compute(mol, JobContext(tmp_path / "job"))
    ref = exkit_reference(mol, dict(inp, scf_conv_tol_grad=1e-7), tmp_path)
    ref_e = np.atleast_1d(np.asarray(ref.e_tot, dtype=float))
    mine = [s.energy for s in res.states if not s.extra.get("scf_reference")] \
        if len(res.states) > 1 else [res.energy]
    n = min(len(mine), ref_e.size)
    np.testing.assert_allclose(mine[:n], ref_e[:n], atol=1e-9)


def test_geometry_unsupported_methods_are_refused(water, tmp_path):
    with pytest.raises(NotImplementedError, match="geometry derivatives"):
        backend(method="tea", basis="sto-3g", nstates=2).compute(water, JobContext(tmp_path))


def test_casscf_is_reproducible_within_one_worker(water, tmp_path):
    """This CAS(4,4) has two SA-CASSCF solutions (-74.9800 / -74.9684 Eh) and
    exkit picks one through numpy's global RNG (initial CI guess). Every job of a
    run must land on the same one, whatever ran before in the worker; the
    backend reseeds per job. (Matching exkit.run's pick is not meaningful: it
    consumes random numbers in its own setup first.)"""
    import exkit  # noqa: F401
    np.random.seed(12345)                     # disturb the global RNG first
    be = backend(method="casscf", basis="sto-3g", nstates=3, cas_ncas=4, cas_nelec=4)
    energies = [be.compute(water, JobContext(tmp_path / f"j{i}")).energies for i in range(4)]
    np.testing.assert_allclose(energies, [energies[0]] * 4, atol=1e-8)
