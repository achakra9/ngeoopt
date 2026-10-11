import numpy as np
import pytest

import models
from ngeoopt import (ConvergenceCriteria, FDSettings, FiniteDifferenceGradient, FunctionBackend,
                     Molecule, Optimizer, OptimizerSettings, StateTracker)
from ngeoopt.cache import ResultCache
from ngeoopt.hessian import bfgs_update, lindh_hessian
from ngeoopt.parallel import SerialExecutor


def make_opt(tmp_path, settings=None, cache=True, fd=None):
    be = FunctionBackend(function="models:morse")
    rc = ResultCache(tmp_path / "cache", "morse") if cache else None
    grad = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path / "jobs"),
                                    StateTracker(), fd or FDSettings(scheme="central4"), rc)
    return Optimizer(grad, settings or OptimizerSettings(), workdir=tmp_path)


def pair_distances(x):
    x = np.asarray(x)
    return sorted(np.linalg.norm(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x)))


@pytest.mark.parametrize("method", ["rfo", "qn"])
def test_morse_triangle_minimum(tmp_path, triatomic, method):
    s = OptimizerSettings(method=method, convergence=ConvergenceCriteria.preset("tight"))
    res = make_opt(tmp_path, s).run(triatomic, restart=False)
    assert res.converged
    # all three pairs at the Morse minimum distance -> equilateral triangle
    np.testing.assert_allclose(pair_distances(res.molecule.coords), [1.8] * 3, atol=1e-4)
    assert res.energy == pytest.approx(-3 * 0.17, abs=1e-8)
    assert (tmp_path / "trajectory.xyz").exists()


def test_steepest_descent_makes_progress(tmp_path, triatomic):
    s = OptimizerSettings(method="sd", max_cycles=5)
    res = make_opt(tmp_path, s).run(triatomic, restart=False)
    assert res.energy < models.morse(None, triatomic.coords)


def test_restart_continues_from_checkpoint(tmp_path, triatomic):
    full = make_opt(tmp_path / "full").run(triatomic, restart=False)

    part = make_opt(tmp_path / "split", OptimizerSettings(max_cycles=2)).run(triatomic)
    assert not part.converged and part.cycles == 2
    opt = make_opt(tmp_path / "split")
    resumed = opt.run(triatomic, restart=True)
    assert resumed.converged
    assert resumed.cycles == full.cycles          # same path, picked up at cycle 3
    np.testing.assert_allclose(resumed.molecule.coords, full.molecule.coords, atol=1e-10)
    assert [h.cycle for h in resumed.history] == list(range(1, full.cycles + 1))

    # a converged checkpoint returns immediately without new jobs
    again = make_opt(tmp_path / "split").run(triatomic)
    assert again.converged and again.cycles == resumed.cycles


def test_restart_reuses_cached_single_points(tmp_path, triatomic):
    opt = make_opt(tmp_path, OptimizerSettings(max_cycles=1))
    opt.run(triatomic, restart=False)
    (tmp_path / "ngeoopt_state.npz").unlink()          # lose the optimizer state ...
    res = make_opt(tmp_path, OptimizerSettings(max_cycles=1)).run(triatomic, restart=True)
    assert res.history[0].ncached == res.history[0].njobs   # ... but not the energies


def test_checkpoint_for_other_molecule_ignored(tmp_path, triatomic):
    make_opt(tmp_path, OptimizerSettings(max_cycles=1)).run(triatomic)
    other = Molecule(["Y", "Y", "Y"], triatomic.coords)
    res = make_opt(tmp_path, OptimizerSettings(max_cycles=1)).run(other)
    assert res.cycles == 1


def test_convergence_presets_and_modes():
    c = ConvergenceCriteria.preset("gaussian")
    g_small = np.full(6, 1e-5)
    assert c.check(g_small, np.full(6, 1e-5), None)["converged"]
    assert c.check(g_small, np.full(6, 1.0), 1e-9)["converged"]       # energy rescues
    assert not c.check(g_small, np.full(6, 1.0), 1e-3)["converged"]
    assert not c.check(np.full(6, 1e-2), np.zeros(6), 0.0)["converged"]
    strict = ConvergenceCriteria.preset("gaussian", mode="all")
    assert not strict.check(g_small, np.full(6, 1.0), 1e-9)["converged"]
    assert ConvergenceCriteria.preset("tight", max_force=1.0).max_force == 1.0
    with pytest.raises(ValueError):
        ConvergenceCriteria.preset("whenever")


def test_lindh_hessian_properties(water):
    H = lindh_hessian(water.symbols, water.coords)
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12)
    w = np.linalg.eigvalsh(H)
    assert w.min() > -1e-8                    # positive semidefinite
    assert (w > 1e-3).sum() == 3              # 3 internal modes of a bent triatomic


def test_bfgs_update_secant():
    rng = np.random.default_rng(0)
    H = np.eye(4)
    s = rng.normal(size=4)
    y = 2.0 * s
    H2 = bfgs_update(H, s, y)
    np.testing.assert_allclose(H2 @ s, y)
    assert bfgs_update(H, s, -s) is H         # curvature condition fails -> unchanged


def test_frozen_atom_stays_put(tmp_path, triatomic):
    res = make_opt(tmp_path, fd=FDSettings(frozen_atoms=[0])).run(triatomic, restart=False)
    np.testing.assert_array_equal(res.molecule.coords[0], triatomic.coords[0])
    assert res.converged
