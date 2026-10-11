import numpy as np
import pytest

import models
from ngeoopt import (FDSettings, FiniteDifferenceGradient, FunctionBackend, StateTracker,
                     TrackingSettings)
from ngeoopt.cache import ResultCache
from ngeoopt.findiff import displacement_plan
from ngeoopt.parallel import JobFailed, SerialExecutor, make_executor


def fd_gradient(mol, settings=None, executor=None, cache=None, tmp="scr"):
    be = FunctionBackend(function="models:morse")
    ex = executor or SerialExecutor(nthreads=1, scratch=tmp)
    return FiniteDifferenceGradient(be, ex, StateTracker(), settings, cache).compute(mol)


@pytest.mark.parametrize("scheme,tol", [("forward", 2e-3), ("central", 2e-5), ("central4", 1e-8)])
def test_schemes_against_analytic(triatomic, tmp_path, scheme, tol):
    s = FDSettings(scheme=scheme, step=0.005, project_rigid=False)
    gr = fd_gradient(triatomic, s, tmp=tmp_path)
    ref = models.morse_gradient(triatomic.coords)
    assert np.abs(gr.gradient - ref).max() < tol
    assert gr.energy == pytest.approx(models.morse(None, triatomic.coords))


def test_job_counts(triatomic):
    for scheme, per in (("forward", 1), ("central", 2), ("central4", 4)):
        assert len(displacement_plan(triatomic, FDSettings(scheme=scheme))) == 9 * per
    plan = displacement_plan(triatomic, FDSettings(frozen_atoms=[1]))
    assert {a for _, a, _, _ in plan} == {0, 2}


def test_frozen_atoms_zero_gradient(triatomic, tmp_path):
    gr = fd_gradient(triatomic, FDSettings(frozen_atoms=[0]), tmp=tmp_path)
    assert np.all(gr.gradient[0] == 0)
    ref = models.morse_gradient(triatomic.coords)
    assert np.abs(gr.gradient[1:] - ref[1:]).max() < 2e-5


def test_rigid_projection_keeps_physical_gradient(triatomic, tmp_path):
    # an isolated-molecule gradient has no net force/torque, so projecting
    # must leave it (nearly) unchanged
    gr = fd_gradient(triatomic, FDSettings(project_rigid=True), tmp=tmp_path)
    ref = models.morse_gradient(triatomic.coords)
    assert np.abs(gr.gradient - ref).max() < 2e-5


def test_process_pool_matches_serial(triatomic, tmp_path):
    serial = fd_gradient(triatomic, tmp=tmp_path / "a")
    with make_executor("process", nworkers=3, device="cpu", scratch=tmp_path / "b") as ex:
        assert ex.nworkers == 3
        par = fd_gradient(triatomic, executor=ex)
    np.testing.assert_allclose(par.gradient, serial.gradient, atol=1e-14)


def test_cache_skips_finished_jobs(triatomic, tmp_path):
    cache = ResultCache(tmp_path / "cache", "morse")
    first = fd_gradient(triatomic, cache=cache, tmp=tmp_path)
    assert first.ncached == 0 and len(cache) == first.njobs
    again = fd_gradient(triatomic, cache=cache, tmp=tmp_path)
    assert again.ncached == again.njobs
    np.testing.assert_array_equal(first.gradient, again.gradient)


def test_failed_job_raises(triatomic, tmp_path):
    be = FunctionBackend(function="models:flaky")
    fd = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path, retries=1))
    with pytest.raises(JobFailed, match="always fails"):
        fd.compute(triatomic)


def test_bad_settings():
    with pytest.raises(ValueError):
        FDSettings(scheme="sideways")
    with pytest.raises(ValueError):
        FDSettings(step=0)


def test_unconverged_target_state_is_reported(triatomic, tmp_path, caplog):
    be = FunctionBackend(function="models:unconverged_states")
    tr = StateTracker(TrackingSettings(root=1))
    fd = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path), tr)
    gr = fd.compute(triatomic)
    assert gr.tracking["unconverged"] == gr.njobs
    assert "not converged" in caplog.text
    # following the ground state of the same backend raises no warning
    fd0 = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path))
    assert fd0.compute(triatomic).tracking["unconverged"] == 0


def test_cache_tag_depends_on_all_settings():
    from ngeoopt.cache import backend_tag
    a = FunctionBackend(function="models:morse", kwargs={"De": 0.17})
    b = FunctionBackend(function="models:morse", kwargs={"De": 0.20})
    assert backend_tag(a) != backend_tag(b)
    assert backend_tag(a) == backend_tag(FunctionBackend(function="models:morse",
                                                         kwargs={"De": 0.17}))
