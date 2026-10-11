import importlib.util

import numpy as np
import pytest

from ngeoopt import FiniteDifferenceGradient, FunctionBackend, Molecule
from ngeoopt import device as devmod
from ngeoopt.parallel import ProcessExecutor, SerialExecutor, make_executor
from ngeoopt.parallel.executors import Job


def test_resolve_cpu():
    info = devmod.resolve("cpu")
    assert info.kind == "cpu" and not info.is_gpu
    assert devmod.worker_device(info, 3) == "cpu"


def test_worker_device_round_robin():
    info = devmod.DeviceInfo("cuda", [0, 2])
    assert [devmod.worker_device(info, i) for i in range(4)] == \
        ["cuda:0", "cuda:2", "cuda:0", "cuda:2"]


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError):
        devmod.resolve("tpu")


@pytest.mark.skipif(devmod.cuda_device_count() > 0, reason="machine has CUDA")
def test_cuda_request_without_gpu_fails():
    with pytest.raises(RuntimeError):
        devmod.resolve("cuda")


def test_device_passed_to_function(tmp_path, triatomic):
    be = FunctionBackend(function="models:device_echo")
    ex = SerialExecutor(scratch=tmp_path, device="cpu")
    res = ex.map(be, [Job("a", triatomic)])
    assert res[0].states[0].extra["device"] == "cpu"
    assert res[0].extra["device"] == "cpu"


def test_auto_worker_count(tmp_path):
    with make_executor("process", nworkers="auto", nthreads=2, device="cpu",
                       scratch=tmp_path) as ex:
        import os
        assert ex.nworkers == max(1, (os.cpu_count() or 1) // 2)


def test_single_worker_is_serial(tmp_path):
    with make_executor("process", nworkers=1, device="cpu", scratch=tmp_path) as ex:
        assert isinstance(ex, SerialExecutor)


def test_threads_are_set_in_workers(tmp_path, triatomic):
    be = FunctionBackend(function="test_device_parallel:report_threads")
    with ProcessExecutor(2, nthreads=3, scratch=tmp_path) as ex:
        res = ex.map(be, [Job(f"j{i}", triatomic) for i in range(4)])
    assert all(r.states[0].extra["omp"] == "3" for r in res)


def report_threads(symbols, coords):
    import os
    from ngeoopt.results import StateInfo
    return {"states": [StateInfo(0.0, extra={"omp": os.environ.get("OMP_NUM_THREADS")})]}


def test_keep_files(tmp_path, triatomic):
    be = FunctionBackend(function="models:morse")
    SerialExecutor(scratch=tmp_path / "k", keep_files=True).map(be, [Job("c/x", triatomic)])
    assert (tmp_path / "k" / "c" / "x").is_dir()
    SerialExecutor(scratch=tmp_path / "d").map(be, [Job("c/x", triatomic)])
    assert not (tmp_path / "d" / "c" / "x").exists()


@pytest.mark.skipif(not devmod.mps_available(), reason="needs Apple MPS")
def test_torch_function_on_mps(tmp_path, triatomic):
    be = FunctionBackend(function="models:torch_harmonic")
    with make_executor("process", nworkers=2, device="mps", scratch=tmp_path) as ex:
        gr = FiniteDifferenceGradient(be, ex).compute(triatomic)
    # float32 on the GPU: only a loose check of g = x (rigid part projected out)
    from ngeoopt.geometry import project_rigid
    ref = project_rigid(triatomic.coords, triatomic.coords)
    assert np.abs(gr.gradient - ref).max() < 5e-3


@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is not None, reason="mpi4py installed")
def test_mpi_executor_needs_mpi4py(tmp_path):
    with pytest.raises(RuntimeError, match="mpi4py"):
        make_executor("mpi", nworkers=2, device="cpu", scratch=tmp_path)
