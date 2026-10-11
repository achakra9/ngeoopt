"""Executors that evaluate many single-point energies concurrently.

All executors expose ``map(backend, jobs) -> list[EnergyResult]`` with the
results in the same order as ``jobs``. They are context managers so that a
worker pool survives across optimization cycles (PySCF imports and GPU
contexts are created only once per worker).

* ``SerialExecutor``  -- in-process, one job at a time (debugging, tiny jobs)
* ``ProcessExecutor`` -- a local process pool; each worker gets its own
  thread count and, on CUDA machines, its own GPU
* ``MPIExecutor``     -- ``mpi4py.futures`` pool for multi-node runs
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .. import device as devmod
from ..backends.base import EnergyBackend, JobContext
from ..geometry.molecule import Molecule
from ..results import EnergyResult

log = logging.getLogger(__name__)

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

# per-process state, filled by the worker initializer
_WORKER = {"device": "cpu", "nthreads": 1}


@dataclass
class Job:
    label: str
    mol: Molecule
    guess: Any = None        # passed to the backend as ctx.guess


class JobFailed(RuntimeError):
    pass


def _set_threads(n: int) -> None:
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    try:  # pyscf caches its own thread count
        from pyscf import lib
        lib.num_threads(n)
    except Exception:
        pass


def _init_worker(slots, nthreads: int, device_kind: str) -> None:
    """Runs once in each worker process before any job."""
    _set_threads(nthreads)
    _WORKER["nthreads"] = nthreads
    if device_kind == "cuda":
        gpu = slots.get()
        # pin the worker to one physical GPU; inside the worker it is cuda:0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        _WORKER["device"] = "cuda:0"
        _WORKER["physical_gpu"] = gpu
    else:
        _WORKER["device"] = device_kind


def _init_mpi_worker(nthreads: int, device_kind: str, gpu_ids: Sequence[int]) -> None:
    local = None
    for var in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID",
                "PMI_LOCAL_RANK"):
        if var in os.environ:
            local = int(os.environ[var])
            break
    if local is None:
        from mpi4py import MPI
        local = MPI.COMM_WORLD.Get_rank()
    _set_threads(nthreads)
    _WORKER["nthreads"] = nthreads
    if device_kind == "cuda" and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[local % len(gpu_ids)])
        _WORKER["device"] = "cuda:0"
    else:
        _WORKER["device"] = device_kind


def run_job(backend: EnergyBackend, job: Job, root: str, keep_files: bool,
            retries: int = 0) -> EnergyResult:
    """Execute one job in the current process (used by every executor)."""
    workdir = Path(root) / job.label
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = JobContext(workdir=workdir, label=job.label, nthreads=_WORKER["nthreads"],
                     device=_WORKER["device"], keep_files=keep_files, guess=job.guess)
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            res = backend.compute(job.mol, ctx)
            res.wall_time = time.perf_counter() - t0
            res.extra.setdefault("label", job.label)
            res.extra.setdefault("device", ctx.device)
            if not keep_files:
                shutil.rmtree(workdir, ignore_errors=True)
            return res
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            last_err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            log.warning("job %s failed (attempt %d/%d): %s", job.label,
                        attempt + 1, retries + 1, exc)
    raise JobFailed(f"job {job.label} failed after {retries + 1} attempt(s):\n{last_err}")


class BaseExecutor:
    nworkers: int = 1

    def __init__(self, scratch: str | Path = "scratch", keep_files: bool = False,
                 retries: int = 0):
        self.scratch = Path(scratch).resolve()
        self.keep_files = keep_files
        self.retries = retries

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def shutdown(self) -> None:
        pass

    def map(self, backend: EnergyBackend, jobs: List[Job],
            on_result: Optional[Callable[[int, EnergyResult], None]] = None
            ) -> List[EnergyResult]:
        """Run ``jobs``; ``on_result(i, result)`` fires as each one finishes."""
        raise NotImplementedError


class SerialExecutor(BaseExecutor):
    def __init__(self, nthreads: Optional[int] = None, device: str = "cpu", **kw):
        super().__init__(**kw)
        self.nthreads = nthreads or os.cpu_count() or 1
        self.device = device

    def map(self, backend, jobs, on_result=None):
        _set_threads(self.nthreads)
        _WORKER.update(nthreads=self.nthreads, device=self.device)
        results = []
        for i, j in enumerate(jobs):
            res = run_job(backend, j, str(self.scratch), self.keep_files, self.retries)
            if on_result is not None:
                on_result(i, res)
            results.append(res)
        return results


class _PoolExecutor(BaseExecutor):
    """Shared map() for concurrent.futures-style pools."""

    _pool = None

    def map(self, backend, jobs, on_result=None):
        futures = {self._pool.submit(run_job, backend, j, str(self.scratch),
                                     self.keep_files, self.retries): i
                   for i, j in enumerate(jobs)}
        results: List[Optional[EnergyResult]] = [None] * len(jobs)
        errors = []
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
                if on_result is not None:
                    on_result(i, results[i])
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if errors:
            raise JobFailed(f"{len(errors)} job(s) failed; first error:\n{errors[0]}")
        return results  # type: ignore[return-value]

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


class ProcessExecutor(_PoolExecutor):
    def __init__(self, nworkers: int, nthreads: int = 1,
                 device: Optional[devmod.DeviceInfo] = None, **kw):
        super().__init__(**kw)
        import multiprocessing as mp

        device = device or devmod.DeviceInfo("cpu")
        self.nworkers = nworkers
        # 'spawn' is required for CUDA and is the safe choice with OpenMP libs
        ctx = mp.get_context("spawn")
        slots = ctx.Queue()
        if device.kind == "cuda":
            for w in range(nworkers):
                slots.put(device.gpu_ids[w % len(device.gpu_ids)])
        self._pool = ProcessPoolExecutor(
            max_workers=nworkers, mp_context=ctx, initializer=_init_worker,
            initargs=(slots, nthreads, device.kind))


class MPIExecutor(_PoolExecutor):
    def __init__(self, nworkers: Optional[int] = None, nthreads: int = 1,
                 device: Optional[devmod.DeviceInfo] = None, **kw):
        super().__init__(**kw)
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError as exc:
            raise RuntimeError("executor 'mpi' needs mpi4py (pip install ngeoopt[mpi])") from exc
        device = device or devmod.DeviceInfo("cpu")
        self._pool = MPIPoolExecutor(max_workers=nworkers, initializer=_init_mpi_worker,
                                     initargs=(nthreads, device.kind, device.gpu_ids))
        self.nworkers = nworkers or 1


def make_executor(kind: str = "process", nworkers="auto", nthreads: int = 1,
                  device: str = "auto", gpu_ids=None, workers_per_gpu: int = 1,
                  scratch="scratch", keep_files: bool = False,
                  retries: int = 0) -> BaseExecutor:
    """Build an executor from user settings."""
    info = devmod.resolve(device, gpu_ids)
    kind = kind.lower()
    common = dict(scratch=scratch, keep_files=keep_files, retries=retries)
    if nworkers in (None, "auto"):
        if info.kind == "cuda":
            nworkers = len(info.gpu_ids) * workers_per_gpu
        else:
            # MPS: one Metal device shared by all workers. Codes without a Metal
            # backend (PySCF) run on the CPU cores, so size the pool by cores.
            nworkers = max(1, (os.cpu_count() or 1) // max(1, nthreads))
    nworkers = int(nworkers)
    log.info("executor=%s workers=%d threads/worker=%d device=%s %s", kind, nworkers,
             nthreads, info.kind, info.gpu_ids or "")
    if kind == "serial" or (kind == "process" and nworkers == 1):
        dev = devmod.worker_device(info, 0)
        return SerialExecutor(nthreads=nthreads, device=dev, **common)
    if kind == "process":
        return ProcessExecutor(nworkers, nthreads, info, **common)
    if kind == "mpi":
        return MPIExecutor(nworkers, nthreads, info, **common)
    raise ValueError(f"Unknown executor {kind!r} (serial | process | mpi)")
