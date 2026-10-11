"""Device detection and assignment.

ngeoopt never needs a GPU itself -- finite-difference bookkeeping is tiny.
The expensive part is the energy evaluation, so the device layer decides
which accelerator each worker hands to its backend:

* NVIDIA: ``cuda`` / ``cuda:<id>``. PySCF uses gpu4pyscf; workers are
  pinned to GPUs round-robin via ``CUDA_VISIBLE_DEVICES``.
* Apple silicon: ``mps``. Passed to user energy functions written in torch
  (ML potentials etc.). PySCF has no Metal backend and falls back to CPU.
* ``cpu`` everywhere else.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    kind: str                               # 'cpu' | 'cuda' | 'mps'
    gpu_ids: List[int] = field(default_factory=list)
    detail: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def cuda_device_count() -> int:
    """Number of visible CUDA devices, without initialising a CUDA context
    in the parent process when we can avoid it."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return 0
    if _has("cupy"):
        try:
            import cupy
            return int(cupy.cuda.runtime.getDeviceCount())
        except Exception:
            pass
    if _has("torch"):
        try:
            import torch
            return int(torch.cuda.device_count())
        except Exception:
            pass
    return 0


def mps_available() -> bool:
    if platform.system() != "Darwin" or not _has("torch"):
        return False
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def detect() -> DeviceInfo:
    n = cuda_device_count()
    if n > 0:
        return DeviceInfo("cuda", list(range(n)), f"{n} CUDA device(s)")
    if mps_available():
        return DeviceInfo("mps", [0], "Apple Metal (MPS)")
    return DeviceInfo("cpu", [], platform.processor() or platform.machine())


def resolve(requested: str = "auto", gpu_ids: Optional[List[int]] = None) -> DeviceInfo:
    """Turn a user setting ('auto', 'cpu', 'cuda', 'cuda:1', 'mps') into a DeviceInfo."""
    req = (requested or "auto").lower()
    if req == "auto":
        info = detect()
    elif req == "cpu":
        info = DeviceInfo("cpu")
    elif req.startswith("cuda"):
        n = cuda_device_count()
        if n == 0:
            raise RuntimeError("device 'cuda' requested but no CUDA device is visible "
                               "(install cupy / gpu4pyscf, check CUDA_VISIBLE_DEVICES)")
        ids = [int(req.split(":")[1])] if ":" in req else list(range(n))
        info = DeviceInfo("cuda", ids, f"{n} CUDA device(s)")
    elif req == "mps":
        if not mps_available():
            raise RuntimeError("device 'mps' requested but torch MPS is not available")
        info = DeviceInfo("mps", [0], "Apple Metal (MPS)")
    else:
        raise ValueError(f"Unknown device {requested!r}")
    if gpu_ids and info.kind == "cuda":
        info.gpu_ids = list(gpu_ids)
    return info


def worker_device(info: DeviceInfo, worker_index: int) -> str:
    """Device string a given worker should use."""
    if info.kind == "cuda":
        return f"cuda:{info.gpu_ids[worker_index % len(info.gpu_ids)]}"
    return info.kind


def to_numpy(x):
    """Copy cupy / torch arrays back to numpy; pass everything else through."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if type(x).__module__.startswith("cupy"):
        return x.get()
    return x


def summary() -> str:
    info = detect()
    lines = [f"auto-detected : {info.kind} ({info.detail})"]
    lines.append(f"cupy          : {'yes' if _has('cupy') else 'no'}")
    lines.append(f"gpu4pyscf     : {'yes' if _has('gpu4pyscf') else 'no'}")
    lines.append(f"torch         : {'yes' if _has('torch') else 'no'}")
    lines.append(f"torch MPS     : {'yes' if mps_available() else 'no'}")
    lines.append(f"mpi4py        : {'yes' if _has('mpi4py') else 'no'}")
    lines.append(f"pyscf         : {'yes' if _has('pyscf') else 'no'}")
    lines.append(f"cpu cores     : {os.cpu_count()}")
    return "\n".join(lines)
