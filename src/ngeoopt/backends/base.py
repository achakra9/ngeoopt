"""Abstract energy backend.

A backend only has to know how to compute energies (and optionally excited
states) at a single geometry. Everything else -- displacements, parallel
dispatch, state tracking, optimization -- lives in ngeoopt itself.

Backends are pickled and shipped to worker processes, so keep them as plain
data (configuration only) and create heavy objects inside ``compute``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..geometry.molecule import Molecule
from ..results import EnergyResult


@dataclass
class JobContext:
    """Per-job information handed to a backend by the executor."""

    workdir: Path                 # private directory for this job
    label: str = "job"           # e.g. 'ref', 'd0003+', 'step005'
    nthreads: int = 1
    device: str = "cpu"          # 'cpu', 'cuda', 'cuda:1', 'mps'
    keep_files: bool = False
    guess: Any = None            # SCF initial guess (e.g. density of the centre geometry)


class EnergyBackend(abc.ABC):
    name: str = "base"

    #: whether this backend can produce overlap signatures for root following
    supports_signatures: bool = False

    #: whether compute() uses ctx.guess and returns result.extra["guess"]
    supports_guess: bool = False

    @abc.abstractmethod
    def compute(self, mol: Molecule, ctx: JobContext) -> EnergyResult:
        """Return energies of the requested state(s) at ``mol``."""

    def describe(self) -> str:
        return self.name

    def nstates_hint(self) -> Optional[int]:
        """Number of states the backend returns, if known in advance."""
        return None
