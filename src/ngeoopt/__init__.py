"""ngeoopt: parallel finite-difference geometry optimization for any energy code."""

__version__ = "0.2.0"

from .backends import (EnergyBackend, FunctionBackend, JobContext, QChemBackend,  # noqa: E402
                       TemplateBackend, make_backend, register_backend)
from .findiff import FDSettings, FiniteDifferenceGradient, GradientResult  # noqa: E402
from .geometry import Molecule, read_xyz, write_xyz  # noqa: E402
from .optimizer import ConvergenceCriteria, Optimizer, OptimizerSettings  # noqa: E402
from .results import EnergyResult, StateInfo  # noqa: E402
from .tracking import StateTracker, TrackingSettings  # noqa: E402
from .driver import gradient, optimize, run  # noqa: E402

__all__ = [
    "__version__", "EnergyBackend", "FunctionBackend", "JobContext", "QChemBackend",
    "TemplateBackend", "make_backend", "register_backend", "FDSettings",
    "FiniteDifferenceGradient", "GradientResult", "Molecule", "read_xyz", "write_xyz",
    "ConvergenceCriteria", "Optimizer", "OptimizerSettings", "EnergyResult", "StateInfo",
    "StateTracker", "TrackingSettings", "gradient", "optimize", "run",
]
