"""Energy backends and a small registry to build them by name."""

from __future__ import annotations

from typing import Dict, Type

from .base import EnergyBackend, JobContext
from .function import FunctionBackend
from .qchem import QChemBackend
from .template import TemplateBackend


def _pyscf():
    from .pyscf_backend import PySCFBackend  # imported lazily: pyscf is optional
    return PySCFBackend


def _exkit():
    from .exkit_backend import ExkitBackend  # imported lazily: exkit is optional
    return ExkitBackend


_REGISTRY: Dict[str, object] = {
    "pyscf": _pyscf,
    "exkit": _exkit,
    "qchem": QChemBackend,
    "template": TemplateBackend,
    "external": TemplateBackend,
    "function": FunctionBackend,
    "python": FunctionBackend,
}


def register_backend(name: str, cls: Type[EnergyBackend]) -> None:
    """Make a custom backend class available to config files by name."""
    _REGISTRY[name.lower()] = cls


def make_backend(name: str, **options) -> EnergyBackend:
    try:
        entry = _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    cls = entry() if entry in (_pyscf, _exkit) else entry
    return cls(**options)


__all__ = ["EnergyBackend", "JobContext", "FunctionBackend", "QChemBackend",
           "TemplateBackend", "make_backend", "register_backend"]
