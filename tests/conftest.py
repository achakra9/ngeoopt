import importlib.util
import os
import sys

import numpy as np
import pytest

# make tests/models.py importable in spawned worker processes too
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from ngeoopt import Molecule  # noqa: E402

HAS_PYSCF = importlib.util.find_spec("pyscf") is not None


@pytest.fixture
def triatomic():
    # a bent, slightly asymmetric 3-atom Morse cluster away from its minimum
    return Molecule(["X", "X", "X"], np.array([[0.0, 0.0, 0.0],
                                               [2.1, 0.1, 0.0],
                                               [0.9, 1.7, 0.2]]))


@pytest.fixture
def water():
    return Molecule.from_lines("O 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467")
