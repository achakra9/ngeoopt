"""Runs examples/generic_code end to end (fake external program + template)."""

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from ngeoopt import run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "generic_code"
BRIGHT_MIN = -76.40 + 7.0 / 27.211386245988     # bright state minimum of fakeqc
DARK_MIN = -76.40 + 7.6 / 27.211386245988


@pytest.fixture
def example(tmp_path):
    for f in EXAMPLE.iterdir():
        if f.is_file():
            shutil.copy(f, tmp_path)

    def load(name, **overrides):
        cfg = yaml.safe_load((tmp_path / name).read_text())
        cfg["backend"]["command"] = f"{sys.executable} fakeqc.py {{input}} > {{output}}"
        cfg["output"]["workdir"] = str(tmp_path / cfg["output"]["workdir"])
        cfg.update(overrides)
        p = tmp_path / f"t_{name}"
        p.write_text(yaml.safe_dump(cfg))
        return p
    return load


def test_generic_gradient(example):
    res = run(example("gradient.yaml"))
    assert res.gradient.shape == (3, 3)
    assert abs(res.gradient).max() > 1e-2


def test_generic_ground_state_optimization(example):
    res = run(example("optimize.yaml"))
    assert res.optimization.converged
    assert res.energy == pytest.approx(-76.40, abs=1e-6)


def test_generic_excited_state_follows_bright_state(example):
    """Bright state starts as root 2 and becomes root 1; it must not end on
    the dark state's minimum."""
    res = run(example("excited.yaml"))
    opt = res.optimization
    assert opt.converged
    assert res.energy == pytest.approx(BRIGHT_MIN, abs=1e-6)
    assert abs(res.energy - DARK_MIN) > 1e-2
    roots = [h.root for h in opt.history]
    assert roots[0] == 2 and roots[-1] == 1
