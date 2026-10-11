import json

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from ngeoopt import run
from ngeoopt.cli import app
from ngeoopt.config import EXAMPLE_CONFIG, config_from_dict, load_config

MORSE_CFG = {
    "molecule": {"geometry": "X 0 0 0\nX 1.1 0.05 0\nX 0.5 0.9 0.1", "units": "angstrom"},
    "backend": {"name": "function", "function": "models:morse"},
    "gradient": {"scheme": "central4"},
    "parallel": {"executor": "serial"},
    "optimizer": {"convergence": {"preset": "tight"}},
}


def test_example_config_parses(tmp_path):
    data = yaml.safe_load(EXAMPLE_CONFIG)
    cfg = config_from_dict(data)
    assert cfg.backend["name"] == "pyscf"
    assert cfg.optimizer.convergence.max_force == 4.5e-4
    assert cfg.gradient.step == 0.005


def test_unknown_keywords_are_rejected():
    with pytest.raises(ValueError, match="unknown keyword.*stepsize"):
        config_from_dict({**MORSE_CFG, "gradient": {"stepsize": 0.01}})
    with pytest.raises(ValueError, match="unknown section"):
        config_from_dict({**MORSE_CFG, "solver": {}})
    with pytest.raises(ValueError, match="backend section needs"):
        config_from_dict({**MORSE_CFG, "backend": {}})
    with pytest.raises(ValueError, match="tracking method"):
        config_from_dict({**MORSE_CFG, "state": {"method": "guess"}})


def test_convergence_spec_forms():
    c = config_from_dict({**MORSE_CFG, "optimizer": {"convergence": "loose"}})
    assert c.optimizer.convergence.max_force == 2.5e-3
    c = config_from_dict({**MORSE_CFG, "optimizer": {"convergence": {"max_force": 1e-5}}})
    assert c.optimizer.convergence.max_force == 1e-5


def test_run_tasks(tmp_path):
    base = {**MORSE_CFG, "output": {"workdir": str(tmp_path / "w")}}
    e = run({**base, "task": "energy"})
    g = run({**base, "task": "gradient"})
    assert g.energy == pytest.approx(e.energy)
    assert (tmp_path / "w" / "gradient.txt").exists()
    o = run(base)
    assert o.optimization.converged
    assert o.energy == pytest.approx(-0.51, abs=1e-8)
    assert (tmp_path / "w" / "final.xyz").exists()
    assert (tmp_path / "w" / "ngeoopt.log").read_text().count("CONVERGED") == 1


def test_xyz_path_relative_to_config(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "m.xyz").write_text("2\n\nX 0 0 0\nX 0 0 1.0\n")
    cfg = {**MORSE_CFG, "molecule": {"xyz": "m.xyz"}}
    p = tmp_path / "sub" / "in.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert load_config(p).build_molecule().natoms == 2


def test_cli_init_run_status(tmp_path):
    runner = CliRunner()
    r = runner.invoke(app, ["init", str(tmp_path / "x.yaml")])
    assert r.exit_code == 0 and (tmp_path / "x.yaml").exists()
    r = runner.invoke(app, ["init", str(tmp_path / "x.yaml")])
    assert r.exit_code == 1

    cfg = {**MORSE_CFG, "output": {"workdir": "out"}}
    p = tmp_path / "morse.yaml"
    p.write_text(yaml.safe_dump(cfg))
    r = runner.invoke(app, ["run", str(p), "--workers", "1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["status", str(tmp_path / "out")])
    assert r.exit_code == 0 and "converged=True" in r.output
    meta = json.loads((tmp_path / "out" / "ngeoopt_state.json").read_text())
    assert meta["converged"]

    r = runner.invoke(app, ["devices"])
    assert r.exit_code == 0 and "cpu cores" in r.output


def test_cli_nonconverged_exit_code(tmp_path):
    cfg = {**MORSE_CFG, "optimizer": {"max_cycles": 1},
           "output": {"workdir": str(tmp_path / "o")}}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(cfg))
    r = CliRunner().invoke(app, ["run", str(p)])
    assert r.exit_code == 2
