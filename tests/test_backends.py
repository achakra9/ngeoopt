"""Template (external program), Q-Chem input writer and Python function backends."""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import models
from ngeoopt import (FiniteDifferenceGradient, FunctionBackend, JobContext, Molecule,
                     QChemBackend, StateTracker, TemplateBackend, make_backend)
from ngeoopt.backends.template import render
from ngeoopt.parallel import SerialExecutor, make_executor
from ngeoopt.units import HARTREE_TO_EV

# A stand-in "quantum chemistry program": reads the geometry block from an
# input file, prints a Morse energy and a couple of excited states.
FAKE_PROGRAM = textwrap.dedent("""
    import sys, numpy as np
    sys.path.insert(0, {models_dir!r})
    import models
    text = open(sys.argv[1]).read()
    rows = [l.split() for l in text.split("BEGIN")[1].split("END")[0].strip().splitlines()]
    x = np.array([[float(v) for v in r[1:4]] for r in rows]) / 0.529177210903
    e = models.morse(None, x)
    print("some banner text")
    print(f"  Final energy is   {{e: .12f}}")
    print(f"  Excited state 1: excitation energy (eV) = {{2.0:.6f}}")
    print(f"  Excited state 2: excitation energy (eV) = {{3.5:.6f}}")
    print("Normal termination")
""")

TEMPLATE = """! fake input {{label}}
charge {{charge}} mult {{multiplicity}} nproc {{nthreads}}
BEGIN
{{geometry}}
END
"""


@pytest.fixture
def fake_program(tmp_path):
    p = tmp_path / "fakeqc.py"
    p.write_text(FAKE_PROGRAM.format(models_dir=str(Path(models.__file__).parent)))
    return p


def template_backend(fake_program, **kw):
    opts = dict(template=TEMPLATE,
                command=f"{sys.executable} {fake_program} {{input}} > {{output}}",
                energy_regex=r"Final energy is\s+(\S+)",
                success_regex="Normal termination")
    opts.update(kw)
    return TemplateBackend(**opts)


def test_render_placeholders():
    assert render("a {{x}} b {{ y }}", {"x": 1, "y": "z"}) == "a 1 b z"
    assert render("$molecule {single} %block", {}) == "$molecule {single} %block"
    with pytest.raises(KeyError):
        render("{{missing}}", {})


def test_template_backend_gradient(tmp_path, triatomic, fake_program):
    be = template_backend(fake_program)
    with make_executor("process", nworkers=4, device="cpu", scratch=tmp_path / "jobs") as ex:
        gr = FiniteDifferenceGradient(be, ex).compute(triatomic)
    # geometry goes through 10-decimal Angstrom text, so compare loosely
    assert np.abs(gr.gradient - models.morse_gradient(triatomic.coords)).max() < 1e-5
    assert gr.energy == pytest.approx(models.morse(None, triatomic.coords), abs=1e-9)


def test_template_excited_states(tmp_path, triatomic, fake_program):
    be = template_backend(fake_program, states_regex=r"excitation energy \(eV\) = (\S+)",
                          states_are_excitations=True, states_units="ev")
    ctx = JobContext(workdir=tmp_path / "one", label="one")
    res = be.compute(triatomic, ctx)
    assert len(res.states) == 3
    assert res.states[0].extra["scf_reference"]
    assert res.states[2].energy - res.states[0].energy == pytest.approx(3.5 / HARTREE_TO_EV)
    assert "nproc 1" in (tmp_path / "one" / "job.inp").read_text()


def test_template_failures(tmp_path, triatomic, fake_program):
    ctx = JobContext(workdir=tmp_path / "x", label="x")
    with pytest.raises(RuntimeError, match="success_regex"):
        template_backend(fake_program, success_regex="never printed").compute(triatomic, ctx)
    with pytest.raises(RuntimeError, match="found nothing"):
        template_backend(fake_program, energy_regex="nope (\\S+)").compute(triatomic, ctx)
    with pytest.raises(RuntimeError, match="exited"):
        template_backend(fake_program, command="exit 3").compute(triatomic, ctx)
    with pytest.raises(ValueError):
        TemplateBackend(template="x", command="true")  # no way to read the energy


def test_template_custom_parser(tmp_path, triatomic, fake_program):
    be = template_backend(fake_program, energy_regex=None,
                          parser="test_backends:parse_fake")
    res = be.compute(triatomic, JobContext(workdir=tmp_path / "p"))
    assert res.energies.size == 1


def parse_fake(text, workdir):
    for line in text.splitlines():
        if "Final energy" in line:
            return float(line.split()[-1])


def test_template_file_and_scratch_env(tmp_path, triatomic):
    tpl = tmp_path / "in.tpl"
    tpl.write_text("{{natoms}}\n")
    be = TemplateBackend(template_file=str(tpl), scratch_env="MYSCR",
                         command='echo "E= $(cat {input}) $MYSCR" > {output}',
                         energy_regex=r"E= (\S+)")
    ctx = JobContext(workdir=tmp_path / "s")
    assert be.compute(triatomic, ctx).energy == 3.0
    assert "/s/scratch" in (tmp_path / "s" / "job.out").read_text()


# -- Q-Chem ---------------------------------------------------------------

def test_qchem_input_generation(tmp_path, water):
    be = QChemBackend(rem={"method": "b3lyp", "basis": "6-31g*", "cis_n_roots": 5,
                           "cis_singlets": True},
                      extra_sections="$solvent\nsolventname water\n$end")
    text = be.make_input(water, JobContext(workdir=tmp_path, nthreads=4))
    assert text.startswith("$molecule\n0 1\nO ")
    assert "$rem" in text and "cis_singlets" in text and " true" in text
    assert "$solvent" in text
    assert be.command.format(input="job.in", output="job.out", nthreads=4) == \
        "qchem -nt 4 job.in job.out"
    assert be.scratch_env == "QCSCRATCH"


def test_qchem_parser_not_built_in_yet(tmp_path):
    be = QChemBackend(rem={"method": "hf"})
    with pytest.raises(NotImplementedError):
        be.parse_output("Total energy in the final basis set = -76.0", JobContext(tmp_path))


def test_qchem_regex_override(tmp_path):
    be = QChemBackend(rem={"method": "hf"},
                      energy_regex=r"Total energy in the final basis set =\s+(\S+)")
    out = "junk\n Total energy in the final basis set =      -76.0234567891\n"
    assert be.parse_output(out, JobContext(tmp_path)).energy == pytest.approx(-76.0234567891)


def test_qchem_needs_rem_or_template():
    with pytest.raises(ValueError):
        QChemBackend()


# -- Python functions -----------------------------------------------------

def test_function_backend_from_file(tmp_path, triatomic):
    f = tmp_path / "pot.py"
    f.write_text("def e(symbols, coords, scale=1.0):\n    return scale * float((coords**2).sum())\n")
    be = FunctionBackend(function=f"{f}:e", kwargs={"scale": 2.0})
    res = be.compute(triatomic, JobContext(tmp_path))
    assert res.energy == pytest.approx(2 * (triatomic.coords ** 2).sum())


def test_function_backend_units(tmp_path, triatomic):
    be = FunctionBackend(function="models:morse", energy_units="ev")
    res = be.compute(triatomic, JobContext(tmp_path))
    assert res.energy == pytest.approx(models.morse(None, triatomic.coords) / HARTREE_TO_EV)


def test_function_backend_bad_spec():
    with pytest.raises((ImportError, AttributeError, ValueError)):
        FunctionBackend(function="models:does_not_exist")
    with pytest.raises(ValueError):
        FunctionBackend(function="no_colon")


def test_registry():
    assert isinstance(make_backend("function", function="models:morse"), FunctionBackend)
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("gaussian16")
