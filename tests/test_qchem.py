"""Q-Chem output parsing and SF/MRSF root following, on real butadiene outputs
(BHHLYP/cc-pVDZ, Q-Chem 7.0 devel) in tests/data/qchem."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ngeoopt import JobContext, Molecule, QChemBackend, StateTracker, TrackingSettings
from ngeoopt.backends.qchem_parser import (QChemParseError, davidson_unconverged,
                                           multiplicity_from_s2, parse_qchem_output,
                                           state_dm_populations)
from ngeoopt.results import EnergyResult, StateInfo
from ngeoopt.tracking import is_contaminated, signature_overlap

DATA = Path(__file__).parent / "data" / "qchem"


def out(name):
    return (DATA / f"butadiene-{name}.out").read_text()


def by_label(res):
    return {s.label: s for s in res.states}


# -- parsing ---------------------------------------------------------------

def test_scf_energy_uses_ten_decimals():
    res = parse_qchem_output(out("scf"))
    assert len(res.states) == 1 and res.states[0].extra["scf_reference"]
    assert res.energy == -155.8948006150      # iteration line, not "Total energy = ...061"


def test_tddft_states():
    res = parse_qchem_output(out("tddft"))
    assert res.extra["qchem_block"] == "TDDFT/TDA Excitation Energies"
    assert not res.extra["spin_flip"]
    assert len(res.states) == 7
    s0, s1 = res.states[0], res.states[1]
    assert s0.extra["scf_reference"] and s0.energy == -155.8948006150
    assert s1.energy == -155.65107727 and s1.multiplicity == 1
    assert s1.oscillator_strength == pytest.approx(1.09838305)
    assert s1.extra["trans_mom"] == (-2.5701, 0.3930, -0.0)
    # phase-free amplitude descriptor: D(15)->V(1), weight 0.9752^2 (normalised)
    assert s1.signature == {"amp:D15>V1": pytest.approx(1.0)}
    assert all(s.extra["converged"] for s in res.states)


def test_sf_tddft_states_and_spin_contamination():
    res = parse_qchem_output(out("sftddft"))
    assert res.extra["spin_flip"]
    # no high-spin reference among the states; SF state 1 is the ground state
    assert not any(s.extra.get("scf_reference") for s in res.states)
    st = by_label(res)
    assert st["SF1"].energy == -155.88779903 and st["SF1"].index == 0
    assert st["SF1"].multiplicity == 1       # <S^2> = 0.0798
    assert st["SF2"].multiplicity == 3       # <S^2> = 2.0179
    assert st["SF3"].multiplicity is None    # 0.9528: contaminated, not a "doublet"
    assert st["SF5"].multiplicity is None    # 1.0629
    assert st["SF4"].multiplicity == 1       # 0.0977
    assert st["SF3"].extra["s2"] == 0.9528
    assert st["SF1"].oscillator_strength is None
    assert "amp:S2>S1a" in st["SF1"].signature
    assert res.extra["scf_energy"] == -155.7762882828


def test_mrsf_states():
    res = parse_qchem_output(out("mrsf"))
    assert res.extra["spin_flip"] and res.extra["qchem_block"].startswith("MRSF")
    assert [s.multiplicity for s in res.states] == [1] * 6
    st = by_label(res)
    assert st["SF2"].oscillator_strength == pytest.approx(1.0936596138)
    assert np.all(np.diff(res.energies) > 0)


def test_mrsf_state_analysis_descriptors_without_tdm():
    text = out("mrsf-analysis")
    dm = state_dm_populations(text)
    assert sorted(dm) == list(range(2, 21))          # state 1 is the reference
    assert dm[2]["ha1"] == pytest.approx(0.185803)
    assert dm[2]["eb1"] == pytest.approx(0.190831)   # |e-| population
    res = parse_qchem_output(text)
    sig = by_label(res)["SF2"].signature
    assert any(k.startswith("dm:") for k in sig) and any(k.startswith("amp:") for k in sig)
    # nothing from transition densities / Omega / NTOs
    assert not any("tdm" in k or "omega" in k for k in sig)
    assert len(res.states) == 20


def test_davidson_unconverged_roots_are_flagged():
    text = out("tddft").replace(
        "  12         6           0      0.000000    0.000000    Roots Converged",
        "  12         4           2      0.000100    0.000050")
    assert davidson_unconverged(text) == 2
    res = parse_qchem_output(text)
    flags = [s.extra["converged"] for s in res.states[1:]]
    assert flags == [True, True, True, True, False, False]   # highest two roots
    assert not res.converged


def test_abnormal_termination_raises():
    text = out("scf").replace("Thank you very much for using Q-Chem", "segfault")
    with pytest.raises(QChemParseError):
        parse_qchem_output(text)


def test_multiplicity_from_s2_parity():
    assert [multiplicity_from_s2(v, 30) for v in (0.0, 0.08, 0.95, 2.02, 6.0)] == \
        [1, 1, None, 3, 5]
    assert [multiplicity_from_s2(v, 31) for v in (0.75, 0.8, 1.3, 3.75)] == [2, 2, None, 4]


# -- root following on SF/MRSF -----------------------------------------------

def test_sf_root_counting_skips_contaminated_states():
    res = parse_qchem_output(out("sftddft"))
    # "first excited singlet": SF3 (<S^2>=0.95) must not count as a singlet
    tr = StateTracker(TrackingSettings(root=1, multiplicity=1))
    assert tr.initialize(res).label == "SF4"
    assert StateTracker(TrackingSettings(root=0, multiplicity=1)).initialize(res).label == "SF1"
    assert StateTracker(TrackingSettings(root=0, multiplicity=3)).initialize(res).label == "SF2"


def test_sf_singlet_not_swapped_for_triplet_with_same_amplitudes():
    """SF2 (triplet) and SF4 (singlet) are the +/- combinations of the same two
    configurations, so their squared amplitudes are almost identical. Only
    <S^2> tells them apart."""
    res = parse_qchem_output(out("sftddft"))
    st = by_label(res)
    assert signature_overlap(st["SF2"].signature, st["SF4"].signature, None) > 0.99
    tr = StateTracker(TrackingSettings(root=1, multiplicity=1, method="overlap"))
    tr.initialize(res)                                   # target SF4
    # at the "next geometry" the triplet sits exactly where the singlet was expected
    m = tr.select(res, expected_energy=st["SF2"].energy)
    assert m.state.label == "SF4"


def test_character_tracking_penalises_s2_change():
    ref = StateInfo(-1.00, 1, 1, extra={"s2": 0.10})
    near_but_contaminated = StateInfo(-1.001, 1, None, extra={"s2": 0.55})
    clean = StateInfo(-0.995, 2, 1, extra={"s2": 0.12})
    tr = StateTracker(TrackingSettings(method="character", energy_scale=0.01))
    tr.set_reference(ref)
    m = tr.select(EnergyResult(states=[near_but_contaminated, clean]))
    assert m.state is clean


def test_s2_jump_filter():
    ref = StateInfo(-1.0, 1, 1, extra={"s2": 0.05})
    jumped = StateInfo(-1.0, 1, None, extra={"s2": 0.95})
    other = StateInfo(-0.9, 2, 1, extra={"s2": 0.06})
    tr = StateTracker(TrackingSettings(method="energy"))
    tr.set_reference(ref)
    assert tr.select(EnergyResult(states=[jumped, other])).state is other


def test_contaminated_target_warns(caplog):
    res = parse_qchem_output(out("sftddft"))
    tr = StateTracker(TrackingSettings(root=2))        # SF3, <S^2> = 0.95
    tr.initialize(res)
    tr.advance(res)
    assert is_contaminated(by_label(res)["SF3"], 0.25)
    assert "spin-contaminated" in caplog.text


def test_tracker_checkpoint_keeps_dict_signature_and_s2():
    res = parse_qchem_output(out("mrsf-analysis"))
    tr = StateTracker(TrackingSettings(root=1, multiplicity=1))
    tr.initialize(res)
    d = tr.state_dict()
    tr2 = StateTracker(TrackingSettings(root=1, multiplicity=1))
    tr2.load_state_dict(d)
    assert tr2.reference.signature == tr.reference.signature
    assert tr2.reference.extra["s2"] == 0.0


# -- backend end to end (canned output instead of running Q-Chem) ----------

def butadiene():
    text = (DATA / "butadiene-sftddft.in").read_text()
    geom = text.split("$molecule")[1].split("$end")[0].strip().splitlines()[1:]
    return Molecule.from_lines("\n".join(geom), multiplicity=3)


def test_backend_with_canned_output(tmp_path, caplog):
    be = QChemBackend(rem={"method": "bhhlyp", "basis": "cc-pvdz", "unrestricted": True,
                           "spin_flip": 1, "cis_n_roots": 6, "scf_convergence": 8},
                      command="cp canned.out {output}")     # stands in for qchem
    assert "SCF_CONVERGENCE" in caplog.text and "XC_GRID" in caplog.text
    ctx = JobContext(workdir=tmp_path / "job", label="job")
    ctx.workdir.mkdir()
    shutil.copy(DATA / "butadiene-sftddft.out", ctx.workdir / "canned.out")
    res = be.compute(butadiene(), ctx)
    assert res.extra["spin_flip"] and len(res.states) == 6
    inp = (ctx.workdir / "job.in").read_text()
    assert "$molecule\n0 3\nC" in inp and "sym_ignore" in inp and "spin_flip" in inp


def test_backend_template_file(tmp_path):
    tpl = tmp_path / "qc.tpl"
    tpl.write_text("$molecule\n{{charge}} {{multiplicity}}\n{{geometry}}\n$end\n"
                   "$rem\nmethod bhhlyp\n$end\n")
    be = QChemBackend(template_file=str(tpl), command="true")
    text = be.make_input(butadiene(), JobContext(tmp_path))
    assert text.startswith("$molecule\n0 3\nC ")


def test_driver_qchem_defaults(tmp_path):
    """Q-Chem run through the YAML driver: step defaults to 0.01 Bohr, tight
    criteria warn, and the parser picks the tracked SF state."""
    from ngeoopt import run
    shutil.copy(DATA / "butadiene-sftddft.out", tmp_path / "canned.out")
    geom = (DATA / "butadiene-sftddft.in").read_text().split("$molecule")[1].split("$end")[0]
    cfg = {"task": "energy",
           "molecule": {"geometry": "\n".join(geom.strip().splitlines()[1:]), "multiplicity": 3},
           "backend": {"name": "qchem", "command": "cp canned.out {output}",
                       "copy_files": [str(tmp_path / "canned.out")],
                       "rem": {"method": "bhhlyp", "basis": "cc-pvdz", "spin_flip": 1,
                               "unrestricted": True, "cis_n_roots": 6,
                               "scf_convergence": 10, "cis_convergence": 8, "xc_grid": 3}},
           "state": {"root": 1, "multiplicity": 1},
           "optimizer": {"convergence": "tight"},
           "parallel": {"executor": "serial"},
           "output": {"workdir": str(tmp_path / "run")}}
    res = run(cfg)
    assert res.energy == -155.65973121            # SF4: first excited *clean* singlet
    log = (tmp_path / "run" / "ngeoopt.log").read_text()
    assert "step set to 0.01 Bohr" in log and "tight/verytight" in log
