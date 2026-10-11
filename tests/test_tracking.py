"""Root following must follow the *state*, not the root number."""

import numpy as np
import pytest

import models
from ngeoopt import (FDSettings, FiniteDifferenceGradient, FunctionBackend, Molecule,
                     OptimizerSettings, StateTracker, TrackingSettings)
from ngeoopt.optimizer import ConvergenceCriteria, Optimizer
from ngeoopt.parallel import SerialExecutor
from ngeoopt.results import EnergyResult, StateInfo
from ngeoopt.tracking import signature_overlap


def diatomic(r):
    return Molecule(["X", "X"], np.array([[0, 0, 0], [0, 0, r]], float))


def test_signature_overlap():
    a = np.diag([1.0, 0.0])
    assert signature_overlap(a, a, None) == pytest.approx(1.0)
    assert signature_overlap(a, -a, None) == pytest.approx(1.0)  # phase-free
    assert signature_overlap(a, np.diag([0.0, 1.0]), None) == pytest.approx(0.0)
    s = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert signature_overlap(a, 3 * a, s) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["overlap", "character"])
def test_follows_state_through_reordering(method):
    """At r=2.0 state A is root 1; at r=1.9 it is root 0. The tracker must pick A."""
    fn = models.crossing_states if method != "character" else models.crossing_states_spin
    tr = StateTracker(TrackingSettings(root=1, method=method, energy_scale=1.0))
    before = fn(None, diatomic(2.0).coords)
    assert before.states[1].label == "A"
    tr.initialize(before)
    after = fn(None, diatomic(1.9).coords)
    assert after.states[0].label == "A"           # ordering flipped
    m = tr.advance(after)
    assert m.state.label == "A" and m.swapped


def test_energy_mode_is_adiabatic_at_true_crossing():
    """Energy-only matching cannot see through an exact crossing: it stays on
    the closest energy (documented limitation; overlap/character do not)."""
    tr = StateTracker(TrackingSettings(root=1, method="energy"))
    tr.initialize(models.crossing_states(None, diatomic(1.96).coords))
    assert tr.reference.label == "A"
    assert tr.select(models.crossing_states(None, diatomic(1.94).coords)).state.label == "B"


def test_index_mode_does_not_track():
    tr = StateTracker(TrackingSettings(root=1, method="index"))
    tr.initialize(models.crossing_states(None, diatomic(2.0).coords))
    m = tr.select(models.crossing_states(None, diatomic(1.9).coords))
    assert m.state.label == "B"   # stays on root 1, which is now the other state


def test_auto_picks_overlap_when_signatures_exist():
    tr = StateTracker(TrackingSettings(root=1))
    tr.initialize(models.crossing_states(None, diatomic(2.0).coords))
    assert tr.select(models.crossing_states(None, diatomic(1.9).coords)).method == "overlap"


def test_multiplicity_filter_counts_within_manifold():
    res = EnergyResult(states=[
        StateInfo(-1.0, 0, 1, extra={"scf_reference": True}),
        StateInfo(-0.9, 1, 3), StateInfo(-0.8, 2, 1), StateInfo(-0.7, 3, 3)])
    tr = StateTracker(TrackingSettings(root=1, multiplicity=3))
    assert tr.initialize(res).energy == -0.7


def test_excited_target_never_collapses_to_reference():
    ref = StateInfo(-1.0, 0, 1, extra={"scf_reference": True})
    s1 = StateInfo(-0.6, 1, 1, oscillator_strength=0.1)
    tr = StateTracker(TrackingSettings(root=1, method="energy", energy_scale=1.0))
    tr.initialize(EnergyResult(states=[ref, s1]))
    # excited state drops close to the reference energy: still must not pick S0
    m = tr.select(EnergyResult(states=[StateInfo(-1.0, 0, 1, extra={"scf_reference": True}),
                                       StateInfo(-0.98, 1, 1, oscillator_strength=0.1)]))
    assert m.state.index == 1


def test_tracker_state_roundtrip():
    tr = StateTracker(TrackingSettings(root=1))
    tr.initialize(models.crossing_states(None, diatomic(2.0).coords))
    d = tr.state_dict()
    tr2 = StateTracker(TrackingSettings(root=1))
    tr2.load_state_dict(d)
    assert tr2.reference.energy == tr.reference.energy
    np.testing.assert_array_equal(tr2.reference.signature, tr.reference.signature)


def test_fd_gradient_at_crossing_uses_one_state(tmp_path):
    """Displacements straddle the crossing; every one must use state A."""
    r = 1.949
    be = FunctionBackend(function="models:crossing_states")
    fd = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path),
                                  StateTracker(TrackingSettings(root=0)),
                                  FDSettings(step=0.01, project_rigid=False))
    # at r=1.949 A is the lower root (crossing at 1.95); displacements of
    # +-0.01 along z move to both sides of the crossing
    tr = fd.tracker
    tr.initialize(models.crossing_states(None, diatomic(r).coords))
    assert tr.reference.label == "A"
    gr = fd.compute(diatomic(r))
    assert gr.state.label == "A"
    assert gr.tracking["swaps"] > 0
    # dE_A/dr = (r - 1.5); atom 1 moves along +z
    assert gr.gradient[1, 2] == pytest.approx(r - 1.5, abs=1e-8)


def test_optimizer_follows_state_through_crossing(tmp_path):
    """Start on the upper state A at r=2.4; it becomes the lower root after
    r=1.95. Overlap tracking must land in A's minimum (r=1.5), not B's (2.5)."""
    be = FunctionBackend(function="models:crossing_states")
    fd = FiniteDifferenceGradient(be, SerialExecutor(scratch=tmp_path / "jobs"),
                                  StateTracker(TrackingSettings(root=1)))
    opt = Optimizer(fd, OptimizerSettings(hessian_init="unit", hessian_unit=1.0,
                                          convergence=ConvergenceCriteria.preset("tight")),
                    workdir=tmp_path)
    res = opt.run(diatomic(2.4), restart=False)
    r = np.linalg.norm(res.molecule.coords[0] - res.molecule.coords[1])
    assert res.converged
    assert r == pytest.approx(1.5, abs=1e-4)
    roots = [h.root for h in res.history]
    assert roots[0] == 1 and roots[-1] == 0   # the root number changed, the state did not
