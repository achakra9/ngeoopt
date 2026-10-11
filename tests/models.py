"""Cheap analytic model potentials for the tests (energies in Hartree, Bohr)."""

import numpy as np

from ngeoopt.results import EnergyResult, StateInfo


def morse(symbols, coords, De=0.17, a=1.0, re=1.8):
    """Pairwise Morse potential; minimum: all bonded pairs at re."""
    x = np.asarray(coords).reshape(-1, 3)
    e = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            r = np.linalg.norm(x[i] - x[j])
            e += De * (1 - np.exp(-a * (r - re))) ** 2 - De
    return e


def morse_gradient(coords, De=0.17, a=1.0, re=1.8):
    x = np.asarray(coords).reshape(-1, 3)
    g = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = x[i] - x[j]
            r = np.linalg.norm(d)
            ex = np.exp(-a * (r - re))
            dEdr = 2 * De * a * ex * (1 - ex)
            g[i] += dEdr * d / r
            g[j] -= dEdr * d / r
    return g


def lj(symbols, coords, eps=0.01, sigma=3.0):
    x = np.asarray(coords).reshape(-1, 3)
    e = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            r = np.linalg.norm(x[i] - x[j])
            e += 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return e


def crossing_states(symbols, coords, shift=0.05):
    """Two diabatic states of a diatomic that cross at r = 1.95 bohr.

    State A: 0.5 (r - 1.5)^2           signature diag(1, 0)
    State B: 0.5 (r - 2.5)^2 - shift   signature diag(0, 1)
    Returned sorted by energy, so the root ordering flips at the crossing.
    """
    x = np.asarray(coords).reshape(-1, 3)
    r = np.linalg.norm(x[0] - x[1])
    a = StateInfo(0.5 * (r - 1.5) ** 2, multiplicity=1, label="A",
                  signature=np.diag([1.0, 0.0]))
    b = StateInfo(0.5 * (r - 2.5) ** 2 - shift, multiplicity=1, label="B",
                  signature=np.diag([0.0, 1.0]))
    states = sorted([a, b], key=lambda s: s.energy)
    for i, s in enumerate(states):
        s.index = i
    return EnergyResult(states=states)


def crossing_states_spin(symbols, coords):
    """Same crossing but the states differ in spin and carry no signature."""
    res = crossing_states(symbols, coords)
    for s in res.states:
        s.multiplicity = 1 if s.label == "A" else 3
        s.signature = None
    return res


def device_echo(symbols, coords, device="cpu"):
    """Harmonic energy; records the device it was called with."""
    x = np.asarray(coords)
    return {"states": [StateInfo(float((x ** 2).sum()), extra={"device": device})]}


def torch_harmonic(symbols, coords, device="cpu"):
    import torch
    t = torch.tensor(np.asarray(coords), dtype=torch.float32, device=device)
    return (0.5 * (t ** 2).sum()).cpu()


def flaky(symbols, coords):
    raise RuntimeError("this backend always fails")


def unconverged_states(symbols, coords):
    """Morse ground state plus an 'excited' state flagged as not converged."""
    e = morse(symbols, coords)
    return EnergyResult(states=[StateInfo(e, 0, 1, extra={"scf_reference": True}),
                                StateInfo(e + 0.2, 1, 1, extra={"converged": False})])


def coupled_pair(symbols, coords, gap=0.002, kappa=0.4):
    """Two states split by a symmetry-breaking coordinate x (atom 1, x):
    E = -/+ sqrt(gap^2/4 + (kappa x)^2): a sharp pseudo-Jahn-Teller saddle."""
    x = np.asarray(coords).reshape(-1, 3)
    q = x[1, 0]
    d = np.sqrt((gap / 2) ** 2 + (kappa * q) ** 2)
    states = [StateInfo(-d, 0, 1), StateInfo(d, 1, 1)]
    return EnergyResult(states=states)


def misleading_signatures(symbols, coords):
    """Two well separated states whose signatures are exchanged for x > 0
    (like a singles-only signature of states that differ in their doubles):
    overlap-first matching then jumps to the other state on one side."""
    x = np.asarray(coords).reshape(-1, 3)
    q = x[1, 0]
    a, b = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    if q > 0:
        a, b = b, a
    return EnergyResult(states=[StateInfo(-1.0, 0, 1, signature=a),
                                StateInfo(-0.95, 1, 1, signature=b)])
