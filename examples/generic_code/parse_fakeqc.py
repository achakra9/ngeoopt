"""Custom output parser for fakeqc.

A regex can only return energies. When the program also prints state
multiplicities and oscillator strengths, a small parser lets ngeoopt use them
for root following (method: character).

ngeoopt calls  parse(output_text, workdir)  after every single point.
"""

import re

from ngeoopt import EnergyResult, StateInfo


def parse(text, workdir):
    e0 = float(re.search(r"FINAL ENERGY:\s+(\S+)", text).group(1))
    # the SCF reference is flagged so an excited-state target never collapses onto it
    states = [StateInfo(energy=e0, multiplicity=1, label="S0", extra={"scf_reference": True})]
    pattern = r"State\s+(\d+):\s+E =\s+(\S+) Eh.*mult =\s+(\d+)\s+f =\s+(\S+)"
    for n, e, mult, f in re.findall(pattern, text):
        states.append(StateInfo(energy=float(e), multiplicity=int(mult),
                                oscillator_strength=float(f), label=f"S{n}"))
    for i, s in enumerate(states):
        s.index = i
    return EnergyResult(states=states)
