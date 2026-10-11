#!/usr/bin/env python3
"""fakeqc -- a stand-in "quantum chemistry program" for the generic example.

It behaves like a real code from ngeoopt's point of view: it reads an input
file, writes an output file in its own format, and exits. Replace it with your
real program (Gaussian, ORCA, Molpro, xtb, an in-house code, ...).

Physics: a simple water model (two O-H stretches + H-O-H bend), plus two
excited states with different equilibrium geometries that cross along the
bend, so the root ordering changes during an excited-state optimization.

Usage:  python3 fakeqc.py job.inp > job.out
"""

import math
import sys

ANG2BOHR = 1.0 / 0.529177210903
EV = 27.211386245988


def read_input(path):
    keys, atoms, in_geom = {}, [], False
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.lower() == "geometry":
            in_geom = True
        elif line.lower() == "end":
            in_geom = False
        elif in_geom:
            s, x, y, z = line.split()[:4]
            atoms.append((s, [float(x) * ANG2BOHR, float(y) * ANG2BOHR, float(z) * ANG2BOHR]))
        else:
            k, *v = line.split()
            keys[k.lower()] = " ".join(v)
    return keys, atoms


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def angle(a, o, b):
    u = [p - q for p, q in zip(a, o)]
    v = [p - q for p, q in zip(b, o)]
    c = sum(p * q for p, q in zip(u, v)) / (dist(a, o) * dist(b, o))
    return math.acos(max(-1.0, min(1.0, c)))


def surface(r1, r2, th, r0, th0, kr, kt, e0):
    return e0 + kr * ((r1 - r0) ** 2 + (r2 - r0) ** 2) + kt * (th - th0) ** 2


def main():
    keys, atoms = read_input(sys.argv[1])
    o = next(c for s, c in atoms if s.upper() == "O")
    hs = [c for s, c in atoms if s.upper() == "H"]
    r1, r2 = dist(hs[0], o), dist(hs[1], o)
    th = angle(hs[0], o, hs[1])

    # ground state: r0 = 0.958 A, theta0 = 104.5 deg
    e_gs = surface(r1, r2, th, 1.81, math.radians(104.5), 0.25, 0.08, -76.40)
    # two excited states (eV above the ground minimum), different preferred angles
    e_a = surface(r1, r2, th, 1.90, math.radians(140.0), 0.20, 0.10, -76.40 + 7.0 / EV)
    e_b = surface(r1, r2, th, 1.86, math.radians(95.0), 0.22, 0.12, -76.40 + 7.6 / EV)
    states = sorted([(e_a, 1, 0.052), (e_b, 1, 0.003)])

    print(" fakeqc 1.0  --  not a real quantum chemistry program")
    print(f" method {keys.get('method', '?')}   basis {keys.get('basis', '?')}"
          f"   threads {keys.get('nthreads', '1')}")
    print(" SCF converged.")
    print(f" FINAL ENERGY:  {e_gs: .12f} Eh")
    print(" Excited states")
    for i, (e, mult, f) in enumerate(states, 1):
        print(f"   State {i}:  E = {e: .12f} Eh  dE = {(e - e_gs) * EV:8.4f} eV"
              f"  mult = {mult}  f = {f:.4f}")
    print(" fakeqc terminated normally")


if __name__ == "__main__":
    main()
