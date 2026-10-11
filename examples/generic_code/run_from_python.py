"""The same three calculations through the Python API instead of YAML.

    python run_from_python.py
"""

import sys
from pathlib import Path

from ngeoopt import (FDSettings, Molecule, OptimizerSettings, TemplateBackend,
                     TrackingSettings, gradient, optimize)

HERE = Path(__file__).resolve().parent


def backend(**kw):
    opts = dict(
        template_file=str(HERE / "job.tpl"),
        variables={"method": "b3lyp", "basis": "def2-svp"},
        copy_files=[str(HERE / "fakeqc.py")],
        command=f"{sys.executable} fakeqc.py {{input}} > {{output}}",
        energy_regex=r"FINAL ENERGY:\s+(\S+)",
        success_regex="terminated normally",
    )
    opts.update(kw)
    return TemplateBackend(**opts)


def main():
    mol = Molecule.from_xyz(HERE / "water.xyz")

    # 1. numerical gradient
    g = gradient(mol, backend(), workers=4, settings=FDSettings(scheme="central", step=0.005))
    print(f"E = {g.energy:.10f}  ({g.njobs} single points, {g.wall_time:.1f} s)")
    print("gradient (Eh/Bohr):\n", g.gradient)

    # 2. ground-state optimization
    res = optimize(mol, backend(), workers=4, workdir="py_opt", restart=False)
    print(f"ground state: E = {res.energy:.10f}, converged = {res.converged}")
    print(res.molecule.to_xyz())

    # 3. excited-state optimization of the bright state, followed by character
    exc = backend(energy_regex=None, parser=f"{HERE / 'parse_fakeqc.py'}:parse")
    res = optimize(mol, exc, workers=4, workdir="py_excited", restart=False,
                   state=TrackingSettings(root=2, multiplicity=1, method="character",
                                          osc_scale=0.02),
                   optimizer=OptimizerSettings(max_cycles=50))
    print(f"bright state: E = {res.energy:.10f}, converged = {res.converged}, "
          f"roots visited = {sorted({h.root for h in res.history})}")


if __name__ == "__main__":      # required: workers are spawned processes
    main()
