# ngeoopt

ngeoopt optimizes molecular geometries using finite-difference gradients, so it works with any electronic-structure method that can give you an energy, analytic gradients or not. All the displaced single points for a gradient (2×3N for central differences) run in parallel, and on NVIDIA machines each worker gets its own GPU.

- **Backends:** PySCF, exkit (SF/MRSF/SF-CISd and every other exkit method), Q-Chem (SCF, TDDFT/TDA/CIS, SF-TDDFT, MRSF-TDDFT parsed), any external program through an input template, and any Python function.
- **Excited states:** root following that never trusts the root number alone. Each candidate is scored on wavefunction/amplitude overlap, energy (against the optimizer's prediction), ⟨S²⟩, oscillator strength, transition dipole, and root number as a tie-breaker. Guards cover spin-contaminated SF roots, and a continuity check flags stencils that mix two states.
- **Tricky SCFs:** displaced SCFs start from the centre-geometry density, so broken-symmetry and open-shell references stay on one branch.
- **Parallel:** local process pool or MPI. Threads per worker, GPU per worker.
- **GPUs:** gpu4pyscf on NVIDIA. On Apple silicon, the MPS device is passed to torch-based energy functions.
- **Restartable:** the optimizer state is checkpointed every cycle, and each single point is cached as soon as it finishes, so a killed job resumes mid-gradient.
- **Configurable:** one YAML file with typo-checked keywords for the stencil, step size, trust radius, Hessian, convergence presets, tracking thresholds and more.

## Install

```bash
pip install .                  # core (numpy, pyyaml, typer)
pip install ".[pyscf]"         # + PySCF
pip install ".[cuda12]"        # + PySCF, gpu4pyscf and cupy for CUDA 12
pip install ".[mps]"           # + torch, for MPS energy functions on Apple silicon
pip install ".[mpi]"           # + mpi4py for multi-node runs
pip install -e /path/to/exkit  # exkit backend (exkit is found if importable)
pip install -e ".[all]"        # development install with tests
```

`ngeoopt devices` shows which GPUs and optional packages it can see.

## Quick start

```bash
ngeoopt init input.yaml        # fully commented template with every keyword
ngeoopt run input.yaml
ngeoopt status ngeoopt_run     # cycle table from the checkpoint (works while running)
```

A minimal input:

```yaml
molecule:
  xyz: water.xyz
backend:
  name: pyscf
  method: b3lyp
  basis: 6-31g*
```

An excited-state (S1) optimization, following the state by overlap:

```yaml
backend:
  name: pyscf
  method: b3lyp
  basis: 6-31g*
  excited: tda
  nroots: 5
state:
  root: 1            # 0 = ground state, 1 = first excited state of `multiplicity`
  multiplicity: 1
  method: overlap
```

Any program, e.g. xtb:

```yaml
backend:
  name: template
  template: |
    {{natoms}}

    {{geometry}}
  input_name: job.xyz
  command: "xtb {input} --sp > {output}"
  energy_regex: 'TOTAL ENERGY\s+(\S+)\s+Eh'
```

From Python:

```python
from ngeoopt import Molecule, optimize, TrackingSettings
from ngeoopt.backends.pyscf_backend import PySCFBackend

if __name__ == "__main__":          # required: workers are spawned processes
    mol = Molecule.from_xyz("water.xyz")
    res = optimize(mol, PySCFBackend(method="b3lyp", basis="6-31g*"), workers=8)
    print(res.energy, res.converged)
```

For your own program, start with [examples/generic_code](examples/generic_code). It is a runnable walkthrough (numerical gradient, optimization, and excited-state root following through a reordering) with a fill-in template. More inputs are in [examples/](examples/). See the [manual](docs/MANUAL.md) for every keyword, how root following works, GPU/MPI setup and how to add a backend.

## Status

| Backend | Energies | Excited states | Overlap tracking | Tested |
|---|---|---|---|---|
| PySCF | HF, DFT, MP2, CCSD, CCSD(T), CASSCF/CASCI | TDDFT, TDA/CIS, TDHF, EOM-EE-CCSD | TDDFT/TDA/CIS | yes, against analytic gradients |
| exkit | exkit methods with geometry support (MP2, double hybrids, CASSCF, …) | CIS/TDA, SF, MRSF, MD-SF, SF-CISd, pp/hh-TDA, DEA/DIP, IP/EA | AO excitation densities (not SF-CISd, MD-SF, CAS) | 20 methods vs exkit.run; CIS gradient vs PySCF analytic |
| Q-Chem | SCF/DFT (MP2/CC via regex for now) | TDDFT/TDA/CIS, SF-TDDFT, MRSF-TDDFT | squared amplitudes (+ state-DM populations) | parser on real outputs; jobs not run here |
| template | anything with a regex or parser | via `states_regex` | — | with a mock program |
| function | anything in Python (numpy, torch, …) | return a list of states | if you return signatures | yes |

Q-Chem MP2/CC/EOM output parsing will be added once reference outputs are available.

## Tests

```bash
pip install -e ".[test,pyscf]"
pytest
```

PySCF tests are skipped if PySCF is not installed, and the MPS test is skipped off Apple silicon.
