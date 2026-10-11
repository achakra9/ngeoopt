# ngeoopt manual

## Contents

1. [How it works](#1-how-it-works)
2. [Installation](#2-installation)
3. [Running](#3-running)
4. [Input reference](#4-input-reference)
5. [Root following (state tracking)](#5-root-following-state-tracking)
6. [Parallel execution and GPUs](#6-parallel-execution-and-gpus)
7. [Checkpoints, restart and output files](#7-checkpoints-restart-and-output-files)
8. [Backends](#8-backends)
9. [Python API](#9-python-api)
10. [Accuracy and performance tips](#10-accuracy-and-performance-tips)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. How it works

```
          ┌──────────── optimizer cycle ─────────────┐
geometry ─► build 1 + k·3N displaced geometries      │
          │   (k = 1 forward, 2 central, 4 central4) │
          ├► run all single points in parallel ───────┤◄── cache (skip finished ones)
          ├► pick the target state in every result    │◄── state tracker
          ├► combine energies into the gradient       │
          ├► update Hessian, take an RFO step         │
          └► write checkpoint + trajectory ───────────┘
```

* Coordinates are Cartesian, in Bohr internally; energies are in Hartree.
* The backend only computes energies. Displacements, parallel dispatch, root following and the optimizer are all handled by ngeoopt.
* Translations and rotations are projected out of the gradient and the step, unless atoms are frozen.

## 2. Installation

```bash
pip install .              # core
pip install ".[pyscf]"     # PySCF backend
pip install ".[cuda12]"    # NVIDIA, CUDA 12: pyscf + gpu4pyscf-cuda12x + cupy-cuda12x
pip install ".[cuda11]"    # NVIDIA, CUDA 11
pip install ".[mps]"       # torch, for MPS-based energy functions on Apple silicon
pip install ".[mpi]"       # mpi4py
pip install -e ".[all]"    # everything plus pytest, editable
```

Python ≥ 3.9. Check what was detected:

```
$ ngeoopt devices
auto-detected : cuda (2 CUDA device(s))
cupy          : yes
gpu4pyscf     : yes
...
```

## 3. Running

| Command | Purpose |
|---|---|
| `ngeoopt init [file]` | write a fully commented template input |
| `ngeoopt run input.yaml` | run the task in the file |
| `ngeoopt run input.yaml --task gradient` | override the task |
| `ngeoopt run input.yaml --workers 16 --device cuda` | override parallel settings |
| `ngeoopt run input.yaml --no-restart` | ignore the existing checkpoint |
| `ngeoopt status [workdir]` | print the cycle table from the checkpoint |
| `ngeoopt devices` | show detected accelerators and optional packages |

Exit code is 0 on success and 2 if an optimization stopped without converging.

Tasks: `energy` (single point with tracking), `gradient` (writes `gradient.txt`), `optimize`.

## 4. Input reference

The input is YAML with the sections below. Every keyword is optional except `molecule` (`xyz` or `geometry`) and `backend.name`. Misspelled keywords are rejected, and the error lists the valid ones.

### `molecule`

| Keyword | Default | Meaning |
|---|---|---|
| `xyz` | – | XYZ file, relative to the input file's directory |
| `geometry` | – | inline `El x y z` lines instead of a file |
| `charge` | 0 | |
| `multiplicity` | 1 | 2S+1 |
| `units` | angstrom | `angstrom` or `bohr` for the coordinates given |

### `backend`

`name` picks the backend: `pyscf`, `exkit`, `qchem`, `template` (alias `external`) or `function` (alias `python`). The other keys depend on the backend (§8).

### `state` — which state to optimize, and how to follow it

| Keyword | Default | Meaning |
|---|---|---|
| `root` | 0 | target state at the first geometry. 0 = ground / lowest state. With `multiplicity` set, it counts only *clean* states of that multiplicity, so spin-contaminated roots are skipped |
| `multiplicity` | null | restrict root counting and matching to one spin manifold |
| `method` | auto | `auto`/`composite`, `overlap`, `character`, `energy`, `index` (§5) |
| `overlap_scale` | 0.1 | 1 − overlap of this size costs one unit in the score |
| `energy_scale` | 0.01 | Hartree; energy difference worth one unit, between optimization cycles |
| `stencil_energy_scale` | 0.002 | Hartree; the same inside a finite-difference stencil, where energies barely move |
| `s2_scale` | 0.2 | ⟨S²⟩ difference worth one unit |
| `osc_scale` | 0.05 | oscillator-strength difference worth one unit (plus 20% of the larger f) |
| `tdip_scale` | 0.1 | transition-dipole difference (a.u.) worth one unit (plus 20% of the larger \|μ\|) |
| `index_weight` | 0.05 | cost per position of root-number change; only breaks ties |
| `s2_max_change` | 0.5 | candidates whose ⟨S²⟩ changes by more than this are excluded |
| `s2_tolerance` | 0.25 | \|⟨S²⟩ − S(S+1)\| above this marks a state as spin-contaminated |
| `min_overlap` | 0.5 | warn when the best wavefunction overlap falls below this |
| `follow` | true | move the reference to the newly matched state after each cycle |

### `gradient`

| Keyword | Default | Meaning |
|---|---|---|
| `scheme` | central | `forward` (3N+1 jobs, O(h)), `backward`, `central` (6N+1, O(h²)), `central4` (12N+1, O(h⁴)) |
| `step` | 0.005 | displacement in Bohr (0.01 by default for the Q-Chem backend, §8) |
| `project_rigid` | true | remove translation/rotation from the gradient |
| `frozen_atoms` | [] | 0-based indices; never displaced, zero gradient, never moved |
| `use_cache` | true | reuse finished single points (restart) |
| `guess_from_center` | true | SCF of every displaced geometry starts from the centre density, and each new centre from the previous one (backends that support it: PySCF, exkit). Costs one sequential single point per gradient |
| `max_curvature` | 5.0 | Eh/Bohr²; larger second differences inside a stencil trigger the continuity warning (§5) |

### `optimizer`

| Keyword | Default | Meaning |
|---|---|---|
| `method` | rfo | `rfo` (rational function), `qn` (Newton step on \|H\|), `sd` (steepest descent) |
| `max_cycles` | 100 | |
| `hessian_init` | lindh | `lindh` model Hessian, or `unit` (scaled identity) |
| `hessian_unit` | 0.5 | Hartree/Bohr² for `unit` |
| `hessian_update` | bfgs | `bfgs`, `sr1`, `psb`, `bofill`, `none` |
| `trust_radius` | 0.3 | initial maximum step norm, Bohr |
| `trust_min` / `trust_max` | 0.001 / 1.0 | trust radius bounds |
| `reject_energy_rise` | 1e-5 | steps that raise the energy by more than this (Hartree) are rejected and retried with a smaller trust radius |
| `convergence` | gaussian | preset name, or a dict (below) |

`convergence` presets (max/rms force in Hartree/Bohr, max/rms step in Bohr, energy in Hartree):

| Preset | max F | rms F | max step | rms step | ΔE |
|---|---|---|---|---|---|
| loose | 2.5e-3 | 1.7e-3 | 1.0e-2 | 6.7e-3 | 1e-5 |
| gaussian | 4.5e-4 | 3.0e-4 | 1.8e-3 | 1.2e-3 | 1e-6 |
| qchem | 3.0e-4 | 3.0e-4 | 1.2e-3 | 1.2e-3 | 1e-6 |
| tight | 1.5e-5 | 1.0e-5 | 6.0e-5 | 4.0e-5 | 1e-8 |
| verytight | 2.0e-6 | 1.0e-6 | 6.0e-6 | 4.0e-6 | 1e-9 |

As a dict: `{preset: tight, max_force: 1e-5, mode: all}`. `mode` is one of:
- `force_and_any` (default): forces, plus either the steps or ΔE
- `all`: every criterion
- `force`: forces only

### `parallel`

| Keyword | Default | Meaning |
|---|---|---|
| `executor` | process | `serial`, `process` (local pool), `mpi` |
| `workers` | auto | number of concurrent single points; `auto` = cores ÷ threads (CPU/MPS) or GPUs × `workers_per_gpu` (CUDA) |
| `threads_per_worker` | 1 | OpenMP/BLAS threads per job (also `{nthreads}` in commands) |
| `device` | auto | `auto`, `cpu`, `cuda`, `cuda:N`, `mps` |
| `gpus` | null | restrict to these CUDA ids, e.g. `[0, 2]` |
| `workers_per_gpu` | 1 | several small jobs can share one GPU |
| `retries` | 0 | re-run a failed single point this many times |

### `output`

| Keyword | Default | Meaning |
|---|---|---|
| `workdir` | ngeoopt_run | everything goes here (relative to the input file) |
| `checkpoint` | ngeoopt_state.npz | optimizer state |
| `trajectory` | trajectory.xyz | accepted geometries, energies in the comment line |
| `final_xyz` | final.xyz | last accepted geometry |
| `keep_files` | false | keep each job's directory (inputs/outputs of external codes) |
| `restart` | true | resume from checkpoint and cache when present |
| `cache` | true | store every single point as it finishes |
| `log_file` | ngeoopt.log | |
| `log_level` | info | `debug`, `info`, `warning` |

## 5. Root following (state tracking)

Root numbers are not stable labels for states. The ordering changes along a coordinate:
- when the target crosses or approaches another state along the optimization path;
- inside one finite-difference stencil, near a crossing or a strongly coupled pair.

If root 2 were taken at every geometry, two different states could be differenced. ngeoopt never trusts the root number on its own. At every geometry, it scores each candidate against a **reference** state using all the evidence the backend provides, and picks the lowest cost.

### The composite score (`method: auto`)

| Evidence | Cost term | From |
|---|---|---|
| wavefunction / orbital amplitudes | (1 − \|⟨ref\|cand⟩\|) / `overlap_scale` | PySCF: AO transition densities; exkit: AO excitation densities ∑ Co X Cvᵀ; Q-Chem: squared amplitudes (+ state-density hole/electron populations with `state_analysis`) |
| energy | \|E − E_expected\| / `energy_scale` | all backends |
| ⟨S²⟩ | \|Δ⟨S²⟩\| / `s2_scale` | Q-Chem SF/MRSF, exkit |
| oscillator strength | \|Δf\| / (`osc_scale` + 0.2 max f) | PySCF, Q-Chem, exkit (spin-conserving) |
| transition dipole | ‖ \|μ\| − \|μ_ref\| ‖ / (`tdip_scale` + 0.2 max \|μ\|) | PySCF, Q-Chem, exkit (spin-conserving) |
| root number | `index_weight` × \|Δindex\| | always; a tie-breaker only |

How the terms are computed:
- Each term is about 1 when two states are clearly different.
- A term drops out when the backend does not provide its data.
- Overlaps are free of orbital phases: AO-basis densities are invariant to orbital sign flips between separate SCF runs, and Q-Chem amplitudes are compared squared.
- Transition dipoles are compared component-wise in absolute value, because their overall sign is arbitrary.

The energy reference:
- **Between optimization cycles**, `E_expected` is the energy the optimizer's quadratic model predicts for the target, E + g·s + ½ sᵀHs. One real step changes the target's own energy by more than the gap to its neighbours, and the prediction keeps the match on the right state.
- **Inside a gradient stencil**, it is the centre energy, weighted with the tighter `stencil_energy_scale`. A 0.005 Bohr displacement moves a state by micro-Hartrees, so a jump of tens of mEh is never the same state, whatever an overlap says.

Hard guards are applied before scoring:
- An excited target never collapses onto the SCF reference determinant. That determinant is flagged `scf_reference` and identified by the flag, never by position.
- Candidates of a different clean multiplicity are excluded.
- Candidates whose ⟨S²⟩ jumps by more than `s2_max_change` are excluded. A clean SF singlet is never swapped for a contaminated singlet/triplet mixture, or for a triplet with the same squared amplitudes. That second case is real: SF2 and SF4 in a butadiene SF-TDDFT output have overlapping amplitudes above 0.99.

The other `method` values are weight presets of the same score:

| `method` | Weights |
|---|---|
| `auto` / `composite` | all terms |
| `overlap` | overlap first; energy only breaks near-ties |
| `character` | energy, ⟨S²⟩, f, transition dipole; no wavefunction |
| `energy` | energy only. Follows the *adiabatic* surface: at a true crossing it switches state |
| `index` | the root number only, i.e. no tracking. Explicit opt-in for debugging |

### Spin-flip methods and spin contamination

SF-TDDFT roots can be strongly spin-contaminated (⟨S²⟩ ≈ 1). ngeoopt handles this as follows:
- **Multiplicity:** it is derived from ⟨S²⟩ with the electron-count parity, so an even-electron state with ⟨S²⟩ = 0.95 is *contaminated* (multiplicity None), not a doublet.
- **Choosing the start state:** `root` with `multiplicity` counts only clean states. `root: 1, multiplicity: 1` is the first excited *clean* singlet.
- **Matching:** ⟨S²⟩ enters every match, and jumps are excluded.
- **Warnings:** the log warns when the tracked state itself is contaminated. Within a gradient, it warns when ⟨S²⟩ varies across the stencil (by more than 0.1).

For spin-flip methods the high-spin reference is not a state of the manifold. The states are only the SF roots, so `root: 0` is the lowest SF state (usually S0).

### Reference updates

- **Inside one gradient:** the target is first matched at the undisplaced geometry, against the previous cycle's state. Every displaced geometry is then matched against that centre state.
- **Between cycles:** with `follow: true`, the reference moves to the state matched at each new accepted geometry, so the state can change character slowly. A rejected step restores the previous reference.

### Continuity check

After each gradient, every coordinate's +h/−h energies are checked. A second difference above `max_curvature` (default 5 Eh/Bohr²) gives one of two warnings:
- **"Two different states were probably differenced":** the matched root changed inside the stencil. This is a tracking problem; check `state.method` and the backend's evidence, or use a smaller step.
- **"Very large curvature … on the same state":** the root did not change, and the gap to the nearest state is reported. This is genuine strong coupling, for example a pseudo-Jahn–Teller point. The surface is real, but finite differences of this step size are unreliable there.

The second case shows up for S1 of twisted ethylene with SF-CISd. Two zwitterionic singlets about 2 mEh apart couple so strongly that any symmetry-breaking displacement lowers the target by about 1 mEh, a curvature of about −170 Eh/Bohr². The optimization moves away from that saddle on its own.

Practical notes:
- Ask for a few more roots (`nroots`/`nstates`/`cis_n_roots`) than the target, so the target cannot drop out of the computed set.
- Overlaps between cycles use the reference geometry's AO overlap matrix. That is accurate for normal steps; with very large steps, lower `trust_max`.
- `ngeoopt status` lists the root number per cycle. It changes when the state changes position in the ordering, and the log says `root flip: target state is now root 3 (was 2)`.
- `log_level: debug` prints every match's cost terms.

## 6. Parallel execution and GPUs

### Local process pool (default)

A gradient needs `1 + 2·3N` (central) independent single points. With `executor: process`, they are distributed over `workers` processes:
- the processes are started with `spawn`;
- they are reused across cycles, so PySCF imports and GPU contexts are paid once;
- each worker gets `threads_per_worker` threads (`OMP_NUM_THREADS`, MKL, OpenBLAS, Accelerate and PySCF's own setting).

Choosing workers and threads: for small molecules, many single-threaded workers are fastest. For larger ones, use fewer workers with more threads, keeping `workers × threads_per_worker ≈ cores`.

> **Python scripts:** guard the entry point with `if __name__ == "__main__":`. Workers are spawned processes that re-import your script. Without the guard, every worker re-runs the script and the pool dies ("process terminated abruptly"). The CLI already does this.

### NVIDIA GPUs (CUDA)

`device: auto` picks CUDA when cupy or torch can see a GPU. Each worker process is pinned to one GPU through `CUDA_VISIBLE_DEVICES`, assigned round-robin, so 4 GPUs give 4 concurrent single points by default:
- With `workers_per_gpu: 2`, two jobs share each GPU (useful when one job does not fill a GPU).
- `gpus: [0, 1]` restricts which devices are used.
- The PySCF backend moves the SCF/TDDFT to the GPU with gpu4pyscf when the worker's device is CUDA (`use_gpu: auto`):
  - `use_gpu: never` keeps it on the CPU;
  - `use_gpu: always` fails instead of falling back.
- Post-HF methods (MP2/CC/CASSCF) run on the CPU from the GPU SCF.

### Apple silicon (MPS)

There is no Metal backend for PySCF or Q-Chem, so their energies run on the CPU cores, and ngeoopt says so in the log. The MPS device is handed to `function` backends: write the energy in torch, accept a `device` argument, and it runs on the Apple GPU (example in `examples/python_function/mypotential.py`). MPS works in float32, so use `step: 0.01` or larger and a `loose`/`gaussian` threshold with such models.

### MPI (multi-node)

```bash
pip install ".[mpi]"
mpiexec -n 33 python -m mpi4py.futures -m ngeoopt.cli run input.yaml
```

with `parallel: {executor: mpi, workers: 32}`. Rank 0 runs the optimizer and the other ranks are workers. On GPU nodes, each worker picks a GPU from its node-local rank (`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`). The MPI path follows mpi4py's standard `MPIPoolExecutor` pattern, but is not yet covered by the test suite.

## 7. Checkpoints, restart and output files

```
ngeoopt_run/
  ngeoopt.log              log
  ngeoopt_state.npz        optimizer checkpoint (geometry, gradient, Hessian, trust, tracker reference)
  ngeoopt_state.json       same metadata, human-readable (used by `ngeoopt status`)
  trajectory.xyz           accepted geometries
  final.xyz                final geometry
  gradient.txt             (task: gradient)
  cache/                   one file per finished single point
  jobs/                    per-job scratch (deleted unless keep_files: true)
```

**What is tracked:**
- the checkpoint is written after every cycle;
- it stores the next geometry to evaluate, the best point so far with its gradient, the Hessian, the trust radius and the full cycle history;
- it also stores the state tracker's reference state, including its transition density, so root following continues correctly after a restart.

**Restart:** run the same command again (`restart: true` is the default). The optimizer continues from the stored cycle. The single-point cache is keyed by geometry and backend settings, so a gradient that was interrupted halfway recomputes only the missing displacements. A checkpoint for a different molecule is ignored. Use `--no-restart` to start fresh; the cache is still used.

Cache files are Python pickles written by ngeoopt; only load caches you created.

## 8. Backends

### PySCF (`name: pyscf`)

| Keyword | Default | Meaning |
|---|---|---|
| `method` | hf | `hf`, any PySCF xc functional (`b3lyp`, `pbe0`, `wb97x`, `camb3lyp`, …), `mp2`, `ccsd`, `ccsd(t)`, `casscf`, `casci` |
| `basis` | sto-3g | name or per-element dict |
| `ecp` | null | |
| `reference` | auto | `rhf`/`uhf`/`rohf`; auto = R for singlets, U otherwise |
| `excited` | null | `tddft`, `tda`, `cis`, `tdhf`, `eom-ccsd` |
| `nroots` | 3 | excited roots per spin manifold |
| `spin_states` | singlet | `singlet`, `triplet`, `both` (restricted references) |
| `conv_tol` | 1e-10 | SCF energy tolerance |
| `conv_tol_grad` | null | SCF orbital gradient tolerance; 1e-7 automatically for excited states and CAS |
| `max_cycle` | 200 | |
| `grids_level` | 4 | DFT grid |
| `density_fit` | false | `true` or an auxiliary basis name |
| `td_conv_tol` | 1e-8 | Davidson tolerance |
| `td_max_cycle` | 200 | |
| `ncas`, `nelecas`, `cas_nroots` | – | CASSCF/CASCI active space, number of state-averaged roots |
| `solvent` | null | `{method: pcm, eps: 78.4}` or `ddcosmo` |
| `max_memory` | 4000 | MB per worker |
| `signatures` | true | attach transition densities for overlap tracking |
| `use_gpu` | auto | `auto`, `never`, `always` |
| `scf_attrs` | {} | any extra attribute set on the SCF object, e.g. `{level_shift: 0.2, init_guess: atom}` |
| `verbose` | 0 | PySCF verbosity (written to `jobs/<label>/pyscf.log` with `keep_files`) |

State list returned:
- for TDDFT/TDA/CIS, it is `[S0 (scf_reference), excited roots sorted by energy]`;
- each excited root has a multiplicity (restricted references), an oscillator strength and a transition-density signature;
- for `spin_states: both`, singlets and triplets are merged and sorted by energy, so use `state.multiplicity` to pick the manifold;
- for CASSCF/CASCI, the states are the (state-averaged) roots, matched by character.

### exkit (`name: exkit`)

exkit methods mostly have no analytic gradients (SF-CIS/TDA, SF-CISd, MRSF-CIS/TDA, MD-SF, pp/hh-TDA, DEA/DIP, IP/EA, RI-MP2, double hybrids, …). ngeoopt drives them by finite differences with the root following of §5. Each single point mirrors exkit's own numerical-geometry energy evaluation:
- the molecule and SCF are built with `exkit.driver.build_mol` / `build_scf`, with no geometry symmetrization, which would undo a small displacement;
- every displaced SCF starts from the centre density (`gradient.guess_from_center`), which keeps broken-symmetry and open-shell references on one branch;
- an SCF that does not converge fails the job instead of being used (`retries` re-runs it).

```yaml
molecule: {xyz: ch2.xyz, multiplicity: 3}     # high-spin reference for spin-flip methods
backend:
  name: exkit
  input_file: ch2_mrsf_s1.inp                  # any exkit input; its geometry is ignored
  # or inline:
  # input: {method: mrsf-tda, basis: 6-31g*, nstates: 6}
state: {root: 1, multiplicity: 1}
```

| Keyword | Default | Meaning |
|---|---|---|
| `input` | {} | exkit keywords (`method`, `basis`, `xc`, `nstates`, `scf_*`, `jk`, `backend`, …), checked by exkit's own parser, so typos fail before any job runs |
| `input_file` | null | an exkit `.inp`; `input` entries override it |
| `signatures` | true | attach AO excitation densities for overlap tracking |
| `davidson_conv_tol` | 1e-7 | upper bound on exkit `conv_tol` |
| `davidson_max_cycle` | 200 | lower bound on exkit `max_cycle` |
| `scf_conv_tol_grad` | 1e-7 | used unless the exkit input sets it. With PySCF's default √(1e-10) the gradient error of CIS S1 was 4e-5 Eh/Bohr; with 1e-7 it is 5e-6 (central) and 3e-7 (`central4`) |
| `require_scf_converged` | true | fail a job whose SCF did not converge |
| `seed` | 0 | reseed NumPy's global random state before every job (`null` = off), see below |
| `cas_conv_tol` | 1e-10 | CAS methods, unless the input sets `cas_conv_tol`; at exkit's default 1e-8, SA-CASSCF energies scatter by ~2e-7 Eh between runs (~2e-5 Eh/Bohr gradient noise) |

ngeoopt controls these itself:
- geometry, units, charge and spin, which come from the `molecule` section;
- `verbose`, `analysis`, `optimize`, `frequency` and `symmetry` (forced to c1);
- per-job `outdir` and `threads`, and `device` on CUDA workers.

The states returned depend on the method:
- **Spin-conserving methods** (reference in the state space, e.g. `cis`, `tda`): the SCF reference, then the roots. Each root has ⟨S²⟩, multiplicity, transition dipole and oscillator strength, computed from the excitation densities. These match PySCF's for CIS exactly.
- **Spin-flip and other methods:** only the roots, with ⟨S²⟩ and an AO excitation-density signature.
- **Methods with doubles (SF-CISd):** no signature. Near-degenerate states can share almost the same singles (twisted ethylene's zwitterionic pair overlaps 0.97 with the *other* state in its singles part), so a singles-only overlap misleads. They are tracked by ⟨S²⟩, the predicted energy and the root number. For the twisted-ethylene S1 run, exkit's own doubles-aware overlaps confirm the path chosen.
- **Ground-state methods** (MP2, double hybrids): one state.
- **Not supported:** methods with `supports_geomopt = False` (e.g. IP/EA-EOM-MP2, TEA/TIP) and periodic methods are refused.
- **CASSCF and multiple solutions:** exkit's CAS path takes its initial CI guess from NumPy's global random state. For water CAS(4,4)/STO-3G, `exkit.run()` called repeatedly in one process lands on two different SA-CASSCF solutions (−74.9800 / −74.9684 Eh). A worker process runs many displaced jobs, so the backend reseeds before every job, and every displacement starts from the same random state and finds the same solution.
- **Q-Chem integrals:** with exkit's `integrals: qchem`, exkit writes and runs Q-Chem in each job's directory.

Verified against `exkit.run()` (energies to ~1e-13 Eh): cis, tda, sf-cis, sf-tda, sf-cisd, mrsf-cis, mrsf-tda, mrsf-tddft, md-sf-cis, md-sf-tda, pp-tda, hh-tda, dea, dip, ip-cis, ip-tda, ea-cis, ea-tda, mp2, b2plyp. The CIS S1 finite-difference gradient matches PySCF's analytic TDA gradient (5e-6 central, 3e-7 `central4`).

Examples: `examples/exkit/mrsf_s1.yaml` (MRSF-TDA S1 of CH₂, 7 cycles) and `examples/exkit/sf_cisd_twisted.yaml` (SF-CISd S1 of twisted ethylene, through a root reordering and a strongly coupled region).

### Q-Chem (`name: qchem`)

The input file is generated from a `rem` dict (or your own `template_file` with `{{placeholders}}`), plus optional `extra_sections`. Each single point runs as `qchem -nt {nthreads} job.in job.out` in its own directory, with its own `QCSCRATCH`, so concurrent jobs never share scratch.

```yaml
backend:
  name: qchem
  rem: {method: bhhlyp, basis: cc-pvdz, unrestricted: false, spin_flip: 2,
        cis_n_roots: 6, cis_singlets: true, scf_convergence: 10, cis_convergence: 8, xc_grid: 3}
```

**Built-in parser.** Written and tested against Q-Chem 7.0 outputs (butadiene, BHHLYP/cc-pVDZ):

| Calculation | What is read |
|---|---|
| SCF / DFT | energy from the converged SCF iteration line (10 decimals; `Total energy =` has only 8) |
| TDDFT / TDA / CIS | the last `… Excitation Energies` block (full TDDFT when TDA is also printed): total energy, `Multiplicity`, oscillator strength, transition moment, amplitudes. The SCF reference is the first state |
| SF-TDDFT, MRSF-TDDFT | the SF roots only (the "ground state" is SF state 1); multiplicity from ⟨S²⟩ with electron parity, so contaminated roots get none; `Strength`/`Trans. Mom.` when printed (MRSF: relative to the lowest MRSF state) |
| Davidson | the last iteration table: if roots are left unconverged, the highest ones are flagged |
| `state_analysis` | hole/electron Mulliken populations per atom from the *state/difference* density matrices, added to the signature |

Root-following evidence:
- squared CI amplitudes, keyed by excitation and spin (phase-free);
- energies, ⟨S²⟩, oscillator strengths and transition moments;
- with `state_analysis: true`, the per-atom hole/electron populations.

Anything derived from *transition* densities (TDM populations, Ω matrices in `ctnum_mulliken.om`, NTO decompositions) is deliberately not used. In MRSF outputs those are currently incomplete: SA-NTO weights above 100% appear.

What ngeoopt adds or warns about:
- `sym_ignore = true` is added unless you set `sym_ignore`/`symmetry`. A 0.005 Bohr displacement can fall inside Q-Chem's symmetry tolerance and be symmetrized away.
- Warnings if `SCF_CONVERGENCE` < 9, if `CIS_CONVERGENCE` < 7, or if a DFT run has no `XC_GRID` (the SG-1 default grid is too coarse for smooth differences).
- Precision: excited-state energies are printed with 8 decimals, so the default step for Q-Chem is 0.01 Bohr unless you set `gradient.step`. The `tight`/`verytight` criteria log a warning, because rounding limits gradients to about 5e-7 Eh/Bohr.
- MP2/CC/EOM outputs are not parsed yet (no reference outputs). For those, set `energy_regex` / `states_regex` or a `parser`, exactly as for the template backend.

### Any program (`name: template`)

| Keyword | Meaning |
|---|---|
| `template` / `template_file` | input text with `{{placeholders}}` |
| `command` | shell command; `{input}`, `{output}`, `{nthreads}`, `{workdir}`, `{label}` are substituted |
| `input_name` / `output_name` | file names inside the job directory (default `job.inp` / `job.out`) |
| `geometry_units` | `angstrom` (default) or `bohr` for `{{geometry}}` |
| `variables` | extra placeholders: `{method: b3lyp}` → `{{method}}` |
| `energy_regex` | regex with one group; the last match is used (`energy_match: first` for the first) |
| `energy_units` | `hartree`, `ev`, `kcal/mol`, `kj/mol` |
| `states_regex` | every match is an extra state |
| `states_are_excitations` | the matches are excitation energies, added to the ground energy |
| `states_units` | units of the state energies (default: `energy_units`) |
| `parser` | `module:function` or `path/file.py:function`, called as `f(output_text, workdir)` and returning an energy, a list of state energies, a dict or an `EnergyResult` |
| `success_regex` | must appear in the output, otherwise the job counts as failed |
| `env` | extra environment variables |
| `scratch_env` | name of an environment variable to point at a private scratch directory |
| `copy_files` | files copied into every job directory (basis sets, restart guesses) |
| `timeout` | seconds per job |

Template placeholders:

| Placeholder | Value |
|---|---|
| `{{geometry}}` | `El x y z` lines |
| `{{xyz}}` | a complete XYZ file |
| `{{charge}}`, `{{multiplicity}}`, `{{natoms}}` | from the molecule |
| `{{nthreads}}`, `{{label}}`, `{{workdir}}` | from the job |
| anything in `variables` | your own values |

Double braces never clash with `$rem`, `%pal` or single braces in real input formats.

File paths in the backend section (`template_file`, `copy_files`, and `.py` files in `parser`/`function`) are relative to the input file's folder.

**Walkthrough:** [`examples/generic_code`](../examples/generic_code) is a complete, runnable example. It covers a numerical gradient, a ground-state optimization, and an excited-state optimization through a root reordering, all with a stand-in external program. It also includes a blank `TEMPLATE_your_code.yaml` to fill in for your own code. Real-program examples are in `examples/template_xtb` and `examples/template_orca`.

### Any Python function (`name: function`)

```yaml
backend:
  name: function
  function: mypotential.py:energy      # or an importable 'package.module:energy'
  kwargs: {model_path: model.pt}
  coords_units: bohr                   # what your function expects
  energy_units: hartree                # what it returns
```

```python
def energy(symbols, coords, device="cpu", model_path=None):
    ...
    return e                     # float, list of state energies, dict or EnergyResult
```

- A `device` argument, if the function accepts one, receives `cpu`, `cuda:0` or `mps`.
- A `workdir` argument receives a private directory.
- torch and cupy return values are copied to the host.
- For overlap tracking, return `StateInfo` objects with a `signature` array (any shape, the same at every geometry).

### Writing a backend class

```python
from dataclasses import dataclass
from ngeoopt import EnergyBackend, EnergyResult, StateInfo, register_backend

@dataclass
class MyCode(EnergyBackend):
    method: str = "x"
    name = "mycode"

    def compute(self, mol, ctx):            # mol.coords in Bohr; ctx.workdir, ctx.nthreads, ctx.device
        ...
        return EnergyResult(states=[StateInfo(energy=e0, extra={"scf_reference": True}), ...])

register_backend("mycode", MyCode)          # now usable as backend: {name: mycode}
```

Backends are pickled to the worker processes, so keep them as plain configuration and create heavy objects inside `compute`.

## 9. Python API

```python
from ngeoopt import (Molecule, optimize, gradient, run, FDSettings, OptimizerSettings,
                     ConvergenceCriteria, TrackingSettings, FunctionBackend, TemplateBackend)
from ngeoopt.backends.pyscf_backend import PySCFBackend

if __name__ == "__main__":
    mol = Molecule.from_xyz("h2co.xyz")
    be = PySCFBackend(method="b3lyp", basis="6-31g*", excited="tda", nroots=5)

    g = gradient(mol, be, workers=8, state=TrackingSettings(root=1, multiplicity=1))
    print(g.energy, g.gradient, g.tracking)

    res = optimize(mol, be, workers=8, workdir="s1_opt",
                   state=TrackingSettings(root=1, multiplicity=1),
                   gradient=FDSettings(scheme="central", step=0.005),
                   optimizer=OptimizerSettings(convergence=ConvergenceCriteria.preset("tight")))
    print(res.converged, res.energy, res.molecule.to_xyz())

    run("input.yaml")                       # same as the CLI
```

The lower-level pieces can be combined freely: `make_executor`, `FiniteDifferenceGradient`, `StateTracker`, `Optimizer` and `ResultCache`.

## 10. Accuracy and performance tips

* **SCF convergence controls gradient noise.** An energy error δ gives a gradient error of about δ/h. With h = 0.005 Bohr, δ = 1e-10 Eh gives 1e-8 Eh/Bohr, far below any threshold. For excited states, the orbital gradient also matters, which is why `conv_tol_grad` tightens automatically. For external codes, use the tightest SCF and response convergence the code offers.
* **Turn symmetry off** in external codes (`sym_ignore true` in Q-Chem, `NoSym` in Gaussian/ORCA). Displacements break symmetry, and a code that reorients or symmetrizes the molecule breaks the stencil.
* **Step size:**
  - 0.005 Bohr central is a good default.
  - Go up to 0.01 for noisy methods or float32 models.
  - `central4` costs twice as much but gives O(h⁴) accuracy for tight optimizations.
  - `forward` halves the cost but is only O(h); use it only for rough pre-optimizations.
* **Cost per cycle** is `1 + 6N` single points (central). Parallel wall time is about ⌈(6N+1)/workers⌉ single points.
* **Freezing atoms** (`frozen_atoms`) removes their displacements entirely.
* **Small molecules:** use many single-thread workers. **Large molecules:** use few multi-thread workers or GPUs.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `A process in the process pool was terminated abruptly` | missing `if __name__ == "__main__":` guard in your script, or a worker ran out of memory (lower `workers` or `max_memory`) |
| `unknown keyword(s) in [gradient]: ['stepsize']` | typo; the message lists the valid keywords |
| `best state overlap 0.3 below threshold` | strong mixing or a too-large step: lower `trust_max`, add roots, or check the state character |
| `tracked state not converged in N single point(s)` | raise `td_max_cycle` or `nroots` |
| optimization oscillates near the end | gradient noise: tighten SCF/response convergence, or use `central4` |
| `device 'cuda' requested but no CUDA device is visible` | install cupy / gpu4pyscf, check `CUDA_VISIBLE_DEVICES` |
| a PySCF job on a Mac logs CPU only | expected: PySCF has no Metal backend |
| Q-Chem `NotImplementedError` for MP2/CC | correlated outputs are not parsed yet; set `energy_regex` (§8) |
| `… Two different states were probably differenced` | the matched root changed inside a stencil and the energy jumped: check `state.method`, add evidence (e.g. `state_analysis`), or use a smaller step |
| `very large curvature … on the same state` | genuine strong coupling to a state a few mEh away (pseudo-Jahn–Teller point); finite differences are unreliable there, but the optimizer usually leaves such saddles on its own |
| `tracked state is spin-contaminated` | the target SF root has ⟨S²⟩ far from S(S+1); consider MRSF, or check that `root`/`multiplicity` pick the state you mean |
| `every candidate changes <S^2> by more than …` | the tracked state lost its spin character between geometries; raise `s2_max_change` only if that is physical |
| `exkit SCF did not converge` | use exkit SCF controls (`scf_algorithm = soscf`, `scf_level_shift`, `scf_damping`); `parallel.retries` re-runs failed jobs |
