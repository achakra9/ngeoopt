# Using any program with ngeoopt

This folder is a complete, runnable example of driving an external program that can only compute energies. ngeoopt writes its inputs, runs them in parallel, reads the energies back, builds numerical gradients and optimizes the geometry.

The "program" here is `fakeqc.py`, a stand-in that reads an input file and prints energies in its own format, like a real code. It models water with a ground state and two excited states: a bright state (f = 0.052) and a dark one (f = 0.003). The two excited states swap order as the H–O–H angle changes. To use your own code, you only change the template, the command and how the energy is read (see [Adapting it to your code](#adapting-it-to-your-code)).

| File | What it is |
|---|---|
| `fakeqc.py` | the stand-in program (replace with yours) |
| `job.tpl` | input template with `{{placeholders}}` |
| `water.xyz` | starting geometry |
| `gradient.yaml` | step 1: one numerical gradient |
| `optimize.yaml` | step 2: ground-state optimization |
| `parse_fakeqc.py`, `excited.yaml` | step 3: excited-state optimization with root following |
| `run_from_python.py` | the same three runs through the Python API |
| `TEMPLATE_your_code.yaml` | blank, fully commented input to fill in for your program |

## The three pieces ngeoopt needs

**1. An input template.** Take a working input for your program and replace the geometry with `{{geometry}}`:

```
# job.tpl
method   {{method}}
basis    {{basis}}
nthreads {{nthreads}}
charge   {{charge}}
mult     {{multiplicity}}
geometry
{{geometry}}
end
```

`{{method}}` and `{{basis}}` come from `variables:` in the YAML. `{{charge}}`, `{{multiplicity}}`, `{{nthreads}}`, `{{natoms}}`, `{{label}}` and `{{xyz}}` are filled in automatically. Double braces never clash with `$`, `%` or `{}` in real input formats.

**2. A command.** It runs inside a private directory for each single point:

```yaml
command: "python3 fakeqc.py {input} > {output}"
copy_files: [fakeqc.py]          # anything every job directory needs
```

**3. A way to read the energy.** For one energy, a regex is enough:

```yaml
energy_regex: 'FINAL ENERGY:\s+(\S+)'    # one group around the number; last match wins
success_regex: 'terminated normally'      # otherwise the job counts as failed
```

For excited states, use `states_regex`, or a small parser function that also returns multiplicities and oscillator strengths. With those, ngeoopt can follow a state by its character (step 3).

## Step 1: numerical gradient

```bash
ngeoopt run gradient.yaml
```

This runs 1 + 6N = 19 single points (central differences, 0.005 Bohr) in parallel on 4 workers. It writes `gradient_run/gradient.txt` (Hartree/Bohr):

```
# Cartesian gradient, Hartree/Bohr
 1.8373288218e-17 -1.0794758566e-01 -4.4232830645e-03
-5.0920367434e-18  5.3565207044e-02 -3.9131983338e-02
-1.3281251475e-17  5.4382378614e-02  4.3555266403e-02
```

`keep_files: true` keeps every job directory (`gradient_run/jobs/gradient/a000x+1/job.inp`, `job.out`, …) so you can check what was run.

## Step 2: geometry optimization

```bash
ngeoopt run optimize.yaml
ngeoopt status optimize_run       # in another terminal, at any time
```

```
cycle   1  E=-76.3907336391  dE=   n/a     maxF=1.08e-01  rmsF=4.82e-02  ...
cycle   2  E=-76.3990630955  dE=-8.33e-03  maxF=3.10e-02  rmsF=1.50e-02  ...
cycle   3  E=-76.3999698891  dE=-9.07e-04  maxF=4.98e-03  rmsF=2.49e-03  ...
cycle   4  E=-76.3999982690  dE=-2.84e-05  maxF=1.23e-03  rmsF=6.05e-04  ...
cycle   5  E=-76.3999999761  dE=-1.71e-06  maxF=7.57e-05  rmsF=4.44e-05  ...  CONVERGED
```

Results go to `optimize_run/`: `final.xyz`, `trajectory.xyz`, `ngeoopt.log` and a checkpoint. If you kill the run and start it again, it continues from the last cycle, and only single points that never finished are recomputed.

## Step 3: excited-state optimization with root following

```bash
ngeoopt run excited.yaml
```

The target is the **bright** state. At the starting geometry it is root 2. As the angle opens, it drops below the dark state and becomes root 1. If the optimizer simply kept "root 2", it would switch to the dark state at that point. Instead, the parser reports every state's multiplicity and oscillator strength:

```python
# parse_fakeqc.py
def parse(text, workdir):
    e0 = float(re.search(r"FINAL ENERGY:\s+(\S+)", text).group(1))
    states = [StateInfo(energy=e0, multiplicity=1, extra={"scf_reference": True})]
    for n, e, mult, f in re.findall(r"State\s+(\d+):\s+E =\s+(\S+) Eh.*mult =\s+(\d+)\s+f =\s+(\S+)", text):
        states.append(StateInfo(energy=float(e), multiplicity=int(mult), oscillator_strength=float(f)))
    ...
```

`excited.yaml` then tells ngeoopt to follow the state by character:

```yaml
backend:
  parser: parse_fakeqc.py:parse
state:
  root: 2              # the bright state, at the starting geometry
  multiplicity: 1
  method: character    # match multiplicity + energy + oscillator strength
```

```
tracking target: position 2 (E=-76.0951083431, mult=1, f=0.052)
cycle   1  E=-76.0951083431  ...  root=2
root flip: target state is now root 1 (was 2)
cycle   2  E=-76.1294522353  ...  root=1
...
cycle   7  E=-76.1427547443  ...  root=1  CONVERGED
```

The run ends at the bright state's minimum (−76.14275 Eh), not the dark state's (−76.12071 Eh), even though the root number changed along the way. Between cycles, ngeoopt compares each state against the energy its quadratic model *predicts* for the target. This is why a large optimization step does not confuse the match.

## The same thing from Python

```bash
python run_from_python.py
```

See the file. Wrap the code in `if __name__ == "__main__":`, because the workers are spawned processes.

## Adapting it to your code

1. Copy `TEMPLATE_your_code.yaml` and fill in every `<<...>>`.
2. Make a template from a working input of your program: put `{{geometry}}` where the coordinates go, and `{{charge}}`/`{{multiplicity}}`/`{{nthreads}}` where they belong.
3. Set `command` to how you run the program in a shell, using `{input}` and `{output}`.
4. Set `energy_regex` to match the final energy line. Check it in Python on a real output: `re.findall(regex, open('out').read())`.
5. Run `task: energy` first, then `task: gradient`, then `task: optimize`.

**For accurate numerical gradients:**
- **SCF convergence:** converge the SCF (and any response or excited-state solver) tightly, to about 1e-9 Eh or better. The gradient error is roughly the energy noise divided by the step (0.01 Bohr for central differences).
- **Symmetry and reorientation:** turn them off in your program (`NoSym`, `sym_ignore`, `nosymm` …). Displacements break symmetry, and reorienting the molecule breaks the finite-difference stencil.
- **Threads:** keep `workers × threads_per_worker` at or below your core count.

Real-program examples using the same mechanism: [`../template_xtb`](../template_xtb) and [`../template_orca`](../template_orca).
