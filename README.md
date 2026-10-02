# Didactic Lamp

A C++17 ecosystem sandbox for exploring evolutionary policies, climate niches,
resources, predation, and disturbances. Version 5 adds reproducible experiments,
more reliable simulation rules, and a comparative Python dashboard.

This is an uncalibrated model with abstract units. Its species are genome
clusters; its results are not predictions about real ecosystems.

## What changed

- **Reliable transitions:** bounded reproduction, deferred newborn insertion,
  consistent parental cooldowns, immediate starvation, and no repeated hunting
  attempts caused by clamped boundary cells.
- **Live ecological pressure:** occupancy now measures living body mass, with
  visitation retained separately. Optional seed-bank recovery allows depleted
  resources to return; set `--resource-recovery 0` to disable it.
- **Whole-genome comparisons:** mating and species clustering include all 295
  genes, and trophic types use separate clusters. The default mating threshold
  is now 0.50 to match the new distance scale.
- **Trustworthy observations:** tick 0 is the initial state; the actual final
  state is always written, including extinction. Snapshot frequency no longer
  changes the classification schedule or threshold.
- **Better measurements:** effective Shannon species, inverse Simpson diversity,
  evenness, exact demographic totals, and population-time integrals.
- **Auditable experiments:** matched seed pairs, paired bootstrap intervals, raw
  CSVs, every command/configuration, and SHA-256 receipts for executable and results.
- **Clearer dashboard:** exact disturbance bands, run metadata, irregular-interval
  demographic bars, comparison overlays, and configurable image destinations.

## Build and check

Requires CMake 3.16+, a C++17 compiler, and Python 3.10+ for the experiment runner.
Matplotlib is only needed for the dashboard and its rendering tests.

Windows with the installed MSYS2 UCRT64 compiler and Ninja:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$pythonExe = (Get-Command python).Source
python -m pip install -r requirements.txt
cmake -S . -B build-research -G Ninja -DCMAKE_BUILD_TYPE=Release "-DPython3_EXECUTABLE=$pythonExe"
cmake --build build-research -j 4
ctest --test-dir build-research --output-on-failure
```

Keep the UCRT64 directory on PATH when running this executable so its runtime
DLLs can be found. Using a separate build directory also avoids old generator
caches. CMake's `BUILD_TESTING=OFF` permits a simulator-only build.

On another C++17 toolchain, use its normal CMake generator. The tests receive the
built executable path from CTest; bitwise outputs are only expected to repeat
within the same executable/runtime environment.

## Run and visualize

```powershell
.\build-research\universe_sim.exe --agents 400 --ticks 900 --width 64 --height 64 --snapshot 25 --seed 7 --output simulation_summary.json
python visualize.py simulation_summary.json --output output\dashboard.png
```

Use `--show` to open a Matplotlib window if an interactive Tk/Qt backend is
available. Omit it for unattended image generation. Legacy positional simulator
arguments (`agents ticks temperature`) still work; use either positional or
named simulation options, not both.

Useful controls:

| Option | Purpose / default |
| --- | --- |
| `--seed` | Unsigned 64-bit random seed; 7 |
| `--agents`, `--max-agents` | Initial population 2048; hard live cap 6000 |
| `--ticks` | Requested transitions; 5000 |
| `--width`, `--height` | Grid dimensions; 128 by 128 |
| `--snapshot` | Reporting interval; 100 |
| `--classification` | Model clustering interval, independent of reporting; 25 |
| `--predator-ratio` | Initial predator fraction; 0.05 |
| `--temperature` | Policy softmax temperature; 0.8 |
| `--hunt-success` | Base hunt probability before modifiers; 0.12 |
| `--reproduction` | Required parental energy; 11 |
| `--speciation`, `--reproductive-distance` | Full-genome RMS thresholds; 0.55 and 0.50 |
| `--shock-interval`, `--shock-duration`, `--shock-strength` | 180 ticks, 45 ticks, 0.18; zero disables shocks |
| `--resource-recovery` | Abstract seed-bank coefficient; 0.002, zero disables |
| `--output` | JSON destination; simulation_summary.json |

Run `universe_sim.exe --help` for valid ranges. Invalid, nonfinite, and
partially parsed values are rejected rather than silently clamped. Output
parents must exist for the simulator; write failures return a nonzero exit code.

## Compare interventions across seeds

```powershell
python experiments.py --simulator build-research\universe_sim.exe --parameter resource-recovery --control 0 --treatment .002 --seeds 1,2,3,4,5,6,7,8,9,10 --agents 400 --ticks 900 --width 64 --height 64 --snapshot 25 --output-dir output\my-recovery-study
```

Alternatively use `--parameter shock-strength --control 0 --treatment .22`.
Output directories must be new or empty. Each seed gets separate control and
treatment summaries, logs, and receipts; the root contains `experiment.json`,
`rows.csv`, `paired_deltas.csv`, and `summary.json`. Failed or partial runs keep
failure records and do not produce a completed statistical summary.

Compare the two trajectories for one seed:

```powershell
python visualize.py output\my-recovery-study\seed_7\control\simulation_summary.json --compare output\my-recovery-study\seed_7\treatment\simulation_summary.json --output output\comparison.png
```

The runner reports treatment-minus-control differences in final population,
mean population over the requested horizon, restricted persistence, and final
effective species. Confidence intervals resample whole seed pairs, never
snapshots. One pair has no interval; fewer than ten pairs are labeled exploratory.
Ten is a reporting convention, not a guarantee of statistical power.

Shared seeds match initial conditions. Later branching can shift random draws,
so these are not fully aligned common-random-number counterfactuals. A positive
change in one metric does not establish general superiority or biological realism.

## Output and model reference

Schema version 2 retains the original `ticks` and `species_records` arrays and
adds complete effective configuration, run status, exact extinction endpoint,
and explicit event windows. Initial/final snapshots and total births/deaths
allow population conservation checks. Summaries are not resumable checkpoints.

Version 5 changes trajectories: corrected occupancy, full-genome distances,
new recovery, fixed classification cadence, and exact step counting all matter.
Do not compare v4 and v5 seeds as if only one parameter changed.

- [Model specification and remaining limits](docs/model.md)
- [Research sources and their implementation relevance](docs/research.md)
- [Validation results](docs/validation.md)

The model still mixes behavioral rewards with energy and uses heuristic
coefficients and sequential agent processing. These boundaries are documented
explicitly so experiments remain interpretable.
