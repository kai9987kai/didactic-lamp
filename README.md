# didactic-lamp

A buildable C++ ecosystem sandbox focused on emergent adaptation, trophic pressure, and world-scale regime shifts.

## What is new

- Climate niche adaptation: agents now evolve preferred temperature and moisture ranges, and survival depends on how well a cell matches that phenotype.
- World shock cycle: droughts, blooms, cold snaps, and toxic blooms periodically rewire the map and create stronger ecological phase changes.
- Stronger reproductive isolation: mating now depends on genetic distance instead of stale species ids, which produces cleaner early diversification.
- Better telemetry: snapshots include habitat match, world resources, toxicity, active event, and event intensity.
- Better operator controls: the sim supports a flag-based CLI instead of only brittle positional arguments.
- Better dashboard: the Python visualizer now plots event windows, world-state metrics, niche fit, and demographic churn.

## Model overview

### World systems

- Procedural island terrain with biome classification
- Seasonal climate and moisture drift
- Renewable resources with occupancy pressure
- Toxic flora dynamics
- Pheromone diffusion and decay
- Periodic macro events that perturb climate and ecology

### Agents

- Genome-encoded recurrent neural policy
- Morphology traits: size, speed, sensory radius, toxicity resistance
- Niche traits: preferred temperature and preferred moisture
- Herbivore foraging and predator hunting
- Energy, metabolism, aging, reproduction, and death

### Evolution

- Genetic-distance speciation
- Distance-gated mating
- Continuous mutation and crossover
- Species birth and extinction tracking

## Build

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cmake -S . -B build
cmake --build build
```

On this Windows/MSYS2 toolchain, `C:\msys64\ucrt64\bin` must be on `PATH` so `cc1plus.exe` can find its runtime DLLs.

## Quickstart

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cmake -S . -B build
cmake --build build
.\build\universe_sim.exe --agents 1200 --ticks 900 --snapshot 100 --temperature 0.75 --shock-interval 150 --shock-duration 35 --shock-strength 0.22
python visualize.py simulation_summary.json
```

## Run

Legacy positional usage still works:

```powershell
.\build\universe_sim.exe 1200 900 0.75
```

Flag-based usage is preferred:

```powershell
.\build\universe_sim.exe `
  --agents 1200 `
  --ticks 900 `
  --snapshot 100 `
  --temperature 0.75 `
  --shock-interval 150 `
  --shock-duration 35 `
  --shock-strength 0.22
```

Useful flags:

- `--agents`
- `--ticks`
- `--temperature`
- `--snapshot`
- `--seed`
- `--predator-ratio`
- `--reproduction`
- `--speciation`
- `--reproductive-distance`
- `--shock-interval`
- `--shock-duration`
- `--shock-strength`

The simulation writes `simulation_summary.json` to the repository root.
Run `.\build\universe_sim.exe --help` to see the full CLI.

## Visualize

```powershell
python visualize.py simulation_summary.json
```

This generates `simulation_dashboard.png` with:

- population dynamics
- fitness vs. niche fit
- world resources, toxicity, and pheromones
- biome distribution
- diversity tracking
- births, deaths, and event intensity

Use `python visualize.py simulation_summary.json --show` if you want the plot window as well.

## Latest verified run

Verified on March 5, 2026 with:

```powershell
.\build\universe_sim.exe --agents 1200 --ticks 900 --snapshot 100 --temperature 0.75 --shock-interval 150 --shock-duration 35 --shock-strength 0.22
```

Observed behavior:

- 3 species were present through the early and middle simulation snapshots.
- A `Bloom` phase was visible at tick 300 and a `ToxicBloom` phase at tick 600.
- The population collapsed to extinction at tick 875 in this configuration.

Generated artifacts:

- `simulation_summary.json`
- `simulation_dashboard.png`
