# didactic-lamp

A buildable C++ universe-simulation prototype with advanced features:

- procedural terrain (fBm over Perlin-like noise)
- dynamic climate seasonality + altitude/latitude temperature effects
- renewable resource field with logistic regrowth and occupancy pressure
- autonomous agents with stochastic policies (softmax action sampling)
- intrinsic novelty reward (exploration pressure)
- social density reward (coordination pressure)
- evolutionary loop with elitism, crossover, and adaptive mutation
- species clustering and diversity tracking (Shannon entropy)
- JSON output for post-run analysis

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/universe_sim [agents] [generations] [steps_per_generation] [temperature]
```

Example:

```bash
./build/universe_sim 2048 8 140 0.7
```

This writes `simulation_summary.json` in the repository root.
