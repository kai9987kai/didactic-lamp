# didactic-lamp

A buildable C++ universe-simulation prototype with advanced features:

## Core Systems

- **Procedural terrain** — fBm over Perlin-like noise with island falloff
- **Water cycle** — altitude-dependent moisture, evaporation/condensation dynamics
- **6 Biome types** — Ocean, Tundra, Desert, Grassland, Forest, Jungle (dynamic classification from height × temperature × moisture)
- **Dynamic climate** — seasonality + altitude/latitude temperature effects
- **Renewable resources** — logistic regrowth modulated by biome carrying capacity and occupancy pressure

## Agent Intelligence

- **Neural policy network** — 2-layer genome-encoded neural net (11 inputs → 8 hidden [tanh] → 5 actions)
- **Softmax action sampling** — temperature-controlled stochastic policies
- **Pheromone communication** — stigmergy-based signaling (food trails + danger signals) with decay and diffusion
- **Biome-aware movement** — movement costs vary by terrain type; ocean is impassable

## Predator-Prey Ecosystem

- **Trophic roles** — Herbivores forage resources, Predators hunt herbivores
- **Hunting mechanics** — spatial proximity search with energy-dependent success probability
- **Co-evolution** — type-segregated elitism ensures both trophic levels evolve independently

## Life & Death

- **Age & metabolism** — metabolic cost increases with age; genome-derived max lifespan (40–220 steps)
- **Natural death** — agents exceeding their lifespan die and release nutrients (corpse recycling)
- **Intrinsic novelty reward** — exploration pressure via visitation count
- **Social density reward** — coordination pressure from moderate neighbor density

## Evolution & Speciation

- **Evolutionary loop** — elitism, crossover, and adaptive mutation rate
- **Genetic-distance speciation** — L2-norm species classification with centroid tracking
- **Reproductive isolation** — type-segregated breeding pools
- **Adaptive radiation** — speciation threshold tightens over generations
- **Species lifecycle tracking** — birth, peak population, extinction events

## Output & Telemetry

- **Rich JSON output** — per-generation metrics, species records, biome distributions
- **Expanded metrics** — herbivore/predator counts, species diversity (Shannon entropy), mean age, pheromone totals, extinction/speciation events

## Architecture

```
src/
  types.h      — Core structs, enums, constants, helpers
  noise.h      — Perlin noise and fBm functions
  world.h      — World building, climate, biomes, pheromone dynamics
  agents.h     — Neural policy, action sampling, movement, hunting
  evolution.h  — Evolution, speciation tracking, genetic distance
  metrics.h    — Metrics computation, JSON output
  main.cpp     — Entry point, CLI parsing, simulation loop
```

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
