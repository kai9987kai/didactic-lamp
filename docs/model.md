# Model specification (v5)

## Purpose and interpretation

Didactic Lamp is an uncalibrated agent-based ecosystem sandbox for exploring
how a particular set of rules interacts with stochastic policies, climate,
resources, reproduction, and disturbances. Space, time, energy, and fitness
use abstract model units. It does not predict real ecosystems, establish
biological speciation, or demonstrate open-ended evolution or intelligence.

This specification follows the organizing principles of the
[ODD protocol](https://www.jasss.org/23/2/7.html). Its assumptions are part of
the model, not empirical findings.

## State and initialization

The bounded 2D grid holds terrain height, temperature, moisture, biome,
resources, toxicity, live occupancy, visitation, and pheromone fields.
Occupancy is the sum of living body sizes at each cell, not a historical trail.
Visitation separately accumulates movement visits.

Agents have position, energy, age, lifespan, trophic type, mating role,
fitness, four recurrent memory values, and a 295-value genome. A network
with 18 inputs, 10 hidden units, and 9 outputs controls five actions and
four recurrent memory updates. Six terminal genes encode morphology and
temperature/moisture preferences. Policy weights evolve through inheritance
and mutation; there is no gradient training during life.

Procedural terrain uses the configured seed. Population initialization and
all stochastic decisions use one `std::mt19937_64` stream. Initial genomes
are normal draws with standard deviation 0.35; agent positions, lifespans,
and mating roles are randomized. The predator fraction is rounded down
to an integer count. Agents can start over ocean and incur escape costs.

## Exact transition order

Tick 0 records the untouched initial world and population after initial
species classification. A requested run of N ticks performs at most N
transitions, numbered 1 through N:

1. Rebuild occupancy from living agents at the start of the step.
2. Update climate, biome labels, resources, toxicity, and pheromones.
   Neighbor pressure reads a fixed occupancy field throughout this update.
3. Process agents in vector order: aging/death, stochastic movement,
   policy/memory, foraging, energy/fitness changes, then hunting.
   Agents without movement opportunities still pay basal metabolism.
4. Select eligible parents among the pre-birth population. Both must afford
   the five-unit contribution, meet the configured threshold, and complete
   a 30-tick cooldown. Append the bounded birth queue after selection.
   Newborns cannot mate in their birth step. Enforce the live population cap.
5. Cull dead agents and rebuild occupancy. Record exact demographic totals
   and add the post-step population to the population-time integral.
6. Classify species every `classification_interval` ticks and at the final
   endpoint. Save observations every `snapshot_interval` ticks and at the
   final endpoint. Stop at extinction or the requested horizon.

Classification never runs because an observation was requested. Changing
`--snapshot` therefore changes reporting only. Initial and final observations
are mandatory; births and deaths cover the interval since the previous
observation. The final world state remains available after extinction.

## Resources, disturbances, and classification

Resources have local logistic growth, occupancy pressure, foraging removal,
carcass inputs, and event effects. The new recovery term is
`resource_recovery * max(0, K - resources)`, where K is a biome reference
capacity. This represents an abstract persistent seed bank, without tracking
seed mass. Set the coefficient to zero for strictly local growth: an empty
cell then stays empty unless another source replenishes it. Recovery defaults
to 0.002; it is an experimental assumption, not a fitted ecological constant.
K is an equilibrium reference, not a hard cap. World updates clamp resources
to [0, 1.5]; subsequent carcass inputs can temporarily exceed that range.

Shocks cycle through drought, bloom, cold snap, and toxic bloom. Their
strength follows the existing within-window envelope. A positive interval,
duration, and strength are all required. JSON `event_windows` gives observed
schedule intervals `[start_tick, end_tick_exclusive)`, truncated at
`ticks_completed + 1`. These are independent of snapshot spacing.

Mating distance and clustering both use root-mean-square differences across
all 295 genes. Herbivores and predators cannot share a cluster. Newborns
inherit a provisional parental cluster until the next classification.
The full-genome distance concentrates more tightly than the old 32-gene
distance: the default mating threshold is now 0.50 instead of 0.42 to allow
initial reproductive compatibility. Neither threshold is biologically fitted.

Cluster centroids update sequentially at a fixed threshold; they depend on
agent order and classification cadence. Recorded cluster births, extinctions,
and peak sizes are observations at that cadence, not exact per-tick lineage
history. “Species” throughout the dashboard means these algorithmic clusters.

## Metrics and reproducibility

Richness counts occupied cluster labels. With living-label proportions p:
Shannon entropy is `-sum(p*log(p))`, effective Shannon species is `exp(H)`,
inverse Simpson diversity is `1/sum(p*p)`, and evenness is `H/log(richness)`.
For no living agents all are zero; one cluster has evenness one. Agent means
exclude dead agents. World means include every grid cell, including ocean.

The paired experiment runner resamples entire seed pairs for percentile
bootstrap intervals. Its mean population uses the exact post-step integral
divided by the requested horizon; population is defined as zero for remaining
ticks after extinction. Persistence is restricted to that horizon and surviving
runs are right-censored. Final diversity is measured at the actual endpoint.

Repeated runs are expected to match with the same executable, standard library,
seed, and configuration. Standard-library random distributions and floating
point behavior prevent a cross-toolchain bitwise guarantee. Identical seeds
align initial states; branching can consume different random draws, so these
are not fully aligned common-random-number counterfactuals. Saved summaries
are observations, not resumable simulation checkpoints.

## Remaining model limits

- Sequential foraging, predation, and parent selection can favor earlier agents.
- Fitness mixes foraging and behavioral bonuses. Novelty, social, and habitat
  rewards still contribute to energy; this is not a conserved energy budget.
- Zero-move agents pay basal metabolism, while movement opportunities still
  govern some other stress/foraging costs. Fast agents can incur multiple costs.
- Toxicity, climate, recovery, mating, and mutation coefficients are heuristic.
- Multiple diversity/endpoint summaries are descriptive, without correction
  for multiple statistical comparisons. Ten seeds is a reporting convention,
  not proof of adequate statistical power.
- No single combined “resilience score” is reported; resistance, recovery,
  diversity, and persistence are different properties. Sparse snapshots cannot
  determine an exact recovery time.
