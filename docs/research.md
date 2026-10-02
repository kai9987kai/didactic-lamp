# Research-informed upgrade

Reviewed for this upgrade, October 2026. These primary sources guide design
and interpretation; they do not validate the simulator's ecological rules.

| Source | Application here | Boundary |
| --- | --- | --- |
| Grimm et al. (2020), [ODD protocol, second update](https://www.jasss.org/23/2/7.html) | Explicit state, initialization, transition schedule, configuration, and model limits in `model.md`. | Reporting improves inspectability, not biological validity. |
| Klein et al. (2024), [Noise-free comparison of stochastic agent-based simulations using common random numbers](https://arxiv.org/html/2409.02086v2) | Fixed seed pairs, saved raw trajectories, provenance, and an explicit warning about draw misalignment. | The preprint's fully aligned algorithm is not implemented. A shared seed alone does not maintain alignment after branching. |
| Jost (2006), [Entropy and diversity](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/j.2006.0030-1299.14714.x) | Effective Shannon species and inverse Simpson diversity in the same units as richness. | These summarize algorithmic cluster abundances, not verified biological species. |
| Barrere et al. (2024), [Forest storm resilience depends on functional composition and climate](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2435.14489) | Explicit disturbance windows and treatment/control trajectories. | No forest coefficients, validated resilience prediction, or inferred recovery times are imported. |
| Runge et al. (2025), [Monitoring terrestrial ecosystem resilience: consensus and limitations across metrics](https://onlinelibrary.wiley.com/doi/10.1111/gcb.70115) | Separate demographic persistence, resources, and diversity instead of combining them into one score. | Different resilience measurements need not agree; snapshot trajectories are descriptive. |

The implementation choices are our inferences from these sources. The recovery
term is an explicit toy-model extension intended for parameter experiments,
not a formula claimed from the papers. Full-genome distances, live occupancy,
bounded reproduction, and exact final output address directly observed code
problems. No claim is made that these changes universally increase survival.

Potential future work includes stable agent/tick/decision random streams,
conserved energy accounting, exact continuation checkpoints, and a calibrated
disturbance-response analysis with explicit recovery censoring. These remain
future work rather than implied capabilities of v5.
