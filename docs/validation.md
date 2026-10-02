# Validation — 2 October 2026

Environment: Windows, GCC 15.2.0 through MSYS2 UCRT64, CMake/Ninja Release,
Python 3.12.10, Matplotlib 3.10.8.

## Automated verification

`ctest --test-dir build-research --output-on-failure` exercises three suites:

- Core C++ regressions: population cap, adult-only birth scheduling, affordable
  reproduction, equal cooldown, inherited provisional cluster, full-genome
  distance, trophic separation, single boundary hunting attempts, starvation,
  zero-move metabolism, live occupancy, and recovery enabled/disabled.
- C++ metrics regressions: live-only denominators, negative fitness, known
  diversity identities, extinct-world telemetry, and disabled event labels.
- 42 Python tests: six real-binary CLI contracts, 26 experiment runner checks,
  and ten dashboard checks including actual image export.

The CLI tests verify byte-identical repeated summaries, observation-interval
independence, demographic conservation, exact extinction endpoints, unsigned
64-bit seeds, sparse event schedules, invalid inputs, and output failure.
Runner tests cover complete-pair resampling and validation/failure receipts;
dashboard tests cover exact bands, legacy summaries, malformed inputs, irregular
intervals, comparisons, and headless PNG rendering.

Actual GUI-window opening was not exercised. Image export was visually checked;
the noninteractive `--show` error is tested. `git diff --check` passed.

## Paired recovery experiment

The final executable compared recovery 0 with 0.002 across seeds 1–10, using
400 initial agents, a 64×64 grid, 900 requested ticks, snapshots every 25 ticks,
and other v5 defaults. The bootstrap used 5,000 resamples and seed 1729.

| Endpoint | Control mean | Recovery mean | Paired change | 95% paired percentile interval |
| --- | ---: | ---: | ---: | ---: |
| Mean population over requested horizon | 73.2763 | 76.7503 | +3.4740 | [1.6514, 5.8911] |
| Restricted persistence (ticks) | 697.4 | 702.5 | +5.1 | [-90.7025, 90.6025] |
| Final population | 0.1 | 0.0 | -0.1 | [-0.3, 0.0] |
| Final effective species | 0.1 | 0.0 | -0.1 | [-0.3, 0.0] |

Recovery increased average population in this configuration, but the persistence
interval spans both negative and positive differences. All treatment runs were
extinct by the horizon; one control run still had one agent. These results do
not establish a survival benefit or universal ecological improvement.

An additional identical-arm check used seeds 21 and 22 with equal shock strength
0.18 in both arms. All four paired differences and intervals were exactly zero.
This checks pipeline identity, not statistical power or model realism.

## Local evidence

Generated artifacts are ignored by Git to avoid committing run output:

- `output/verified-recovery-study/summary.json`: paired results and CSV hashes.
- `output/verified-recovery-study/experiment.json`: full request and executable/
  runner identities.
- `output/verified-recovery-study/seed_*/{control,treatment}/`: raw summaries,
  command/configuration/hash receipts, and process logs for all 20 runs.
- `output/verified-recovery-comparison.png`: seed 7 comparison, control extinction
  at tick 594 and recovery extinction at tick 688; this seed is illustrative.
- `output/identical-arm-check/`: identity-check outputs.

Executable SHA-256:
`272392baa41dbcf4fe1d7e51e6dc7bb92fa8b74a0062b1d6d42bd8f43485083b`.

The raw receipt hashes describe the exact local artifacts. Rebuilding with
another compiler/runtime can change both binary and trajectory; preserve these
receipts when comparing results.
