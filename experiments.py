"""Run auditable control/treatment comparisons, paired by independent RNG seeds.

Uses only the Python standard library. Each seed contributes one pair to the
bootstrap; snapshots within a trajectory are never treated as replicates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
from subprocess import run as run_process
import sys
import time
from typing import Any


METRICS = ("final_population", "mean_population", "persistence_ticks", "final_effective_species")
UINT64_MAX = 2**64 - 1
PARAMETERS = ("shock-strength", "resource-recovery")


@dataclass(frozen=True)
class ExperimentConfig:
    simulator: Path
    output_dir: Path
    parameter: str
    control: float
    treatment: float
    seeds: tuple[int, ...]
    agents: int = 400
    ticks: int = 600
    snapshot: int = 50
    width: int = 64
    height: int = 64
    predator_ratio: float = 0.05
    timeout: float = 120.0
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 1729


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def _number(value: Any, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite numeric data")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse unique unsigned 64-bit seed IDs without float conversion."""
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not re.fullmatch(r"[0-9]+", token) for token in tokens):
        raise ValueError("seeds must be comma-separated unsigned integers")
    seeds = tuple(_integer(int(token), "seed", maximum=UINT64_MAX) for token in tokens)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique; repeated seeds are not independent replicates")
    return seeds


def _finite_json(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report contains nonfinite data")


def validate_report(data: Any, *, expected_seed: int,
                    expected_config: dict[str, Any]) -> dict[str, float]:
    """Require a complete schema-v2 trajectory and internally consistent endpoints."""
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 2:
        raise ValueError("report must use schema_version 2")
    _finite_json(data)
    if _integer(data.get("seed"), "seed", maximum=UINT64_MAX) != expected_seed:
        raise ValueError("report seed does not match requested seed")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError("report is missing configuration")
    if "seed" in config and _integer(config["seed"], "config seed", maximum=UINT64_MAX) != expected_seed:
        raise ValueError("configuration seed does not match requested seed")
    for name, expected in expected_config.items():
        actual = config.get(name)
        if type(expected) is int:
            matches = type(actual) is int and actual == expected
        elif type(expected) is float:
            matches = type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"report configuration mismatch for {name}: expected {expected}, got {actual}")
    horizon = _integer(config.get("simulation_ticks"), "simulation_ticks", 1)
    capacity = _integer(config.get("max_agents"), "max_agents", 1)
    initial = _integer(config.get("initial_agents"), "initial_agents", 1, capacity)
    run = data.get("run")
    ticks = data.get("ticks")
    if not isinstance(run, dict) or not isinstance(ticks, list) or len(ticks) < 2:
        raise ValueError("report requires run metadata and initial/final endpoints")
    completed = _integer(run.get("ticks_completed"), "ticks_completed", 1, horizon)
    final = _integer(run.get("final_population"), "final_population", 0, capacity)
    births = _integer(run.get("total_births"), "total_births")
    deaths = _integer(run.get("total_deaths"), "total_deaths")
    integral = _integer(run.get("population_time_integral"), "population_time_integral", 0, capacity * completed)
    extinct = run.get("extinction_tick")
    if run.get("status") == "completed":
        if completed != horizon or extinct is not None or final == 0:
            raise ValueError("completed report has inconsistent final status")
    elif run.get("status") == "extinct":
        if _integer(extinct, "extinction_tick", 1, horizon) != completed or final != 0:
            raise ValueError("extinct report has inconsistent extinction endpoint")
    else:
        raise ValueError("run status must be completed or extinct")
    if initial + births - deaths != final:
        raise ValueError("run population does not balance initial population, births and deaths")
    previous_tick = -1
    for snapshot in ticks:
        if not isinstance(snapshot, dict):
            raise ValueError("every snapshot must be an object")
        tick = _integer(snapshot.get("tick"), "snapshot tick", 0, completed)
        population = _integer(snapshot.get("population"), "snapshot population", 0, capacity)
        herbivores = _integer(snapshot.get("herbivore_count"), "herbivore_count")
        predators = _integer(snapshot.get("predator_count"), "predator_count")
        richness = _integer(snapshot.get("species_count"), "species_count", 0, population)
        if tick <= previous_tick or herbivores + predators != population:
            raise ValueError("snapshot order or population composition is inconsistent")
        if population == 0 and tick != completed:
            raise ValueError("empty population before terminal snapshot")
        _number(snapshot.get("mean_resources"), "mean_resources", 0)
        _number(snapshot.get("mean_habitat_match"), "mean_habitat_match", 0, 1.00001)
        for name in ("effective_species", "inverse_simpson"):
            diversity = _number(snapshot.get(name), name, 0, richness + 1e-5)
            if population > 0 and diversity < 1 - 1e-5:
                raise ValueError(f"nonempty population requires {name} >= 1")
            if population == 0 and diversity != 0:
                raise ValueError(f"empty population requires {name} == 0")
        previous_tick = tick
    if ticks[0]["tick"] != 0 or ticks[0]["population"] != initial:
        raise ValueError("report is missing its initial endpoint")
    if ticks[-1]["tick"] != completed or ticks[-1]["population"] != final:
        raise ValueError("report is missing its final endpoint")
    if integral < final:
        raise ValueError("population integral cannot be smaller than the final population")
    return {
        "final_population": final,
        "mean_population": integral / horizon,
        "persistence_ticks": extinct if extinct is not None else horizon,
        "final_effective_species": ticks[-1]["effective_species"],
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize_pairs(rows: list[dict[str, Any]], *, bootstrap_samples: int = 5000,
                    bootstrap_seed: int = 1729) -> dict[str, Any]:
    """Percentile bootstrap of mean treatment-minus-control deltas across seed pairs."""
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    _integer(bootstrap_seed, "bootstrap_seed", maximum=UINT64_MAX)
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = _integer(row.get("seed"), "row seed", maximum=UINT64_MAX)
        arm = row.get("arm")
        if arm not in ("control", "treatment") or arm in pairs.setdefault(seed, {}):
            raise ValueError("each seed must have exactly one control and one treatment")
        for metric in METRICS:
            _number(row.get(metric), metric)
        pairs[seed][arm] = row
    if not pairs or any(set(pair) != {"control", "treatment"} for pair in pairs.values()):
        raise ValueError("each seed must have exactly one control and one treatment")
    ordered = [pairs[seed] for seed in sorted(pairs)]
    count = len(ordered)
    deltas = {name: [pair["treatment"][name] - pair["control"][name] for pair in ordered]
              for name in METRICS}
    bootstrap: dict[str, list[float]] = {name: [] for name in METRICS}
    if count >= 2:
        rng = random.Random(bootstrap_seed)
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(count) for _ in range(count)]
            for name in METRICS:
                bootstrap[name].append(math.fsum(deltas[name][i] for i in indices) / count)
    metrics = {}
    for name in METRICS:
        samples = sorted(bootstrap[name])
        metrics[name] = {
            "control_mean": math.fsum(pair["control"][name] for pair in ordered) / count,
            "treatment_mean": math.fsum(pair["treatment"][name] for pair in ordered) / count,
            "paired_delta_mean": math.fsum(deltas[name]) / count,
            "paired_delta_ci95": [_percentile(samples, .025), _percentile(samples, .975)] if samples else None,
        }
    status = "insufficient_replication" if count < 2 else "exploratory" if count < 10 else "replicated"
    notes = [
        "Deltas are treatment minus control. Seed pairs, not snapshots or agents, are the sampling units.",
        "Same seeds match initial conditions. Branching trajectories consume random draws differently; "
        "this is not aligned common random numbers after branching.",
        "Mean population is the sum of post-step populations divided by requested ticks, including "
        "zeros after extinction. It is not the mean of saved snapshots.",
        "Persistence is restricted to the requested horizon; surviving runs are right-censored there.",
        "Percentile intervals describe seed variability for this executable and configuration. "
        "They are not guarantees, multiple-comparison-adjusted tests, or evidence about real ecosystems.",
    ]
    if count < 2:
        notes.append("Insufficient replication: one seed pair cannot estimate between-seed uncertainty; CI is null.")
    elif count < 10:
        notes.append("Exploratory: fewer than 10 independent seed pairs give unstable uncertainty estimates.")
    return {
        "n_seed_pairs": count, "replication_status": status, "metrics": metrics,
        "bootstrap": {"method": "paired_percentile", "confidence_level": .95,
                      "resamples": bootstrap_samples, "seed": bootstrap_seed,
                      "sampling_unit": "seed_pair", "quantile_interpolation": "linear"},
        "notes": notes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_config(config: ExperimentConfig) -> None:
    if config.parameter not in PARAMETERS:
        raise ValueError(f"parameter must be one of {PARAMETERS}")
    for name in ("control", "treatment"):
        _number(getattr(config, name), name, 0, 1)
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a nonempty collection of unique integers")
    for seed in config.seeds:
        _integer(seed, "seed", maximum=UINT64_MAX)
    # This runner uses the simulator's default population cap of 6000.
    _integer(config.agents, "agents", 1, 6000)
    for name in ("ticks", "snapshot"):
        _integer(getattr(config, name), name, 1, 10000000)
    for name in ("width", "height"):
        _integer(getattr(config, name), name, 2, 512)
    _integer(config.bootstrap_samples, "bootstrap_samples", 1)
    _integer(config.bootstrap_seed, "bootstrap_seed", maximum=UINT64_MAX)
    _number(config.predator_ratio, "predator_ratio", 0, 1)
    if _number(config.timeout, "timeout", 0) == 0:
        raise ValueError("timeout must be positive")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"invalid JSON duplicate key: {key}")
        result[key] = value
    return result


def _run_arm(config: ExperimentConfig, simulator: dict[str, str], output_dir: Path,
             seed: int, arm: str, value: float,
             shared_config: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = output_dir / f"seed_{seed}" / arm
    directory.mkdir(parents=True)
    report_path = directory / "simulation_summary.json"
    expected = {"width": config.width, "height": config.height, "initial_agents": config.agents,
                "simulation_ticks": config.ticks, "snapshot_interval": config.snapshot,
                "predator_ratio": float(config.predator_ratio), config.parameter.replace("-", "_"): float(value)}
    command = [simulator["path"], "--seed", str(seed), "--output", str(report_path),
               "--agents", str(config.agents), "--ticks", str(config.ticks),
               "--snapshot", str(config.snapshot), "--width", str(config.width),
               "--height", str(config.height), "--predator-ratio", str(config.predator_ratio),
               "--" + config.parameter, str(value)]
    receipt: dict[str, Any] = {
        "schema_version": 1, "seed": seed, "arm": arm, "parameter": config.parameter,
        "parameter_value": value, "command": command, "cwd": str(directory),
        "simulator": simulator, "expected_config": expected,
        "started_at": _now(), "status": "running", "timeout_seconds": config.timeout,
        "report": {"path": str(report_path)},
    }
    receipt_path = directory / "receipt.json"
    _write_json(receipt_path, receipt)
    start = time.monotonic()
    try:
        if _sha256(Path(simulator["path"])) != simulator["sha256"]:
            raise ValueError("simulator executable changed during experiment")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = run_process(command, cwd=directory, stdout=stdout, stderr=stderr,
                                  timeout=config.timeout, check=False)
        receipt["returncode"] = process.returncode
        if process.returncode:
            raise ValueError(f"simulator exited with code {process.returncode}")
        if _sha256(Path(simulator["path"])) != simulator["sha256"]:
            raise ValueError("simulator executable changed during experiment")
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                              object_pairs_hook=_unique_object)
            _finite_json(data)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"invalid or missing simulator JSON: {error}") from error
        if isinstance(data, dict):
            receipt["reported_config"] = data.get("config")
        metrics = validate_report(data, expected_seed=seed, expected_config=expected)
        common = {key: item for key, item in data["config"].items()
                  if key not in ("seed", config.parameter.replace("-", "_"))}
        if shared_config is not None and common != shared_config:
            raise ValueError("reported configuration changed outside the requested intervention")
        receipt["status"] = "completed"
        receipt["run"] = data["run"]
        receipt["metrics"] = metrics
        row = {"seed": seed, "arm": arm, "parameter_value": value, **metrics,
               "report_file": str(report_path.relative_to(output_dir)),
               "report_sha256": _sha256(report_path),
               "receipt_file": str(receipt_path.relative_to(output_dir))}
        return row, common
    except subprocess.TimeoutExpired as error:
        receipt.update(status="failed", error=f"simulator timed out after {config.timeout} seconds")
        raise ValueError(receipt["error"]) from error
    except (OSError, ValueError) as error:
        receipt.update(status="failed", error=str(error))
        raise ValueError(str(error)) from error
    finally:
        receipt["finished_at"] = _now()
        receipt["elapsed_seconds"] = time.monotonic() - start
        for name in ("simulation_summary.json", "stdout.log", "stderr.log"):
            path = directory / name
            if path.is_file():
                artifact = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                receipt["report" if name == "simulation_summary.json" else name] = artifact
        _write_json(receipt_path, receipt)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute all requested arms; retain failed receipts and never summarize partial pairs."""
    _validate_config(config)
    executable = Path(config.simulator).resolve()
    if not executable.is_file():
        raise ValueError(f"simulator executable does not exist: {executable}")
    simulator = {"path": str(executable), "sha256": _sha256(executable)}
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty; previous experiments are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment.json"
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_at": _now(), "simulator": simulator,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)),
                   "python": sys.version, "platform": platform.platform()},
        "request": {**asdict(config), "simulator": str(executable), "output_dir": str(output_dir)},
        "completed_arms": 0,
    }
    # Exclusive reservation also prevents two runners from claiming the same empty directory.
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    rows = []
    common = None
    try:
        for seed in config.seeds:
            for arm, value in (("control", config.control), ("treatment", config.treatment)):
                row, common = _run_arm(config, simulator, output_dir, seed, arm, value, common)
                rows.append(row)
                _write_csv(output_dir / "rows.csv", rows)
                manifest["completed_arms"] = len(rows)
                _write_json(manifest_path, manifest)
        result = summarize_pairs(rows, bootstrap_samples=config.bootstrap_samples,
                                 bootstrap_seed=config.bootstrap_seed)
        deltas = []
        for index in range(0, len(rows), 2):
            control, treatment = rows[index:index + 2]
            deltas.append({"seed": control["seed"],
                           **{name: treatment[name] - control[name] for name in METRICS}})
        _write_csv(output_dir / "paired_deltas.csv", deltas)
        result.update(schema_version=1, status="completed", simulator=simulator,
                      parameter=config.parameter, control=config.control, treatment=config.treatment,
                      seeds=list(config.seeds), common_reported_config=common,
                      artifacts={name: {"path": name, "sha256": _sha256(output_dir / name)}
                                 for name in ("rows.csv", "paired_deltas.csv")})
        _write_json(output_dir / "summary.json", result)
        manifest.update(status="completed", summary_sha256=_sha256(output_dir / "summary.json"))
        return result
    except (OSError, ValueError) as error:
        manifest.update(status="failed", error=str(error))
        raise ValueError(str(error)) from error
    finally:
        manifest["finished_at"] = _now()
        _write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True, help="Path to the simulator executable")
    parser.add_argument("--parameter", choices=PARAMETERS, required=True)
    parser.add_argument("--control", type=float, required=True, help="Control parameter value in [0, 1]")
    parser.add_argument("--treatment", type=float, required=True, help="Treatment parameter value in [0, 1]")
    parser.add_argument("--seeds", type=parse_seeds, required=True, help="Unique uint64 seeds separated by commas")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty results directory")
    for name, default in (("agents", 400), ("ticks", 600), ("snapshot", 50), ("width", 64), ("height", 64)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--predator-ratio", type=float, default=.05)
    parser.add_argument("--timeout", type=float, default=120, help="Per-arm timeout in seconds")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    arguments = parser.parse_args(argv)
    try:
        result = run_experiment(ExperimentConfig(**vars(arguments)))
    except (OSError, ValueError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1
    print(f"Completed {result['n_seed_pairs']} seed pairs ({result['replication_status']}).")
    print(f"Summary: {(arguments.output_dir / 'summary.json').resolve()}")
    for name, metric in result["metrics"].items():
        print(f"  {name}: treatment - control = {metric['paired_delta_mean']:.6g}; "
              f"95% paired CI = {metric['paired_delta_ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
