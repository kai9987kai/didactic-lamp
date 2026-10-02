"""Behavioral checks for paired experiments; synthetic outputs are not evidence of ecology."""

import copy
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def report(seed=7, value=0.2, final=20, integral=150):
    config = {
        "width": 16, "height": 16, "initial_agents": 10, "max_agents": 6000,
        "simulation_ticks": 10, "snapshot_interval": 5, "classification_interval": 25,
        "seed": seed, "predator_ratio": 0.05, "shock_strength": value,
        "resource_recovery": 0.002,
    }

    def snapshot(tick, population):
        return {
            "tick": tick, "population": population, "herbivore_count": population,
            "predator_count": 0, "mean_resources": 0.5, "mean_habitat_match": 0.5,
            "species_count": 2 if population else 0,
            "effective_species": 2.0 if population else 0.0,
            "inverse_simpson": 2.0 if population else 0.0,
        }

    return {
        "schema_version": 2, "seed": seed, "config": config,
        "ticks": [snapshot(0, 10), snapshot(10, final)],
        "run": {"status": "completed", "ticks_completed": 10,
                "extinction_tick": None, "final_population": final,
                "total_births": max(0, final - 10), "total_deaths": max(0, 10 - final),
                "population_time_integral": integral},
    }


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.exp = importlib.import_module("experiments")
        except ModuleNotFoundError:
            self.fail("The paired experiment runner has not been implemented")

    def config(self, directory, **kwargs):
        fields = dict(simulator=Path(sys.executable), output_dir=Path(directory) / "experiment",
                      parameter="shock-strength", control=0.2, treatment=0.4,
                      seeds=(7, 9), agents=10, ticks=10, snapshot=5,
                      width=16, height=16, predator_ratio=0.05,
                      bootstrap_samples=1000, bootstrap_seed=3)
        fields.update(kwargs)
        return self.exp.ExperimentConfig(**fields)

    def test_seed_list_preserves_uint64_and_rejects_duplicate_or_invalid_seeds(self):
        self.assertEqual(self.exp.parse_seeds("0, 18446744073709551615"),
                         (0, 18446744073709551615))
        for value in ("", "1,1", "-1", "18446744073709551616", "1.5", "1,,2", "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.exp.parse_seeds(value)

    def test_population_mean_uses_integral_not_sparse_snapshot_mean(self):
        data = report(integral=123)
        metrics = self.exp.validate_report(data, expected_seed=7, expected_config=data["config"])
        self.assertEqual(metrics, {"final_population": 20, "mean_population": 12.3,
                                   "persistence_ticks": 10, "final_effective_species": 2.0})

    def test_resource_stock_above_one_is_valid_and_seed_can_be_top_level_only(self):
        data = report()
        del data["config"]["seed"]
        data["ticks"][-1]["mean_resources"] = 1.6
        metrics = self.exp.validate_report(data, expected_seed=7, expected_config=data["config"])
        self.assertEqual(metrics["final_population"], 20)

    def test_extinction_counts_zero_population_for_remaining_horizon(self):
        data = report(final=0, integral=25)
        data["run"].update(status="extinct", ticks_completed=4, extinction_tick=4)
        data["ticks"][-1]["tick"] = 4
        metrics = self.exp.validate_report(data, expected_seed=7, expected_config=data["config"])
        self.assertEqual(metrics["mean_population"], 2.5)
        self.assertEqual(metrics["persistence_ticks"], 4)
        self.assertEqual(metrics["final_effective_species"], 0)

    def test_incomplete_or_inconsistent_reports_are_rejected(self):
        changes = [
            lambda d: d.update(schema_version=1),
            lambda d: d.update(seed=8),
            lambda d: d["config"].update(simulation_ticks=9),
            lambda d: d["ticks"].pop(),
            lambda d: d["ticks"][0].update(tick=1),
            lambda d: d["ticks"][-1].pop("effective_species"),
            lambda d: d["ticks"][-1].update(effective_species=math.nan),
            lambda d: d["ticks"][-1].update(population=21),
            lambda d: d["run"].update(final_population=19),
            lambda d: d["run"].update(ticks_completed=9),
            lambda d: d["run"].update(status="extinct", extinction_tick=10),
            lambda d: d["run"].update(population_time_integral=-1),
            lambda d: d["run"].update(population_time_integral=10**400),
            lambda d: d["run"].update(population_time_integral=60001),
            lambda d: d["run"].update(total_births=0),
        ]
        for change in changes:
            data = report()
            expected = copy.deepcopy(data["config"])
            change(data)
            with self.subTest(report=data), self.assertRaises(ValueError):
                self.exp.validate_report(data, expected_seed=7, expected_config=expected)

    def test_identical_seed_pairs_have_exact_zero_interval_despite_between_seed_spread(self):
        rows = []
        for seed, value in ((1, 1), (2, 1000), (3, 100000)):
            for arm in ("control", "treatment"):
                rows.append(dict(seed=seed, arm=arm, **{name: value for name in self.exp.METRICS}))
        summary = self.exp.summarize_pairs(rows, bootstrap_samples=1000, bootstrap_seed=4)
        for metric in summary["metrics"].values():
            self.assertEqual(metric["paired_delta_mean"], 0)
            self.assertEqual(metric["paired_delta_ci95"], [0, 0])
        self.assertEqual(summary["replication_status"], "exploratory")

    def test_constant_treatment_effect_is_preserved_when_resampling_seed_pairs(self):
        rows = []
        for seed, value in ((1, 1), (2, 1000), (3, 100000)):
            for arm, offset in (("control", 0), ("treatment", 3)):
                rows.append(dict(seed=seed, arm=arm, **{name: value + offset for name in self.exp.METRICS}))
        result = self.exp.summarize_pairs(rows, bootstrap_samples=1000, bootstrap_seed=4)
        for metric in result["metrics"].values():
            self.assertEqual(metric["paired_delta_mean"], 3)
            self.assertEqual(metric["paired_delta_ci95"], [3, 3])

    def test_single_seed_has_no_confidence_interval(self):
        rows = [dict(seed=7, arm=arm, **{name: value for name in self.exp.METRICS})
                for arm, value in (("control", 1), ("treatment", 4))]
        result = self.exp.summarize_pairs(rows, bootstrap_samples=1000, bootstrap_seed=4)
        self.assertEqual(result["replication_status"], "insufficient_replication")
        self.assertTrue(all(m["paired_delta_ci95"] is None for m in result["metrics"].values()))

    def test_bootstrap_is_reproducible_and_uses_seed_pair_count(self):
        rows = [dict(seed=seed, arm=arm, **{name: value for name in self.exp.METRICS})
                for seed, delta in ((1, -2), (2, 0), (3, 4), (4, 8))
                for arm, value in (("control", 100), ("treatment", 100 + delta))]
        first = self.exp.summarize_pairs(rows, bootstrap_samples=2000, bootstrap_seed=4)
        second = self.exp.summarize_pairs(list(reversed(rows)), bootstrap_samples=2000, bootstrap_seed=4)
        self.assertEqual(first, second)
        self.assertEqual(first["n_seed_pairs"], 4)
        self.assertEqual(first["metrics"]["mean_population"]["paired_delta_mean"], 2.5)
        low, high = first["metrics"]["mean_population"]["paired_delta_ci95"]
        self.assertLess(low, 2.5)
        self.assertGreater(high, 2.5)
        self.assertGreaterEqual(low, -2)
        self.assertLessEqual(high, 8)

    def test_missing_or_duplicate_arms_and_nonfinite_values_fail_summary(self):
        row = dict(seed=1, arm="control", **{name: 1 for name in self.exp.METRICS})
        for rows in ([], [row], [row, row], [row, dict(row, arm="treatment", mean_population=math.inf)],
                     [row, dict(row, arm="treatment", mean_population=10**400)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.exp.summarize_pairs(rows)

    def fake_simulator(self, command, **kwargs):
        args = dict(zip(command[1::2], command[2::2]))
        data = report(seed=int(args["--seed"]), value=float(args["--shock-strength"]))
        Path(args["--output"]).write_text(json.dumps(data), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def test_success_writes_raw_rows_and_hash_bound_receipts_for_every_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch("experiments.run_process", side_effect=self.fake_simulator):
                summary = self.exp.run_experiment(config)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["n_seed_pairs"], 2)
            self.assertEqual(summary["simulator"]["sha256"], hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
            with (config.output_dir / "rows.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(int(row["seed"]), row["arm"]) for row in rows],
                             [(7, "control"), (7, "treatment"), (9, "control"), (9, "treatment")])
            receipts = list(config.output_dir.glob("seed_*/*/receipt.json"))
            self.assertEqual(len(receipts), 4)
            for receipt_file in receipts:
                receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(receipt["reported_config"]["simulation_ticks"], 10)
                self.assertEqual(receipt["command"][0], str(Path(sys.executable).resolve()))
                raw = Path(receipt["report"]["path"]).read_bytes()
                self.assertEqual(receipt["report"]["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(receipt["simulator"]["sha256"], summary["simulator"]["sha256"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config.output_dir.mkdir()
            sentinel = config.output_dir / "keep.txt"
            sentinel.write_text("existing research", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.exp.run_experiment(config)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing research")

    def test_invalid_inputs_fail_before_creating_output(self):
        for fields in ({"control": math.nan}, {"control": 10**400}, {"treatment": 1.1}, {"seeds": (1, 1)},
                       {"seeds": (-1,)}, {"ticks": 0}, {"timeout": math.inf},
                       {"predator_ratio": -0.1}, {"bootstrap_samples": 0},
                       {"width": 1}, {"height": 513}, {"ticks": 10000001},
                       {"snapshot": 10000001}, {"agents": 6001}):
            with tempfile.TemporaryDirectory() as directory, self.subTest(fields=fields):
                config = self.config(directory, **fields)
                with self.assertRaises(ValueError):
                    self.exp.run_experiment(config)
                self.assertFalse(config.output_dir.exists())

    def test_invalid_json_keeps_failure_receipt_without_summary(self):
        def malformed(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text("{broken", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        self.check_failed_run(malformed, "invalid")

    def test_nonfinite_report_configuration_preserves_failed_receipt(self):
        def nonfinite(command, **kwargs):
            result = self.fake_simulator(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            data = json.loads(path.read_text(encoding="utf-8"))
            data["config"]["extra_setting"] = 1
            path.write_text(json.dumps(data).replace('"extra_setting": 1', '"extra_setting": 1e999'),
                            encoding="utf-8")
            return result

        self.check_failed_run(nonfinite, "nonfinite")

    def test_missing_report_preserves_failed_receipt(self):
        self.check_failed_run(lambda command, **kwargs: subprocess.CompletedProcess(command, 0), "missing")

    def test_wrong_config_seed_is_rejected_even_if_top_level_seed_matches(self):
        def wrong_seed(command, **kwargs):
            result = self.fake_simulator(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            data = json.loads(path.read_text(encoding="utf-8"))
            data["config"]["seed"] = 12345
            path.write_text(json.dumps(data), encoding="utf-8")
            return result

        self.check_failed_run(wrong_seed, "seed")

    def test_cli_rejects_nonfinite_parameter_before_any_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            result = subprocess.run(
                [sys.executable, str(Path(self.exp.__file__)), "--simulator", sys.executable,
                 "--parameter", "resource-recovery", "--control", "nan", "--treatment", "0.01",
                 "--seeds", "1,2", "--output-dir", str(output)],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("finite", result.stderr)
            self.assertFalse(output.exists())

    def test_real_non_simulator_process_failure_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with self.assertRaisesRegex(ValueError, "exited with code"):
                self.exp.run_experiment(config)
            self.assertFalse((config.output_dir / "summary.json").exists())
            receipt = json.loads(next(config.output_dir.glob("seed_*/*/receipt.json")).read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertNotEqual(receipt["returncode"], 0)

    def test_recovery_intervention_is_passed_and_reported_in_each_arm(self):
        def recovery(command, **kwargs):
            args = dict(zip(command[1::2], command[2::2]))
            data = report(seed=int(args["--seed"]))
            data["config"]["resource_recovery"] = float(args["--resource-recovery"])
            Path(args["--output"]).write_text(json.dumps(data), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, parameter="resource-recovery", control=0, treatment=0.01)
            with patch("experiments.run_process", side_effect=recovery):
                result = self.exp.run_experiment(config)
            self.assertEqual(result["parameter"], "resource-recovery")
            self.assertEqual(result["control"], 0)
            self.assertEqual(result["treatment"], 0.01)
            for arm, expected in (("control", 0), ("treatment", 0.01)):
                receipt = json.loads((config.output_dir / "seed_7" / arm / "receipt.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["reported_config"]["resource_recovery"], expected)

    def test_all_predator_initial_condition_is_accepted(self):
        def all_predators(command, **kwargs):
            result = self.fake_simulator(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            data = json.loads(path.read_text(encoding="utf-8"))
            data["config"]["predator_ratio"] = 1.0
            for snapshot in data["ticks"]:
                snapshot.update(herbivore_count=0, predator_count=snapshot["population"])
            path.write_text(json.dumps(data), encoding="utf-8")
            return result

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, predator_ratio=1)
            with patch("experiments.run_process", side_effect=all_predators):
                result = self.exp.run_experiment(config)
            self.assertEqual(result["status"], "completed")

    def test_simulator_double_does_not_intercept_runtime_provenance_queries(self):
        def runtime_probe():
            process = subprocess.run([sys.executable, "-c", "print('runtime-probe')"],
                                     capture_output=True, text=True, timeout=10, check=True)
            return process.stdout.strip()

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch("experiments.run_process", side_effect=self.fake_simulator), \
                    patch("experiments.platform.platform", side_effect=runtime_probe):
                self.exp.run_experiment(config)
            manifest = json.loads((config.output_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runner"]["platform"], "runtime-probe")

    def test_nonzero_exit_keeps_failure_receipt_without_summary(self):
        self.check_failed_run(lambda command, **kwargs: subprocess.CompletedProcess(command, 3), "3")

    def test_binary_replaced_during_execution_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = Path(directory) / "simulator.exe"
            simulator.write_bytes(b"original simulator")
            config = self.config(directory, simulator=simulator)

            def replacement(command, **kwargs):
                result = self.fake_simulator(command, **kwargs)
                simulator.write_bytes(b"different simulator")
                return result

            with patch("experiments.run_process", side_effect=replacement), \
                    self.assertRaisesRegex(ValueError, "executable changed"):
                self.exp.run_experiment(config)
            receipt = json.loads(next(config.output_dir.glob("seed_*/*/receipt.json")).read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["simulator"]["sha256"], hashlib.sha256(b"original simulator").hexdigest())
            self.assertFalse((config.output_dir / "summary.json").exists())

    def test_timeout_keeps_failure_receipt_without_summary(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        self.check_failed_run(timeout, "timed out")

    def test_configuration_drift_between_arms_fails_experiment(self):
        def drift(command, **kwargs):
            result = self.fake_simulator(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            data = json.loads(path.read_text(encoding="utf-8"))
            if float(command[command.index("--shock-strength") + 1]) == 0.4:
                data["config"]["classification_interval"] = 100
            path.write_text(json.dumps(data), encoding="utf-8")
            return result

        self.check_failed_run(drift, "configuration")

    def check_failed_run(self, simulate, message):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch("experiments.run_process", side_effect=simulate):
                with self.assertRaisesRegex(ValueError, message):
                    self.exp.run_experiment(config)
            self.assertFalse((config.output_dir / "summary.json").exists())
            manifest = json.loads((config.output_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            receipts = [json.loads(p.read_text(encoding="utf-8"))
                        for p in config.output_dir.glob("seed_*/*/receipt.json")]
            self.assertTrue(any(r["status"] == "failed" for r in receipts))


if __name__ == "__main__":
    unittest.main()
