"""Integration invariants for the real compiled simulator (SIM_BINARY)."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BINARY = Path(os.environ.get("SIM_BINARY", "build-research/universe_sim.exe")).resolve()


class SimulatorContract(unittest.TestCase):
    def run_sim(self, directory, name="run", **options):
        path = Path(directory) / f"{name}.json"
        args = {"agents": 30, "ticks": 40, "width": 16, "height": 16,
                "snapshot": 13, "seed": 7, **options}
        command = [str(BINARY), "--output", str(path)]
        for key, value in args.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(path.read_text()), path.read_bytes()

    def test_initial_final_and_demographic_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = self.run_sim(directory)
        self.assertEqual(data["schema_version"], 2)
        ticks = data["ticks"]
        self.assertEqual(ticks[0]["tick"], 0)
        self.assertEqual(ticks[0]["population"], 30)
        self.assertEqual(ticks[0]["mean_age"], 0)
        self.assertEqual(ticks[-1]["tick"], data["run"]["ticks_completed"])
        for previous, current in zip(ticks, ticks[1:]):
            self.assertEqual(current["population"], previous["population"] +
                             current["births"] - current["deaths"])
        self.assertEqual(data["run"]["final_population"], 30 +
                         data["run"]["total_births"] - data["run"]["total_deaths"])

    def test_repeatability_and_observation_does_not_change_dynamics(self):
        with tempfile.TemporaryDirectory() as directory:
            first, raw1 = self.run_sim(directory, "first", snapshot=1)
            _, raw2 = self.run_sim(directory, "repeat", snapshot=1)
            coarse, _ = self.run_sim(directory, "coarse", snapshot=17)
        self.assertEqual(raw1, raw2)
        self.assertEqual(first["run"], coarse["run"])
        self.assertEqual(first["species_records"], coarse["species_records"])
        for key in ("population", "mean_fitness", "effective_species", "mean_resources"):
            self.assertEqual(first["ticks"][-1][key], coarse["ticks"][-1][key])

    def test_extinction_keeps_world_and_exact_final_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = self.run_sim(directory, agents=2, predator_ratio=1,
                                   ticks=800, snapshot=137, shock_strength=0)
        self.assertEqual(data["run"]["status"], "extinct")
        self.assertEqual(data["ticks"][-1]["population"], 0)
        self.assertEqual(data["ticks"][-1]["tick"], data["run"]["extinction_tick"])
        self.assertGreater(data["ticks"][-1]["mean_resources"], 0)
        self.assertAlmostEqual(sum(data["ticks"][-1]["biome_distribution"].values()), 1)
        self.assertTrue(all(r["tick_extinct"] >= 0 for r in data["species_records"]))

    def test_uint64_seed_and_exact_event_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = self.run_sim(directory, seed=18446744073709551615,
                                   shock_interval=10, shock_duration=3, ticks=25)
        self.assertEqual(data["seed"], 18446744073709551615)
        self.assertEqual([(e["start_tick"], e["end_tick_exclusive"]) for e in data["event_windows"]],
                         [(10, 13), (20, 23)])

    def test_invalid_parameters_fail_without_output(self):
        invalid = [("--ticks", "10oops"), ("--agents", "-1"),
                   ("--shock-strength", "nan"), ("--temperature", "inf"),
                   ("--predator-ratio", "1.1"), ("--seed", "-1"),
                   ("--seed", "18446744073709551616"), ("--width", "1"),
                   ("--resource-recovery", "-0.1")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            for flag, value in invalid:
                with self.subTest(flag=flag, value=value):
                    p = subprocess.run([str(BINARY), flag, value, "--output", str(output)],
                                       capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertFalse(output.exists())

    def test_output_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p = subprocess.run([str(BINARY), "--ticks", "1", "--agents", "2",
                                "--output", str(Path(directory) / "missing" / "run.json")],
                               capture_output=True, text=True, timeout=10)
            self.assertNotEqual(p.returncode, 0)
            self.assertNotIn("Wrote ", p.stdout)


if __name__ == "__main__":
    unittest.main()
