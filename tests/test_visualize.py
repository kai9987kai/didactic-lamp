"""Dashboard contracts: exact event timing, faithful metrics and real PNG export."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualize


def summary(extinct=False):
    """Small, intentionally synthetic telemetry fixture for render validation."""
    rows = []
    for tick, population in ((0, 100), (100, 80), (175, 0 if extinct else 60)):
        richness = 0 if population == 0 else 3
        rows.append({
            "tick": tick, "population": population,
            "herbivore_count": population * 0.9, "predator_count": population * 0.1,
            "species_count": richness, "species_shannon": math.log(2) if richness else 0,
            "effective_species": 2 if richness else 0,
            "inverse_simpson": 1.8 if richness else 0,
            "species_evenness": 0.63 if richness else 0,
            "mean_fitness": tick / 100, "max_fitness": tick / 40,
            "mean_habitat_match": 0.75 if population else 0,
            "mean_resources": 0.4 + tick / 1000, "mean_toxicity": 0.08,
            "births": 0 if tick == 0 else 10, "deaths": 0 if tick == 0 else 30,
            "active_event": "None",
            "biome_distribution": {"Ocean": 0.2, "Tundra": 0.1, "Desert": 0.1,
                                   "Grassland": 0.3, "Forest": 0.2, "Jungle": 0.1},
        })
    return {
        "schema_version": 2, "seed": 7, "config": {"simulation_ticks": 200},
        "run": {"status": "extinct" if extinct else "completed", "ticks_completed": 175,
                "extinction_tick": 175 if extinct else None,
                "final_population": rows[-1]["population"], "total_births": 20, "total_deaths": 60},
        "event_windows": [
            {"type": "Drought", "start_tick": 35, "end_tick_exclusive": 50, "strength": 0.3},
            {"type": "Bloom", "start_tick": 150, "end_tick_exclusive": 190, "strength": 0.2},
            {"type": "ColdSnap", "start_tick": 200, "end_tick_exclusive": 215, "strength": 0.2},
        ],
        "ticks": rows,
    }


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ecosystem-visualize-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, data, name="summary.json"):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_exact_events_survive_sparse_snapshots_and_clip_at_final_tick(self):
        data = summary()
        events = visualize.event_windows(data)
        self.assertEqual([(e["type"], e["start_tick"], e["end_tick_exclusive"]) for e in events],
                         [("Drought", 35, 50), ("Bloom", 150, 175)])
        self.assertTrue(all(row["active_event"] == "None" for row in data["ticks"]))
        self.assertEqual(data["event_windows"][1]["end_tick_exclusive"], 190)
        plt = visualize._pyplot(False)
        fig, ax = plt.subplots()
        try:
            visualize.shade_events(ax, events)
            spans = [(p.get_x(), p.get_x() + p.get_width()) for p in ax.patches]
            self.assertEqual(spans, [(35, 50), (150, 175)])
            self.assertEqual(len(ax.texts), 0)
        finally:
            plt.close(fig)

    def test_explicit_empty_schedule_does_not_infer_legacy_events(self):
        data = summary()
        data["event_windows"] = []
        data["ticks"][0]["active_event"] = "Drought"
        self.assertEqual(visualize.event_windows(data), [])

    def test_legacy_summary_population_diversity_and_sampled_event_fallback(self):
        data = summary(extinct=True)
        del data["event_windows"]
        del data["run"]
        for row in data["ticks"]:
            del row["population"]
            del row["effective_species"]
            del row["inverse_simpson"]
        data["ticks"][0]["active_event"] = "Drought"
        data["ticks"][1]["active_event"] = "Bloom"
        loaded, rows = visualize.read_ticks(self.write(data))
        self.assertEqual(visualize.population_values(rows), [100, 80, 0])
        self.assertEqual(visualize.effective_species_values(rows), [2, 2, 0])
        self.assertEqual([(e["type"], e["start_tick"], e["end_tick_exclusive"])
                          for e in visualize.event_windows(loaded)],
                         [("Drought", 0, 100), ("Bloom", 100, 175)])
        output = visualize.create_visualizations(self.root / "summary.json", output_path=self.root / "legacy.png")
        self.assertGreater(output.stat().st_size, 10000)

    def test_demographic_bars_follow_irregular_intervals(self):
        centers, widths, births, deaths = visualize.demographic_intervals(summary()["ticks"])
        self.assertEqual(centers, [50, 137.5])
        self.assertEqual(widths, [88, 66])
        self.assertEqual(births, [10, 10])
        self.assertEqual(deaths, [-30, -30])
        self.assertTrue(all(c - w / 2 >= 0 for c, w in zip(centers, widths)))

    def test_invalid_empty_and_nonfinite_json_have_useful_errors(self):
        invalid = self.root / "broken.json"
        invalid.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            visualize.read_ticks(invalid)
        for data, message in (([], "JSON object"), ({}, "nonempty 'ticks'"),
                              ({"ticks": []}, "nonempty 'ticks'"),
                              ({"ticks": [{"tick": 1}, {"tick": 1}]}, "strictly increasing"),
                              ({"ticks": [{"tick": 1, "population": float('nan')}]}, "finite number")):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                visualize.read_ticks(self.write(data))
        with self.assertRaisesRegex(ValueError, "Cannot read simulation summary"):
            visualize.read_ticks(self.root / "missing.json")
        invalid_event = summary()
        invalid_event["event_windows"][0]["end_tick_exclusive"] = 0
        with self.assertRaisesRegex(ValueError, "0 <= start < end"):
            visualize.read_ticks(self.write(invalid_event))
        invalid_match = summary()
        invalid_match["ticks"][0]["mean_habitat_match"] = 2
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            visualize.read_ticks(self.write(invalid_match))

    def test_extinction_render_and_comparison_have_correct_units_and_series(self):
        primary, other = summary(extinct=True), summary()
        source = self.write(primary)
        comparison = self.write(other, "compare.json")
        plt = visualize._pyplot(False)
        existing = set(plt.get_fignums())
        for name, compare in (("primary.png", None), ("comparison.png", comparison)):
            with self.subTest(name=name):
                output = visualize.create_visualizations(source, compare_path=compare,
                                                         output_path=self.root / "exports" / name)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                image = plt.imread(output)
                self.assertGreater(image.shape[0], 1000)
                self.assertGreater(float(image.std()), 0.02)
                self.assertEqual(set(plt.get_fignums()), existing)
        fig = visualize._build_figure(plt, primary, primary["ticks"], other, other["ticks"])
        try:
            texts = "\n".join(text.get_text() for text in fig.texts)
            self.assertIn("seed 7", texts)
            self.assertIn("extinction at 175", texts)
            self.assertIn("ticks 175/200", texts)
            self.assertIn("Shared seed", texts)
            panels = fig.axes[:6]
            self.assertIn("B · total", panels[0].get_legend_handles_labels()[1])
            self.assertIn("B · resources", panels[2].get_legend_handles_labels()[1])
            self.assertIn("B · effective Shannon", panels[4].get_legend_handles_labels()[1])
            habitat_axis = next(ax for ax in fig.axes if ax.get_ylabel() == "Habitat match [0–1]")
            self.assertEqual(habitat_axis.get_ylim(), (0, 1))
            self.assertEqual(panels[4].get_ylabel(), "Species / effective species")
            self.assertEqual(panels[0].get_xticks()[-1], 175)
        finally:
            plt.close(fig)

    def test_cli_output_comparison_and_bad_input(self):
        source = self.write(summary())
        comparison = self.write(summary(extinct=True), "comparison.json")
        output = self.root / "chosen.png"
        script = Path(visualize.__file__).resolve()
        result = subprocess.run([sys.executable, str(script), str(source), "--compare", str(comparison),
                                 "--output", str(output)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.is_file())
        self.assertFalse((self.root / "simulation_dashboard.png").exists())
        result = subprocess.run([sys.executable, str(script), str(self.write({}))], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("nonempty 'ticks'", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_show_rejects_noninteractive_backend_instead_of_silently_doing_nothing(self):
        with patch.dict(os.environ, {"MPLBACKEND": "Agg"}):
            with self.assertRaisesRegex(ValueError, "interactive Matplotlib backend"):
                visualize.create_visualizations(self.write(summary()), show_plot=True,
                                                output_path=self.root / "show.png")

    def test_input_summary_cannot_be_overwritten_by_output(self):
        source = self.write(summary())
        before = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            visualize.create_visualizations(source, output_path=source)
        self.assertEqual(source.read_bytes(), before)

    def test_failed_render_closes_only_new_figures(self):
        source = self.write(summary())
        plt = visualize._pyplot(False)
        original = plt.figure()
        def fail_after_creating_figure(*args, **kwargs):
            plt.figure()
            raise RuntimeError("render failed")
        try:
            with patch.object(visualize, "_build_figure", side_effect=fail_after_creating_figure):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    visualize.create_visualizations(source, output_path=self.root / "failed.png")
            self.assertEqual(plt.get_fignums(), [original.number])
        finally:
            plt.close(original)


if __name__ == "__main__":
    unittest.main()
