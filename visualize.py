"""Render a static scientific dashboard from ecosystem simulation telemetry."""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib


EVENT_COLORS = {
    "Drought": "#d69a36", "ColdSnap": "#75a9cf",
    "Bloom": "#92af5b", "ToxicBloom": "#b57498",
}
COLORS = {"ink": "#24374b", "blue": "#276b96", "teal": "#29877d",
          "orange": "#c77836", "purple": "#876090", "muted": "#607082"}
NUMERIC_FIELDS = (
    "population", "herbivore_count", "predator_count", "species_count",
    "species_shannon", "effective_species", "inverse_simpson", "species_evenness",
    "mean_fitness", "max_fitness", "mean_habitat_match", "mean_resources",
    "mean_toxicity", "total_pheromone", "births", "deaths", "event_intensity",
)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def read_ticks(json_path):
    """Read legacy/current summaries, with actionable errors instead of exit()."""
    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ValueError(f"Cannot read simulation summary '{path}': {exc.strerror}.") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}.") from exc
    if not isinstance(data, dict):
        raise ValueError("Simulation summary must be a JSON object.")
    rows = data.get("ticks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Simulation summary needs a nonempty 'ticks' array.")
    previous = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"ticks[{index}] must be an object.")
        tick = _number(row.get("tick"), f"ticks[{index}].tick")
        if tick < 0 or tick <= previous:
            raise ValueError("Snapshot ticks must be nonnegative and strictly increasing.")
        previous = tick
        for key in NUMERIC_FIELDS:
            if key in row:
                _number(row[key], f"ticks[{index}].{key}")
        for key in ("mean_habitat_match", "species_evenness"):
            if key in row and not 0 <= row[key] <= 1:
                raise ValueError(f"ticks[{index}].{key} must be between 0 and 1.")
        biomes = row.get("biome_distribution", {})
        if not isinstance(biomes, dict):
            raise ValueError(f"ticks[{index}].biome_distribution must be an object.")
        for name, value in biomes.items():
            if not 0 <= _number(value, f"ticks[{index}].biome_distribution.{name}") <= 1:
                raise ValueError("Biome proportions must be between 0 and 1.")
    run, config = data.get("run", {}), data.get("config", {})
    if not isinstance(run, dict) or not isinstance(config, dict):
        raise ValueError("'run' and 'config' must be objects when provided.")
    completed = _number(run.get("ticks_completed", rows[-1]["tick"]), "run.ticks_completed")
    if completed < rows[-1]["tick"]:
        raise ValueError("run.ticks_completed cannot precede the final snapshot.")
    for obj, keys in ((run, ("final_population",)), (config, ("simulation_ticks",))):
        for key in keys:
            if key in obj:
                _number(obj[key], key)
    event_windows(data, rows)
    effective_species_values(rows)
    return data, rows


def event_windows(data, rows=None):
    """Get exact scheduled intervals clipped to the last observed tick.

    Explicit [] means no events. Legacy event labels can only approximate
    intervals, and can miss a complete shock between two snapshots.
    """
    rows = data["ticks"] if rows is None else rows
    final_tick = rows[-1]["tick"]
    windows = []
    if "event_windows" in data:
        schedule = data["event_windows"]
        if not isinstance(schedule, list):
            raise ValueError("event_windows must be an array.")
        for index, item in enumerate(schedule):
            if not isinstance(item, dict):
                raise ValueError(f"event_windows[{index}] must be an object.")
            name = item.get("type")
            if not isinstance(name, str) or not name:
                raise ValueError(f"event_windows[{index}].type must be a nonempty string.")
            start = _number(item.get("start_tick"), f"event_windows[{index}].start_tick")
            end = _number(item.get("end_tick_exclusive"), f"event_windows[{index}].end_tick_exclusive")
            strength = _number(item.get("strength", 1), f"event_windows[{index}].strength")
            if start < 0 or end <= start or strength < 0:
                raise ValueError("Event windows need 0 <= start < end and nonnegative strength.")
            if start < final_tick and name != "None":
                windows.append({"type": name, "start_tick": start,
                                "end_tick_exclusive": min(end, final_tick), "strength": strength})
        return windows
    start, name = None, "None"
    for row in rows:
        next_name = row.get("active_event", "None") or "None"
        if not isinstance(next_name, str):
            raise ValueError("Legacy active_event labels must be strings.")
        if next_name != name:
            if start is not None and row["tick"] > start:
                windows.append({"type": name, "start_tick": start,
                                "end_tick_exclusive": row["tick"], "strength": 1})
            start = row["tick"] if next_name != "None" else None
            name = next_name
    if start is not None and final_tick > start:
        windows.append({"type": name, "start_tick": start,
                        "end_tick_exclusive": final_tick, "strength": 1})
    return windows


def shade_events(ax, ticks_or_windows, event_names=None):
    """Shade exact windows; retain the old (axis, ticks, labels) call form."""
    windows = ticks_or_windows
    if event_names is not None:
        rows = [{"tick": tick, "active_event": name}
                for tick, name in zip(ticks_or_windows, event_names)]
        windows = event_windows({"ticks": rows}) if rows else []
    for item in windows:
        ax.axvspan(item["start_tick"], item["end_tick_exclusive"],
                   color=EVENT_COLORS.get(item["type"], "#9ba4ad"), alpha=0.09, linewidth=0, zorder=0)


def population_values(rows):
    return [row.get("population", row.get("herbivore_count", math.nan) + row.get("predator_count", math.nan))
            for row in rows]


def effective_species_values(rows):
    values = []
    for row in rows:
        if "effective_species" in row:
            values.append(row["effective_species"])
        elif row.get("species_count") == 0:
            values.append(0.0)
        elif "species_shannon" in row:
            try:
                values.append(math.exp(row["species_shannon"]))
            except OverflowError as exc:
                raise ValueError("species_shannon is too large to convert to effective species.") from exc
        else:
            values.append(math.nan)
    return values


def demographic_intervals(rows):
    """Place interval totals over (previous tick, current tick]."""
    centers, widths, births, deaths = [], [], [], []
    previous = 0
    for row in rows:
        tick = row["tick"]
        if tick > previous:
            centers.append((previous + tick) / 2)
            widths.append((tick - previous) * 0.88)
            births.append(row.get("births", math.nan))
            deaths.append(-row.get("deaths", math.nan))
        previous = tick
    return centers, widths, births, deaths


def _pyplot(show_plot):
    # Select the backend before importing pyplot, so --show is never silently
    # routed to the noninteractive Agg renderer.
    if show_plot:
        backend = os.environ.get("MPLBACKEND", "TkAgg")
        if backend.lower() in {"agg", "pdf", "svg", "ps", "pgf", "cairo", "template"}:
            raise ValueError("--show needs an interactive Matplotlib backend; unset MPLBACKEND or set it to TkAgg/QtAgg.")
        matplotlib.use(backend, force=True)
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _metadata(data, rows, label):
    run = data.get("run", {})
    final = run.get("final_population", population_values(rows)[-1])
    status = run.get("status", "extinct" if final == 0 else "legacy summary")
    completed = run.get("ticks_completed", rows[-1]["tick"])
    horizon = data.get("config", {}).get("simulation_ticks", completed)
    result = f"{label}  |  seed {data.get('seed', '?')}  |  {status}  |  ticks {completed:g}/{horizon:g}  |  final population {final:g}"
    if run.get("extinction_tick") is not None:
        result += f"  |  extinction at {run['extinction_tick']}"
    return result


def _plot_series(ax, rows, field, label, color, **kwargs):
    if any(field in row for row in rows):
        ax.plot([row["tick"] for row in rows], [row.get(field, math.nan) for row in rows],
                label=label, color=color, linewidth=1.8, **kwargs)


def _legend(ax, extra=None, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if extra is not None:
        others, other_labels = extra.get_legend_handles_labels()
        handles += others
        labels += other_labels
    if handles:
        ax.legend(handles, labels, frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.9, fontsize=9, **kwargs)


def _build_figure(plt, data, rows, compare_data=None, compare_rows=None):
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    style = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": COLORS["ink"],
             "axes.labelcolor": COLORS["muted"], "xtick.color": COLORS["muted"],
             "ytick.color": COLORS["muted"], "axes.edgecolor": "#ced5db", "axes.titleweight": "bold"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(16, 12.5))
        fig.patch.set_facecolor("#f7f9fb")
        fig.subplots_adjust(left=0.075, right=0.925, bottom=0.10, top=0.77, hspace=0.52, wspace=0.30)
        fig.text(0.075, 0.959, "ECOSYSTEM DYNAMICS", fontsize=23, weight="bold")
        fig.text(0.075, 0.929, _metadata(data, rows, "Run A"), fontsize=10)
        if compare_data is not None:
            fig.text(0.075, 0.905, _metadata(compare_data, compare_rows, "Run B"), fontsize=10, color=COLORS["purple"])
        else:
            fig.text(0.075, 0.905, "Population, habitat, resources and diversity across the simulated world", color=COLORS["muted"])
        windows = event_windows(data, rows)
        other_windows = event_windows(compare_data, compare_rows) if compare_data is not None else []
        maximum_tick = max(rows[-1]["tick"], compare_rows[-1]["tick"] if compare_rows else 0, 1)
        axis_ticks = [0] + [tick for tick in MaxNLocator(nbins=6, integer=True).tick_values(0, maximum_tick)
                            if 0 < tick < maximum_tick * 0.9] + [maximum_tick]
        for ax in axes.flat:
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e1e6eb", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0, maximum_tick)
            ax.set_xticks(axis_ticks)
            ax.set_xlabel("Simulation tick")
            shade_events(ax, windows)

        # One timeline carries exact A/B event timing without six text overlays.
        event_ax = fig.add_axes([0.075, 0.835, 0.85, 0.041])
        event_ax.set_facecolor("#edf1f5")
        event_ax.set_xlim(0, maximum_tick)
        event_ax.set_xticks(axis_ticks)
        event_ax.set_ylim(0, 2 if compare_data is not None else 1)
        groups = [windows, other_windows] if compare_data is not None else [windows]
        for index, group in enumerate(groups):
            y = 1.05 - index if compare_data is not None else 0.1
            for item in group:
                event_ax.broken_barh([(item["start_tick"], item["end_tick_exclusive"] - item["start_tick"])],
                                     (y, 0.8), facecolors=EVENT_COLORS.get(item["type"], "#9ba4ad"), alpha=0.85)
        event_ax.set_yticks([1.45, 0.45] if compare_data is not None else [0.5], ["A", "B"] if compare_data is not None else ["A"])
        event_ax.tick_params(axis="both", labelsize=8, length=0)
        for spine in event_ax.spines.values():
            spine.set_visible(False)
        names = list(dict.fromkeys(item["type"] for item in windows + other_windows))
        if names:
            event_ax.legend(handles=[Patch(facecolor=EVENT_COLORS.get(name, "#9ba4ad"), label=name) for name in names],
                            loc="lower right", bbox_to_anchor=(1, 1.04), ncol=min(4, len(names)), frameon=False, fontsize=9)
        legacy = "event_windows" not in data or (compare_data is not None and "event_windows" not in compare_data)
        event_ax.set_title("Disturbance schedule" + (" · legacy bands inferred from snapshots" if legacy else " · exact intervals"),
                           loc="left", fontsize=9, pad=3)
        ids = [row["tick"] for row in rows]

        ax = axes[0, 0]
        ax.set_title("01  Population", loc="left")
        ax.plot(ids, population_values(rows), color=COLORS["ink"], linewidth=2.3, label="A · total")
        _plot_series(ax, rows, "herbivore_count", "A · herbivores", COLORS["teal"])
        _plot_series(ax, rows, "predator_count", "A · predators", COLORS["orange"])
        if compare_rows:
            ax.plot([r["tick"] for r in compare_rows], population_values(compare_rows), color=COLORS["purple"],
                    linestyle="--", linewidth=2, label="B · total")
        ax.set_ylabel("Living agents")
        ax.set_ylim(bottom=0)
        _legend(ax, ncol=2)

        ax = axes[0, 1]
        ax.set_title("02  Fitness and habitat match", loc="left")
        _plot_series(ax, rows, "mean_fitness", "Mean fitness", COLORS["blue"])
        _plot_series(ax, rows, "max_fitness", "Best fitness", COLORS["blue"], linestyle=":", alpha=0.7)
        fit_ax = ax.twinx()
        _plot_series(fit_ax, rows, "mean_habitat_match", "Habitat match", COLORS["orange"])
        fit_ax.set_ylim(0, 1)
        fit_ax.set_ylabel("Habitat match [0–1]")
        fit_ax.spines["top"].set_visible(False)
        ax.set_ylabel("Fitness (model units)")
        _legend(ax, fit_ax, loc="upper right")

        ax = axes[1, 0]
        ax.set_title("03  World resources and toxicity", loc="left")
        _plot_series(ax, rows, "mean_resources", "A · resources", COLORS["teal"])
        _plot_series(ax, rows, "mean_toxicity", "A · toxicity", COLORS["orange"], linestyle=":")
        if compare_rows:
            _plot_series(ax, compare_rows, "mean_resources", "B · resources", COLORS["purple"], linestyle="--")
        ax.set_ylabel("Mean per cell (model units)")
        ax.set_ylim(bottom=0)
        _legend(ax)

        ax = axes[1, 1]
        ax.set_title("04  Habitat composition", loc="left")
        biomes = ["Ocean", "Tundra", "Desert", "Grassland", "Forest", "Jungle"]
        colors = ["#8eaec3", "#d8e2e7", "#dfbd76", "#b4c787", "#648e73", "#345f55"]
        if any("biome_distribution" in row for row in rows):
            histories = [[row.get("biome_distribution", {}).get(name, math.nan) * 100 for row in rows] for name in biomes]
            ax.stackplot(ids, histories, labels=biomes, colors=colors, alpha=0.95)
            _legend(ax, ncol=3, loc="lower center")
        else:
            ax.text(0.5, 0.5, "Biome telemetry unavailable", ha="center", transform=ax.transAxes, color=COLORS["muted"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Map area (%)")

        ax = axes[2, 0]
        ax.set_title("05  Species diversity", loc="left")
        _plot_series(ax, rows, "species_count", "A · richness", COLORS["ink"], alpha=0.65)
        ax.plot(ids, effective_species_values(rows), color=COLORS["blue"], linewidth=2, label="A · effective Shannon")
        _plot_series(ax, rows, "inverse_simpson", "A · inverse Simpson", COLORS["teal"], linestyle=":")
        if compare_rows:
            ax.plot([r["tick"] for r in compare_rows], effective_species_values(compare_rows), color=COLORS["purple"],
                    linewidth=2, linestyle="--", label="B · effective Shannon")
        ax.set_ylabel("Species / effective species")
        ax.set_ylim(bottom=0)
        _legend(ax, ncol=2)

        ax = axes[2, 1]
        ax.set_title("06  Births and deaths by interval", loc="left")
        centers, widths, births, deaths = demographic_intervals(rows)
        ax.bar(centers, births, width=widths, color=COLORS["blue"], alpha=0.85, label="Births")
        ax.bar(centers, deaths, width=widths, color=COLORS["orange"], alpha=0.85, label="Deaths (below zero)")
        ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
        ax.set_ylabel("Agents per snapshot interval")
        _legend(ax)
        note = "Bands in panels: Run A events. Species are model genome clusters. Values describe an uncalibrated simulation."
        if compare_data is not None:
            matching = data.get("seed") == compare_data.get("seed") and "seed" in data
            note += "\n" + ("Shared seed" if matching else "Different or unspecified seeds") + "; purple dashed lines = Run B. Individual runs do not establish a treatment effect."
        fig.text(0.075, 0.022, note, fontsize=9, color=COLORS["muted"], linespacing=1.6)
        return fig


def create_visualizations(json_path, show_plot=False, compare_path=None, output_path=None):
    """Save a dashboard and return its path; defaults preserve previous callers."""
    data, rows = read_ticks(json_path)
    compare_data, compare_rows = read_ticks(compare_path) if compare_path else (None, None)
    destination = Path(output_path) if output_path else Path("simulation_dashboard.png")
    if destination.resolve() in {Path(path).resolve() for path in (json_path, compare_path) if path}:
        raise ValueError("Dashboard output must not overwrite an input summary.")
    fig, plt, initial_figures = None, None, set()
    try:
        plt = _pyplot(show_plot)
        initial_figures = set(plt.get_fignums())
        fig = _build_figure(plt, data, rows, compare_data, compare_rows)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
        print(f"Visualization saved to {destination.resolve()}")
        if show_plot:
            plt.show(block=True)
    except (ImportError, RuntimeError) as exc:
        if show_plot:
            raise ValueError("Interactive display is unavailable. Install a GUI backend (Tk/Qt), set MPLBACKEND accordingly, or omit --show to save the image.") from exc
        raise
    finally:
        if plt is not None:
            for number in set(plt.get_fignums()) - initial_figures:
                plt.close(number)
    return destination.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", default="simulation_summary.json", help="Simulation JSON summary")
    parser.add_argument("--compare", metavar="SUMMARY", help="Overlay another run's population, resources and effective diversity")
    parser.add_argument("--output", metavar="PATH", help="Output image (default: simulation_dashboard.png)")
    parser.add_argument("--show", action="store_true", help="Open an interactive Matplotlib window after saving")
    args = parser.parse_args(argv)
    try:
        create_visualizations(args.summary, show_plot=args.show, compare_path=args.compare, output_path=args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
