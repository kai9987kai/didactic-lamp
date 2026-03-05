import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENT_COLORS = {
    "Drought": "#f4a261",
    "ColdSnap": "#8ecae6",
    "Bloom": "#90be6d",
    "ToxicBloom": "#b56576",
}


def shade_events(ax, ticks, event_names):
    start = None
    current_event = None

    for idx, event_name in enumerate(event_names):
        active = event_name and event_name != "None"
        if active and current_event is None:
            start = idx
            current_event = event_name
            continue

        if active and event_name == current_event:
            continue

        if current_event is not None:
            left = ticks[start]
            right = ticks[idx] if idx < len(ticks) else ticks[-1]
            ax.axvspan(left, right, color=EVENT_COLORS.get(current_event, "#cccccc"), alpha=0.10)
            ax.text(
                (left + right) / 2.0,
                0.96,
                current_event,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=EVENT_COLORS.get(current_event, "#555555"),
            )
            current_event = None
            start = None

        if active:
            start = idx
            current_event = event_name

    if current_event is not None and start is not None:
        left = ticks[start]
        right = ticks[-1]
        ax.axvspan(left, right, color=EVENT_COLORS.get(current_event, "#cccccc"), alpha=0.10)
        ax.text(
            (left + right) / 2.0,
            0.96,
            current_event,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=EVENT_COLORS.get(current_event, "#555555"),
        )


def read_ticks(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        print("Run the C++ simulation first to generate this file.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    ticks = data.get("ticks", [])
    if not ticks:
        print("Error: No tick data found in the JSON file.")
        sys.exit(1)

    return data, ticks


def create_visualizations(json_path, show_plot=False):
    _, ticks = read_ticks(json_path)
    tick_ids = [entry["tick"] for entry in ticks]
    event_names = [entry.get("active_event", "None") for entry in ticks]

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle("Universe Simulation telemetry", fontsize=17)

    ax = axes[0, 0]
    herbivores = [entry.get("herbivore_count", 0) for entry in ticks]
    predators = [entry.get("predator_count", 0) for entry in ticks]
    alive = [h + p for h, p in zip(herbivores, predators)]
    ax.plot(tick_ids, alive, color="#1d3557", linestyle="--", linewidth=2, label="Total alive")
    ax.plot(tick_ids, herbivores, color="#2a9d8f", linewidth=2, label="Herbivores")
    ax.plot(tick_ids, predators, color="#e76f51", linewidth=2, label="Predators")
    shade_events(ax, tick_ids, event_names)
    ax.set_title("Population Dynamics")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Agents")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    mean_fitness = [entry.get("mean_fitness", 0.0) for entry in ticks]
    best_fitness = [entry.get("max_fitness", value) for entry, value in zip(ticks, mean_fitness)]
    habitat_match = [entry.get("mean_habitat_match", 0.0) for entry in ticks]
    ax.plot(tick_ids, mean_fitness, color="#264653", linewidth=2, label="Mean fitness")
    ax.plot(tick_ids, best_fitness, color="#264653", linestyle="--", alpha=0.6, label="Best fitness")
    ax2 = ax.twinx()
    ax2.plot(tick_ids, habitat_match, color="#e9c46a", linewidth=2, label="Habitat match")
    shade_events(ax, tick_ids, event_names)
    ax.set_title("Fitness vs. Niche Fit")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Fitness")
    ax2.set_ylabel("Habitat match")
    ax.grid(True, alpha=0.25)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax = axes[1, 0]
    mean_resources = [entry.get("mean_resources", 0.0) for entry in ticks]
    mean_toxicity = [entry.get("mean_toxicity", 0.0) for entry in ticks]
    pheromones = [entry.get("total_pheromone", 0.0) for entry in ticks]
    ax.plot(tick_ids, mean_resources, color="#2a9d8f", linewidth=2, label="Mean resources")
    ax.plot(tick_ids, mean_toxicity, color="#b56576", linewidth=2, label="Mean toxicity")
    ax2 = ax.twinx()
    ax2.plot(tick_ids, pheromones, color="#6d597a", linestyle="--", linewidth=2, label="Total pheromone")
    shade_events(ax, tick_ids, event_names)
    ax.set_title("World State")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Resources / toxicity")
    ax2.set_ylabel("Pheromone")
    ax.grid(True, alpha=0.25)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax = axes[1, 1]
    biomes = ["Ocean", "Tundra", "Desert", "Grassland", "Forest", "Jungle"]
    biome_colors = ["#1d3557", "#a8dadc", "#e9c46a", "#90be6d", "#2a9d8f", "#1b4332"]
    biome_history = {name: [] for name in biomes}
    for entry in ticks:
        biome_dist = entry.get("biome_distribution", {})
        for biome_name in biomes:
            biome_history[biome_name].append(biome_dist.get(biome_name, 0.0) * 100.0)
    ax.stackplot(
        tick_ids,
        [biome_history[name] for name in biomes],
        labels=biomes,
        colors=biome_colors,
        alpha=0.88,
    )
    shade_events(ax, tick_ids, event_names)
    ax.set_title("Biome Distribution")
    ax.set_xlabel("Tick")
    ax.set_ylabel("% of map")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax = axes[2, 0]
    species_count = [entry.get("species_count", 0) for entry in ticks]
    diversity = [entry.get("species_shannon", 0.0) for entry in ticks]
    ax.plot(tick_ids, species_count, color="#1d3557", linewidth=2, label="Species count")
    ax2 = ax.twinx()
    ax2.plot(tick_ids, diversity, color="#e76f51", linewidth=2, linestyle="--", label="Shannon diversity")
    shade_events(ax, tick_ids, event_names)
    ax.set_title("Diversity")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Species count")
    ax2.set_ylabel("Shannon diversity")
    ax.grid(True, alpha=0.25)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax = axes[2, 1]
    births = [entry.get("births", 0) for entry in ticks]
    deaths = [entry.get("deaths", 0) for entry in ticks]
    event_intensity = [entry.get("event_intensity", 0.0) for entry in ticks]
    ax.bar(tick_ids, births, width=18, color="#2a9d8f", alpha=0.8, label="Births")
    ax.bar(tick_ids, [-value for value in deaths], width=18, color="#e76f51", alpha=0.7, label="Deaths")
    ax2 = ax.twinx()
    ax2.fill_between(tick_ids, event_intensity, color="#6d597a", alpha=0.25, label="Event intensity")
    shade_events(ax, tick_ids, event_names)
    ax.set_title("Demographic Churn")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Births / deaths")
    ax2.set_ylabel("Event intensity")
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.grid(True, alpha=0.25)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_file = "simulation_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    source_path = "simulation_summary.json"
    show_plot = False
    for arg in sys.argv[1:]:
        if arg == "--show":
            show_plot = True
        else:
            source_path = arg
    create_visualizations(source_path, show_plot=show_plot)
