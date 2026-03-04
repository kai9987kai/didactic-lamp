#include "agents.h"
#include "evolution.h"
#include "metrics.h"
#include "types.h"
#include "world.h"

#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  sim::Config cfg;
  if (argc > 1) cfg.agents = std::max(100, std::stoi(argv[1]));
  if (argc > 2) cfg.generations = std::max(1, std::stoi(argv[2]));
  if (argc > 3) cfg.steps_per_generation = std::max(1, std::stoi(argv[3]));
  if (argc > 4) cfg.softmax_temperature = std::max(0.05f, std::stof(argv[4]));

  std::cout << "=== Universe Simulation v2.0 ===\n";
  std::cout << "Agents: " << cfg.agents
            << "  Generations: " << cfg.generations
            << "  Steps/gen: " << cfg.steps_per_generation
            << "  Temp: " << cfg.softmax_temperature << "\n";
  std::cout << "Predator ratio: " << cfg.predator_ratio
            << "  Hunt prob: " << cfg.hunt_success_prob
            << "  Genome size: " << sim::kGenomeSize << "\n";
  std::cout << "Neural arch: " << sim::kInputFeatures << " -> "
            << sim::kHiddenNeurons << " -> " << sim::kActions << "\n\n";

  std::mt19937_64 rng(cfg.seed);
  sim::WorldFields world = sim::build_world(cfg);

  std::vector<sim::Agent> population;
  sim::init_population(population, cfg, rng);

  // Species tracker with centroid-based classification
  sim::SpeciesTracker species_tracker;

  std::vector<sim::Metrics> all_metrics;
  all_metrics.reserve(static_cast<size_t>(cfg.generations));

  // Adaptive speciation threshold
  float spec_threshold = cfg.speciation_threshold;

  for (int gen = 0; gen < cfg.generations; ++gen) {
    // Reset agents for new generation
    for (auto& a : population) {
      a.energy = 6.0f;
      a.fitness = 0.0f;
      a.novelty_score = 0.0f;
      a.age = 0;
      a.alive = true;
      a.kills = 0;
    }

    sim::reset_dynamic_fields(world);

    // Classify species at start of generation
    species_tracker.classify(population, spec_threshold, gen);

    // Run simulation steps
    for (int step = 0; step < cfg.steps_per_generation; ++step) {
      sim::update_climate_and_resources(world, cfg, step + gen * cfg.steps_per_generation);
      sim::step_agents(population, cfg, world, step, rng);
    }

    // Compute metrics
    sim::Metrics m = sim::compute_metrics(population, world, cfg);
    m.extinction_events = species_tracker.count_extinctions(gen);
    m.speciation_events = species_tracker.count_speciations(gen);
    all_metrics.push_back(m);

    // Console output
    std::cout << "gen=" << gen << std::fixed << std::setprecision(4)
              << " mean=" << m.mean_fitness
              << " best=" << m.best_fitness
              << " novelty=" << m.mean_novelty
              << " H=" << m.diversity_shannon
              << " | species=" << m.species_count
              << " herb=" << m.herbivore_count
              << " pred=" << m.predator_count
              << " alive=" << m.alive_count
              << " age=" << m.mean_age
              << " pheromone=" << m.total_pheromone
              << "\n";

    // Adaptive radiation: tighten speciation threshold over time
    spec_threshold *= 0.98f;
    spec_threshold = std::max(spec_threshold, 0.3f);

    if (gen + 1 < cfg.generations) {
      sim::evolve(population, cfg, rng);
    }
  }

  // Write output
  std::ofstream out("simulation_summary.json", std::ios::binary);
  out << sim::summary_json(cfg, all_metrics, species_tracker.records);
  out.close();
  std::cout << "\nWrote simulation_summary.json ("
            << species_tracker.records.size() << " species tracked)\n";

  return 0;
}
